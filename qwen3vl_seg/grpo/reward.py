"""GRPO reward functions for Qwen3-VL-Seg: box IoU + mask IoU.

`completions` are the assistant (generated) texts; ``solution`` is each dataset
item's gold response (JSON) carrying GT.  Custom dataset columns
(``image_path``/``mask_paths``/``bboxes``/``image_size``) and ``messages``
flow into ``**kwargs`` via ``to_reward_row``.
"""
from __future__ import annotations

import json
import numpy as np
from typing import Any
from PIL import Image
import torch

from swift.rewards.orm import ORM


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


_STAGE2_CKPT = "/file_storage01/home/mingli/data/xyk/checkpoints/small/stage2-100k"


class SegIoUReward(ORM):
    """Box IoU + mask IoU reward for referring segmentation GRPO."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.box_weight = float(kwargs.get("box_weight", 0.5))
        self.mask_weight = float(kwargs.get("mask_weight", 0.5))
        self._wrapper = None

    # ---- lazy model (base + decoder) for mask IoU ---------------
    def _get_wrapper(self):
        if self._wrapper is None:
            from qwen3vl_seg.model.model_wrapper import Qwen3VLSegForSegmentation
            from qwen3vl_seg.eval.runner import _load_stage2
            wrapper = Qwen3VLSegForSegmentation.from_pretrained(
                "/mingli01/models/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16
            )
            _load_stage2(wrapper, _STAGE2_CKPT)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            wrapper.to(device, dtype=torch.bfloat16)
            wrapper.eval()
            self._wrapper = wrapper
        return self._wrapper

    # ---- parse ---------------------------------------------------
    def _parse_completion(self, text):
        from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks
        try:
            records = parse_generated_boxes_and_masks(text)
        except Exception:
            return []
        boxes = []
        for r in records:
            b = r.get("bbox_2d")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append([float(v) for v in b])
        return boxes

    def _parse_solution(self, sol):
        if isinstance(sol, dict):
            return sol
        if not sol:
            return {}
        try:
            return json.loads(sol)
        except Exception:
            return {}

    # ---- reward --------------------------------------------------
    def __call__(self, completions, solution, **kwargs):
        rewards = []
        messages = kwargs.get("messages")
        for i, (comp, sol) in enumerate(zip(completions, solution)):
            gt = self._parse_solution(sol)
            pred_boxes = self._parse_completion(comp)
            gt_boxes = gt.get("bbox_1000", [])
            box_iou = 0.0
            if gt_boxes and pred_boxes:
                box_iou = max(_box_iou(p, g) for p in pred_boxes for g in gt_boxes)
            mask_iou = 0.0
            if self.mask_weight > 0.0:
                try:
                    msgs = messages[i] if messages is not None else None
                    mask_iou = self._mask_iou(comp, gt, msgs)
                except Exception:
                    mask_iou = 0.0
            rewards.append(self.box_weight * box_iou + self.mask_weight * mask_iou)
        return rewards

    def _mask_iou(self, comp, gt, sample_messages):
        """Run the frozen decoder to get a predicted mask, IoU vs GT mask."""
        from qwen3vl_seg.model.prompt_format import (
            MASK_START, MASK_TOKEN, MASK_END,
        )
        from qwen3vl_seg.eval.runner import _inject_mask_placeholders, parse_generated_boxes_and_masks
        from qwen3vl_seg.data.dataset import normalize_decoder_image
        from qwen3vl_seg.eval.metrics import mask_iou

        records = parse_generated_boxes_and_masks(comp)
        if not records:
            return 0.0
        decoder_text = comp
        if comp.count(MASK_TOKEN) == 0:
            decoder_text = _inject_mask_placeholders(comp)
        if decoder_text.count(MASK_TOKEN) != len(records):
            return 0.0

        wrapper = self._get_wrapper()
        processor = wrapper.processor
        device = wrapper.base.device

        # rebuild messages: user (prompt+image) + assistant (decoder_text)
        if sample_messages:
            msgs = [dict(message) for message in sample_messages]
            for m in msgs:
                if m.get("role") == "assistant":
                    m["content"] = decoder_text
                if m.get("role") == "user" and isinstance(m.get("content"), list):
                    for c in m["content"]:
                        if c.get("type") == "image":
                            c["image"] = Image.open(gt["image_path"]).convert("RGB")
        else:
            user_text = "Locate and segment the object, report bbox and mask in JSON."
            msgs = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(gt["image_path"]).convert("RGB")},
                    {"type": "text", "text": user_text},
                ],
            }, {"role": "assistant", "content": decoder_text}]

        full = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            return_tensors="pt", return_dict=True,
            processor_kwargs={"images_kwargs": {"max_pixels": 1048576, "min_pixels": 56 * 56}},
        )
        full = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full.items()}
        grid = full["image_grid_thw"][0]
        grid_h, grid_w = int(grid[1]), int(grid[2])
        stride_h, stride_w = grid_h * 2, grid_w * 2
        num = len(records)
        boxes = torch.tensor([[v / 1000.0 for v in r["bbox_2d"]] for r in records], dtype=torch.float32).to(device)
        masks = torch.zeros(num, stride_h, stride_w, dtype=torch.float32).to(device)
        image_tensor = normalize_decoder_image(
            Image.open(gt["image_path"]).convert("RGB"), grid_h * 16, grid_w * 16
        ).unsqueeze(0).to(device)
        batch = {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "mm_token_type_ids": full["mm_token_type_ids"],
            "pixel_values": full["pixel_values"],
            "image_grid_thw": full["image_grid_thw"],
            "boxes": boxes.unsqueeze(0),
            "masks": masks.unsqueeze(0),
            "num_instances": [num],
            "has_seg": True,
            "image": image_tensor,
        }
        with torch.no_grad():
            out_dec = wrapper(batch)
        logits = out_dec["mask_logits"][0]
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

        # GT masks -> stride size
        gt_masks = []
        for mask_rel in gt.get("mask_paths", [])[:num]:
            with Image.open(mask_rel) as mh:
                m = mh.convert("L").resize((stride_w, stride_h), Image.BILINEAR)
            gt_masks.append(np.asarray(m, dtype=np.uint8))
        if not gt_masks:
            return 0.0
        ious = []
        for pj in range(len(preds)):
            if pj < len(gt_masks):
                ious.append(mask_iou(np.expand_dims(preds[pj], 0), np.expand_dims(gt_masks[pj], 0)))
        return float(np.mean(ious)) if ious else 0.0