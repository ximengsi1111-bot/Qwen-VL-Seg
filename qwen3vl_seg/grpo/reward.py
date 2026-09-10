"""GRPO reward functions for Qwen3-VL-Seg: box IoU + mask IoU.

`completions` are the assistant (generated) texts; ``solution`` is each dataset
item's gold response (JSON) carrying GT.  Custom dataset columns
(``image_path``/``mask_paths``/``bboxes``/``image_size``) and ``messages``
flow into ``**kwargs`` via ``to_reward_row``.

Mask term is aligned to the formal eval (runner_multi.py) semantics:
  - Hungarian optimal instance matching (NOT index pairing)
  - per-GT-instance recall aggregation (sum of matched IoU / n_gt), so missing
    GT instances are penalised
  - identical image/mask resolution as the eval (``GRPO_MAX_PIXELS``, default 262144)
  - base model = GRPO policy base, and an optional saved adapter (``GRPO_ADAPTER``)
    can be merged for OFFLINE reward evaluation on a checkpoint.
NOTE: ms-swift's ORM reward does not receive the in-training model, so the live
GRPO reward cannot see the training-LoRA; ``GRPO_ADAPTER`` only applies to
post-hoc/offline reward checks on a saved adapter.
"""
from __future__ import annotations

import json
import os
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


_STAGE2_CKPT = os.environ.get("GRPO_STAGE2", "/file_storage01/home/mingli/data/xyk/checkpoints/small/stage2-100k")
# Base for the seg wrapper = the GRPO policy base (SFT policy). The stage2 mask
# decoder is loaded on top via _load_stage2.
_BASE_MODEL = os.environ.get("GRPO_BASE", "/mingli01/data/xyk/model/grpo-policy")
# Optional adapter to merge for OFFLINE reward evaluation (see module docstring).
_GRPO_ADAPTER = os.environ.get("GRPO_ADAPTER", "")
# Same image/mask resolution as the formal eval (runner_multi --max-pixels).
_MAX_PIXELS = int(os.environ.get("GRPO_MAX_PIXELS", "262144"))


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
                _BASE_MODEL, torch_dtype=torch.bfloat16
            )
            _load_stage2(wrapper, _STAGE2_CKPT)
            if _GRPO_ADAPTER:
                from peft import PeftModel
                peft_model = PeftModel.from_pretrained(wrapper.base, _GRPO_ADAPTER)
                wrapper.base = peft_model.merge_and_unload()
                print(f"[SegIoUReward] merged adapter: {_GRPO_ADAPTER}", flush=True)
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
        """Run the frozen decoder to get a predicted mask, scored like the formal eval.

        Uses Hungarian instance matching and per-GT recall aggregation, matching
        the per-sample miou definition in ``runner_multi.py`` / ``metrics.py``.
        """
        from qwen3vl_seg.model.prompt_format import (
            MASK_START, MASK_TOKEN, MASK_END,
        )
        from qwen3vl_seg.eval.runner import _inject_mask_placeholders, parse_generated_boxes_and_masks
        from qwen3vl_seg.data.dataset import normalize_decoder_image
        from qwen3vl_seg.eval.metrics import match_masks

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
            processor_kwargs={"images_kwargs": {"max_pixels": _MAX_PIXELS, "min_pixels": 56 * 56}},
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

        # GT masks -> stride size (ALL GT instances, so missing GT is penalised)
        gt_masks = []
        for mask_rel in gt.get("mask_paths", []):
            with Image.open(mask_rel) as mh:
                m = mh.convert("L").resize((stride_w, stride_h), Image.BILINEAR)
            gt_masks.append(np.asarray(m, dtype=np.uint8))
        if not gt_masks:
            return 0.0

        # Hungarian optimal assignment + per-GT-recall aggregation (same as eval).
        pred_list = [preds[i] for i in range(len(preds))]
        gt_list = [np.asarray(g, dtype=np.uint8) for g in gt_masks]
        pred_idx, gt_idx, total = match_masks(pred_list, gt_list)
        return float(total / max(len(gt_list), 1))
