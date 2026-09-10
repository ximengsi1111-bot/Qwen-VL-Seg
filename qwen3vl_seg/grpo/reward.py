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
import torch.nn.functional as F

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
_REWARD_BATCH = os.environ.get("GRPO_REWARD_BATCH", "1") != "0"


class SegIoUReward(ORM):
    """Box IoU + mask IoU reward for referring segmentation GRPO."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.box_weight = float(kwargs.get("box_weight", 0.5))
        self.mask_weight = float(kwargs.get("mask_weight", 0.5))
        self._wrapper = None
        self._batch_warned = False

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
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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
        messages = kwargs.get("messages")
        n = len(completions)
        gts = [self._parse_solution(sol) for sol in solution]
        pred_boxes = [self._parse_completion(comp) for comp in completions]
        box_ious = []
        for i in range(n):
            gt_boxes = gts[i].get("bbox_1000", [])
            preds = pred_boxes[i]
            box_ious.append(max(_box_iou(p, g) for p in preds for g in gt_boxes) if gt_boxes and preds else 0.0)

        mask_ious = [0.0] * n
        if self.mask_weight > 0.0:
            msgs_list = [self._messages_at(messages, i) for i in range(n)]
            if _REWARD_BATCH and n > 1:
                try:
                    mask_ious = self._mask_iou_batch(completions, gts, msgs_list)
                except Exception as exc:
                    if not self._batch_warned:
                        print(f"[SegIoUReward] batch mask decode failed, falling back: {exc}", flush=True)
                        self._batch_warned = True
                    mask_ious = [0.0] * n
                    for i in range(n):
                        try:
                            mask_ious[i] = self._mask_iou(completions[i], gts[i], msgs_list[i])
                        except Exception:
                            mask_ious[i] = 0.0
            else:
                for i in range(n):
                    try:
                        mask_ious[i] = self._mask_iou(completions[i], gts[i], msgs_list[i])
                    except Exception:
                        mask_ious[i] = 0.0

        return [self.box_weight * box_ious[i] + self.mask_weight * mask_ious[i] for i in range(n)]

    def _messages_at(self, messages, i):
        if messages is None:
            return None
        try:
            return messages[i]
        except Exception:
            return None

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

    def _mask_iou_batch(self, comps, gts, msgs_list):
        """Batch the frozen decoder forward for completions sharing image/GT."""
        from collections import defaultdict
        from qwen3vl_seg.model.prompt_format import MASK_TOKEN
        from qwen3vl_seg.eval.runner import _inject_mask_placeholders, parse_generated_boxes_and_masks
        from qwen3vl_seg.data.dataset import normalize_decoder_image

        wrapper = self._get_wrapper()
        processor = wrapper.processor
        device = wrapper.base.device
        prepared = []
        out = [0.0] * len(comps)

        for i, (comp, gt, sample_messages) in enumerate(zip(comps, gts, msgs_list)):
            records = parse_generated_boxes_and_masks(comp)
            if not records:
                continue
            decoder_text = comp
            if comp.count(MASK_TOKEN) == 0:
                decoder_text = _inject_mask_placeholders(comp)
            if decoder_text.count(MASK_TOKEN) != len(records):
                continue

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

            enc = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_tensors="pt", return_dict=True,
                processor_kwargs={"images_kwargs": {"max_pixels": _MAX_PIXELS, "min_pixels": 56 * 56}},
            )
            enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
            grid = enc["image_grid_thw"][0]
            grid_h, grid_w = int(grid[1]), int(grid[2])
            boxes = torch.tensor(
                [[v / 1000.0 for v in r["bbox_2d"]] for r in records],
                dtype=torch.float32,
            ).to(device)
            image_tensor = normalize_decoder_image(
                Image.open(gt["image_path"]).convert("RGB"), grid_h * 16, grid_w * 16
            ).unsqueeze(0).to(device)
            key = (str(gt.get("image_path", "")), tuple(gt.get("mask_paths", [])), grid_h, grid_w)
            prepared.append({
                "idx": i,
                "key": key,
                "gt": gt,
                "comp": comp,
                "msgs": sample_messages,
                "enc": enc,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "boxes": boxes,
                "image": image_tensor,
            })

        groups = defaultdict(list)
        for item in prepared:
            groups[item["key"]].append(item)

        for group in groups.values():
            try:
                scores = self._score_mask_group(group)
            except Exception as exc:
                if not self._batch_warned:
                    print(f"[SegIoUReward] batch group decode failed, falling back: {exc}", flush=True)
                    self._batch_warned = True
                scores = {}
                for item in group:
                    try:
                        scores[item["idx"]] = self._mask_iou(item["comp"], item["gt"], item["msgs"])
                    except Exception:
                        scores[item["idx"]] = 0.0
            for idx, score in scores.items():
                out[idx] = float(score)
        return out

    def _load_gt_masks(self, gt, stride_h, stride_w):
        gt_masks = []
        for mask_rel in gt.get("mask_paths", []):
            with Image.open(mask_rel) as mh:
                m = mh.convert("L").resize((stride_w, stride_h), Image.BILINEAR)
            gt_masks.append(np.asarray(m, dtype=np.uint8))
        return gt_masks

    def _score_mask_group(self, group):
        from qwen3vl_seg.eval.metrics import match_masks

        wrapper = self._get_wrapper()
        device = wrapper.base.device
        encs = [item["enc"] for item in group]
        max_len = max(e["input_ids"].shape[1] for e in encs)

        def _pad(key_, padval):
            return torch.stack([
                F.pad(e[key_].detach().squeeze(0), (max_len - e[key_].shape[1], 0), value=padval)
                for e in encs
            ], dim=0).to(device)

        input_ids = _pad("input_ids", 0)
        attention_mask = _pad("attention_mask", 0)
        mm_token_type_ids = _pad("mm_token_type_ids", 0)
        pixel_values = torch.cat([e["pixel_values"].detach() for e in encs], dim=0).to(device)
        image_grid_thw = torch.stack([e["image_grid_thw"][0].detach() for e in encs], dim=0).to(device)

        max_inst = max(item["boxes"].shape[0] for item in group)
        boxes_list = []
        masks_list = []
        image_list = []
        num_instances = []
        for item in group:
            num = item["boxes"].shape[0]
            boxes = item["boxes"]
            if num < max_inst:
                boxes = torch.cat([
                    boxes,
                    torch.zeros(max_inst - num, 4, dtype=boxes.dtype, device=device),
                ], dim=0)
            boxes_list.append(boxes)
            masks_list.append(torch.zeros(
                num, item["grid_h"] * 2, item["grid_w"] * 2,
                dtype=torch.float32, device=device,
            ))
            image_list.append(item["image"].squeeze(0))
            num_instances.append(num)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "boxes": torch.stack(boxes_list, dim=0),
            "masks": masks_list,
            "num_instances": num_instances,
            "has_seg": [True] * len(group),
            "image": image_list,
        }
        with torch.no_grad():
            out_dec = wrapper(batch)

        logits_all = out_dec["mask_logits"]
        stride_h = int(group[0]["grid_h"]) * 2
        stride_w = int(group[0]["grid_w"]) * 2
        gt_masks = self._load_gt_masks(group[0]["gt"], stride_h, stride_w)
        if not gt_masks:
            return {item["idx"]: 0.0 for item in group}

        gt_list = [np.asarray(g, dtype=np.uint8) for g in gt_masks]
        scores = {}
        for j, item in enumerate(group):
            num = item["boxes"].shape[0]
            preds = (torch.sigmoid(logits_all[j][:num]) > 0.5).float().cpu().numpy()
            pred_list = [preds[k] for k in range(num)]
            _, _, total = match_masks(pred_list, gt_list)
            scores[item["idx"]] = float(total / max(len(gt_list), 1))
        return scores
