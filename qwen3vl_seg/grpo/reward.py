"""GRPO reward functions for Qwen3-VL-Seg: box IoU + mask IoU.

Usage with ms-swift ``swift rlhf`` (GRPO): pass this ORM via a callable/
registered reward function. ``completions`` are the assistant (generated)
texts; ``solution`` is each dataset item's gold response (a JSON string) with::

    {
      "bbox_1000": [[x1,y1,x2,y2], ...],   # GT boxes in 0..1000 normalized coords
      "image_path": "datasets/raw/...jpg", # for mask IoU (relative to data_root)
      "mask_paths": ["datasets/small/public/masks/mask_xxx.png", ...],
      "image_size": [h, w]
    }
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from swift.rewards.orm import ORM


def _box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes in the same coordinate frame."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0.0 else 0.0


class SegIoUReward(ORM):
    """Box IoU + mask IoU reward for referring segmentation GRPO."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.box_weight = float(kwargs.get("box_weight", 0.5))
        self.mask_weight = float(kwargs.get("mask_weight", 0.5))
        self._decoder = None  # lazy-loaded mask decoder module (for mask IoU)

    # --- parsing helpers -------------------------------------------------
    def _parse_completion(self, text: str) -> list[list[float]]:
        """Parse assistant text into a list of [x1,y1,x2,y2] boxes (0..1000)."""
        from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks
        try:
            records = parse_generated_boxes_and_masks(text)
        except Exception:
            return []
        boxes: list[list[float]] = []
        for rec in records:
            b = rec.get("bbox_2d")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append([float(v) for v in b])
        return boxes

    def _parse_solution(self, sol: Any) -> dict[str, Any]:
        if isinstance(sol, dict):
            return sol
        if not sol:
            return {}
        try:
            return json.loads(sol)
        except Exception:
            return {}

    # --- reward ----------------------------------------------------------
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        rewards: list[float] = []
        for comp, sol in zip(completions, solution):
            gt = self._parse_solution(sol)
            pred_boxes = self._parse_completion(comp)
            gt_boxes = gt.get("bbox_1000", [])
            box_iou = 0.0
            if gt_boxes and pred_boxes:
                box_iou = max(
                    _box_iou(p, g) for p in pred_boxes for g in gt_boxes
                )
            mask_iou = 0.0
            if self.mask_weight > 0.0:
                try:
                    mask_iou = self._mask_iou(comp, gt)
                except Exception:
                    mask_iou = 0.0
            rewards.append(self.box_weight * box_iou + self.mask_weight * mask_iou)
        return rewards

    def _mask_iou(self, comp: str, gt: dict[str, Any]) -> float:
        """Compute mask IoU using the frozen box-guided decoder.

        Reuses the eval decode path. ``gt`` must carry image_path / mask_paths /
        image_size. The decoder/model is loaded lazily via ``_get_decoder``.
        """
        raise NotImplementedError(
            "mask IoU needs the frozen decoder; wired in a later step"
        )