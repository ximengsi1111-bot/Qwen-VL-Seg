from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    inter = float(np.logical_and(pred_b, gt_b).sum())
    union = float(np.logical_or(pred_b, gt_b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def bbox_iou(pred: Iterable[float], gt: Iterable[float]) -> float:
    px1, py1, px2, py2 = (float(v) for v in pred)
    gx1, gy1, gx2, gy2 = (float(v) for v in gt)
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_masks(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pred_indices, gt_indices, total_iou) from a Hungarian match."""
    if not pred_masks or not gt_masks:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            iou_matrix[i, j] = mask_iou(pred, gt)
    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    total = float(iou_matrix[pred_idx, gt_idx].sum())
    return pred_idx, gt_idx, total


def aggregate_instance_iou(
    per_gt_iou: list[float], thresholds: Iterable[float] = (0.5, 0.7, 0.9)
) -> dict[str, float]:
    arr = np.asarray(per_gt_iou, dtype=float)
    metrics: dict[str, float] = {"miou": float(arr.mean()) if arr.size else 0.0}
    for t in thresholds:
        metrics[f"p@{t:g}"] = float(np.mean(arr >= t)) if arr.size else 0.0
    return metrics


def aggregate_ciou(matched_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    inter = sum(float(np.logical_and(p > 0, g > 0).sum()) for p, g in matched_pairs)
    union = sum(float(np.logical_or(p > 0, g > 0).sum()) for p, g in matched_pairs)
    if union == 0:
        return 0.0
    return inter / union

