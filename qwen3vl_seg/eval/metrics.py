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


def _mask_iou_matrix(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]
) -> np.ndarray:
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=float)
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            iou_matrix[i, j] = mask_iou(pred, gt)
    return iou_matrix


def match_mask_ious(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-to-one assignment indices and IoU for each matched pair."""
    if not pred_masks or not gt_masks:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )
    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    return pred_idx, gt_idx, iou_matrix[pred_idx, gt_idx]


def match_masks(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pred_indices, gt_indices, total_iou) from a Hungarian match."""
    pred_idx, gt_idx, ious = match_mask_ious(pred_masks, gt_masks)
    return pred_idx, gt_idx, float(ious.sum()) if ious.size else 0.0


def instance_metrics(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute one-to-one instance metrics without hiding missed GT instances.

    ``miou`` is the matched IoU sum divided by the number of GT instances, so
    missed GT instances contribute zero. ``fp`` counts every unmatched
    prediction, including a matched pair whose IoU is below ``threshold``.
    """
    pred_idx, gt_idx, ious = match_mask_ious(pred_masks, gt_masks)
    iou_values = [float(v) for v in ious]
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    tp = int(np.count_nonzero(ious >= threshold)) if ious.size else 0
    fp = int(n_pred - tp)
    fn = int(n_gt - tp)
    denom_pr = tp + fp
    denom_rc = tp + fn
    precision = float(tp / denom_pr) if denom_pr else 0.0
    recall = float(tp / denom_rc) if denom_rc else 0.0
    f1 = (
        float(2.0 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn)
        else 0.0
    )
    miou = float(ious.sum() / n_gt) if n_gt else 0.0
    matched_p = float(tp / len(iou_values)) if iou_values else 0.0
    return {
        "pred_indices": pred_idx.tolist(),
        "gt_indices": gt_idx.tolist(),
        "ious": iou_values,
        "miou": miou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_p": matched_p,
        "threshold": float(threshold),
    }


def union_ciou(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]
) -> float:
    """IoU between the union of all predictions and the union of all GT masks."""
    if not pred_masks or not gt_masks:
        return 0.0
    pred_union = np.logical_or.reduce(
        [np.asarray(mask) > 0 for mask in pred_masks]
    )
    gt_union = np.logical_or.reduce(
        [np.asarray(mask) > 0 for mask in gt_masks]
    )
    return mask_iou(pred_union, gt_union)


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
