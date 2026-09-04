"""Segmentation losses used by the box-guided decoder."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits).flatten(2)
    gt = target.flatten(2)
    intersection = (probability * gt).sum(-1)
    union = probability.sum(-1) + gt.sum(-1) + eps
    return (1.0 - (2.0 * intersection + eps) / union).mean()


def _iou_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        probability = (torch.sigmoid(logits) > 0.5).float().flatten(2)
        gt = target.flatten(2)
        intersection = (probability * gt).sum(-1)
        union = probability.sum(-1) + gt.sum(-1) - intersection
        iou = torch.where(union > 0, intersection / union.clamp_min(1e-6), torch.zeros_like(intersection))
    return iou


def segmentation_losses(
    mask_logits: torch.Tensor,
    iou_scores: torch.Tensor,
    gt_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute BCE, Dice and IoU-head MSE on stride-8 ground truth.

    All tensors have shape (B, N, H, W) / (B, N) and contain no padding.
    """
    logits_float = mask_logits.float()
    gt_float = gt_masks.float()
    bce = F.binary_cross_entropy_with_logits(logits_float, gt_float)
    dice = _dice_loss(logits_float, gt_float)
    iou_mse = F.mse_loss(iou_scores.float(), _iou_target(logits_float, gt_float))
    return {
        "bce_loss": bce,
        "dice_loss": dice,
        "iou_mse_loss": iou_mse,
        "seg_loss": bce + dice,
    }
