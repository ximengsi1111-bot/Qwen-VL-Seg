import torch

from qwen3vl_seg.model.losses import segmentation_losses


def test_segmentation_losses_finite_and_weighted() -> None:
    logits = torch.randn(1, 2, 8, 8)
    iou = torch.rand(1, 2)
    target = (torch.rand(1, 2, 8, 8) > 0.5).float()
    losses = segmentation_losses(logits, iou, target)
    for value in losses.values():
        assert torch.isfinite(value)
        assert value.ndim == 0
    assert torch.allclose(
        losses["seg_loss"],
        losses["bce_loss"] + losses["dice_loss"],
        atol=1e-6,
    )


def test_perfect_mask_loss_goes_to_zero() -> None:
    target = torch.ones(1, 1, 8, 8)
    logits = torch.full_like(target, 20.0)
    losses = segmentation_losses(logits, torch.ones(1, 1), target)
    assert losses["seg_loss"] < 0.1
