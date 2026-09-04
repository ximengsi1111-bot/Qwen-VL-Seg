"""Collation helpers for the segmentation data loader."""

from __future__ import annotations

from typing import Any


class SegmentationCollator:
    """Collate exactly one sample per micro-batch.

    Qwen3-VL image tokens vary with the image grid, and the box-guided decoder
    currently runs per image. Training therefore uses ``micro_batch_size=1`` and
    the collator returns that single processed sample unchanged.
    """

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError(
                f"SegmentationCollator expects one sample per micro-batch, got {len(batch)}. "
                "Configure micro_batch_size=1 for the segmentation dataloader."
            )
        return batch[0]
