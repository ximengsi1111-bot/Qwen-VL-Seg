"""Collation helpers for the segmentation data loader."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class SegmentationCollator:
    """Collate one or more same-bucket samples into one micro-batch.

    With bucket padding every sample in a batch has the same ``image_grid_thw``
    and the same number of image patches, so pixel_value sequences can be
    concatenated. Text is padded; masks and decoder images stay as lists because
    the crop/stride grid can differ per instance count.
    """

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) == 1:
            return batch[0]

        def pad_seq(key: str, value: int) -> torch.Tensor:
            return torch.stack(
                [
                    F.pad(
                        item[key].squeeze(0) if item[key].dim() == 2 else item[key],
                        (0, max_seq - item[key].shape[-1]),
                        value=value,
                    )
                    for item in batch
                ],
                dim=0,
            )

        max_seq = max(item["input_ids"].shape[-1] for item in batch)
        max_inst = max(item["boxes"].shape[0] for item in batch)
        boxes = []
        for item in batch:
            box = item["boxes"]
            if box.shape[0] < max_inst:
                pad = torch.zeros(
                    max_inst - box.shape[0], box.shape[1], dtype=box.dtype
                )
                box = torch.cat([box, pad], dim=0)
            boxes.append(box)

        return {
            "sample_id": [item["sample_id"] for item in batch],
            "split": [item["split"] for item in batch],
            "kind": [item["kind"] for item in batch],
            "expression": [item["expression"] for item in batch],
            "label": [item["label"] for item in batch],
            "image_path": [item["image_path"] for item in batch],
            "mask_paths": [item["mask_paths"] for item in batch],
            "orig_size": [item["orig_size"] for item in batch],
            "bucket": [item.get("bucket") for item in batch],
            "pad_right": [item.get("pad_right", 0) for item in batch],
            "pad_bottom": [item.get("pad_bottom", 0) for item in batch],
            "input_ids": pad_seq("input_ids", self.pad_token_id),
            "attention_mask": pad_seq("attention_mask", 0),
            "mm_token_type_ids": pad_seq("mm_token_type_ids", 0),
            "labels": pad_seq("labels", -100),
            "pixel_values": torch.cat(
                [item["pixel_values"] for item in batch], dim=0
            ),
            "image_grid_thw": torch.cat(
                [item["image_grid_thw"] for item in batch], dim=0
            ),
            "image": [item["image"] for item in batch],
            "image_grid": [item["image_grid"] for item in batch],
            "boxes": torch.stack(boxes, dim=0),
            "masks": [item["masks"] for item in batch],
            "num_instances": [int(item["num_instances"]) for item in batch],
            "has_seg": [bool(item["has_seg"]) for item in batch],
        }
