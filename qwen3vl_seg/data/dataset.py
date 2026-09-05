"""Training dataset that turns samples.jsonl into model-ready batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from qwen3vl_seg.data.buckets import (
    choose_bucket,
    pad_mask_canvas,
    pad_right_bottom,
)
from qwen3vl_seg.data.schema import parse_sample
from qwen3vl_seg.model.prompt_format import (
    MASK_TOKEN,
    build_target_text,
    build_user_prompt,
    ensure_seg_special_tokens,
)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_decoder_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((width, height), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
    tensor = torch.from_numpy(array) / 255.0
    mean = torch.tensor(_CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(_CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _mask_grid_tensor(mask: Image.Image, height: int, width: int) -> torch.Tensor:
    resized = mask.convert("L").resize((width, height), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return torch.from_numpy((array > 127.5).astype(np.float32))


class SegmentationDataset(Dataset):
    """One sample per row; returns tensors for a single image/text sequence.

    Qwen3-VL images are dynamically resized by the processor, so the mask grid
    is aligned with the decoder output by deriving the resized image size from
    ``image_grid_thw`` instead of guessing a fixed resolution.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        processor: Any | None = None,
        split: str | None = None,
        limit: int | None = None,
        max_pixels: int = 589824,
        min_pixels: int | None = None,
        ignore_mask_token_in_ce: bool = True,
        use_bucket: bool = False,
        bucket_sizes: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        if processor is None:
            raise ValueError("SegmentationDataset requires a Qwen3-VL processor.")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        ensure_seg_special_tokens(self.tokenizer)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels or 56 * 56
        self.ignore_mask_token_in_ce = ignore_mask_token_in_ce
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        self.use_bucket = bool(use_bucket)
        self.bucket_sizes = tuple(bucket_sizes) if bucket_sizes else None

        rows = []
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = parse_sample(json.loads(line))
            if split is not None and row.split != split:
                continue
            rows.append(row)
        if limit is not None:
            rows = rows[:limit]
        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self.data_root / sample.image_path
        image = Image.open(image_path).convert("RGB")
        orig_h, orig_w = (int(v) for v in sample.image_size)
        bucket = None
        pad_right = 0
        pad_bottom = 0
        norm_w = orig_w
        norm_h = orig_h
        if self.use_bucket:
            bucket = choose_bucket(orig_h, orig_w, self.bucket_sizes)
            image, pad_right, pad_bottom = pad_right_bottom(image, bucket)
            norm_w = bucket
            norm_h = bucket

        user_text = build_user_prompt(sample)
        target_text = build_target_text(sample)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": target_text},
        ]
        processor_kwargs = {
            "images_kwargs": {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
        }
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs=processor_kwargs,
        )

        input_ids = encoded["input_ids"][0]
        labels = self._build_labels(input_ids)
        boxes = []
        for bbox in sample.bboxes:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            x1 = min(max(x1, 0.0), float(orig_w))
            y1 = min(max(y1, 0.0), float(orig_h))
            x2 = min(max(x2, 0.0), float(orig_w))
            y2 = min(max(y2, 0.0), float(orig_h))
            boxes.append(
                [
                    x1 / norm_w,
                    y1 / norm_h,
                    x2 / norm_w,
                    y2 / norm_h,
                ]
            )

        num_instances = len(boxes)
        boxes_tensor = torch.zeros(num_instances, 4, dtype=torch.float32)
        if num_instances:
            boxes_tensor.copy_(torch.tensor(boxes, dtype=torch.float32))

        grid = encoded["image_grid_thw"][0]
        grid_h, grid_w = int(grid[1]), int(grid[2])
        resize_h, resize_w = grid_h * 16, grid_w * 16
        stride_h, stride_w = resize_h // 8, resize_w // 8
        mask_grids: list[torch.Tensor] = []
        for mask_rel in sample.mask_paths[:num_instances]:
            with Image.open(self.data_root / mask_rel) as mask_handle:
                mask = mask_handle.convert("L")
            if self.use_bucket and bucket is not None:
                mask = pad_mask_canvas(mask, bucket)
            mask_grids.append(_mask_grid_tensor(mask, stride_h, stride_w))
        masks_tensor = (
            torch.stack(mask_grids, dim=0)
            if mask_grids
            else torch.zeros(0, stride_h, stride_w, dtype=torch.float32)
        )

        item = {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "kind": sample.kind.value,
            "expression": sample.expression,
            "label": sample.label,
            "image_path": sample.image_path,
            "mask_paths": sample.mask_paths,
            "orig_size": [orig_h, orig_w],
            "bucket": bucket,
            "pad_right": pad_right,
            "pad_bottom": pad_bottom,
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": encoded["attention_mask"],
            "mm_token_type_ids": encoded["mm_token_type_ids"],
            "labels": labels.unsqueeze(0),
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
            "image": normalize_decoder_image(image, resize_h, resize_w),
            "image_grid": [grid_h, grid_w],
            "boxes": boxes_tensor,
            "masks": masks_tensor,
            "num_instances": num_instances,
            "has_seg": num_instances > 0,
        }
        return item

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        starts = (input_ids == im_start_id).nonzero().flatten()
        if starts.numel() < 2:
            raise ValueError(f"sample does not contain an assistant turn: {starts}")
        assistant_role = int(starts[-1])
        content_start = assistant_role + 3
        labels[:content_start] = -100
        if self.ignore_mask_token_in_ce:
            labels[labels == self.mask_token_id] = -100
        return labels
