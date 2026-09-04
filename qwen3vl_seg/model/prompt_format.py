"""Build Qwen3-VL-Seg prompts and target JSON records."""

from __future__ import annotations

import json
from typing import Any

MASK_START = "<mask_start>"
MASK_TOKEN = "<mask_token>"
MASK_END = "<mask_end>"
SEG_SPECIAL_TOKENS = (MASK_START, MASK_TOKEN, MASK_END)

CATEGORY_TEMPLATE = (
    'Locate and segment every instance that belongs to the following categories '
    '"{category}", report bbox coordinates and masks in JSON format.'
)
DESCRIPTION_TEMPLATE = (
    'Locate and segment the object that matches the description '
    '"{description}", report bbox coordinates and masks in JSON format.'
)


def ensure_seg_special_tokens(tokenizer: Any) -> list[int]:
    """Idempotently add the mask placeholder tokens to a tokenizer."""
    missing = [tok for tok in SEG_SPECIAL_TOKENS if tokenizer.convert_tokens_to_ids(tok) is None]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return [tokenizer.convert_tokens_to_ids(tok) for tok in SEG_SPECIAL_TOKENS]


def _scale_box(bbox: list[float], width: float, height: float) -> list[int]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 <= x1:
        x2 = min(x1 + 1.0, width)
    if y2 <= y1:
        y2 = min(y1 + 1.0, height)
    return [
        round(x1 * 1000.0 / width),
        round(y1 * 1000.0 / height),
        round(x2 * 1000.0 / width),
        round(y2 * 1000.0 / height),
    ]


def build_user_prompt(sample: Any) -> str:
    kind = getattr(sample, "kind", None)
    kind_name = kind if isinstance(kind, str) else getattr(kind, "value", kind)
    if kind_name in ("category_single", "category_multi"):
        category = sample.label or sample.expression
        return CATEGORY_TEMPLATE.format(category=category)
    return DESCRIPTION_TEMPLATE.format(description=sample.expression)


def build_target_records(sample: Any) -> list[dict[str, Any]]:
    """Return records using the paper's 0..1000 bbox convention."""
    height, width = (int(v) for v in sample.image_size)
    label = sample.label or sample.expression
    records: list[dict[str, Any]] = []
    for bbox in sample.bboxes:
        records.append(
            {
                "bbox_2d": _scale_box(bbox, width, height),
                "label": label,
                "mask": f"{MASK_START}{MASK_TOKEN}{MASK_END}",
            }
        )
    return records


def build_target_text(sample: Any) -> str:
    return json.dumps(build_target_records(sample), ensure_ascii=False)


def mask_placeholder_positions(input_ids: Any, token_ids: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Locate (start, token, end) triplets in one token sequence."""
    start_id, token_id, end_id = token_ids
    starts = (input_ids == start_id).nonzero().flatten().tolist()
    result = []
    for start_pos in starts:
        end = start_pos + 2
        if (
            end < input_ids.shape[0]
            and int(input_ids[start_pos + 1]) == token_id
            and int(input_ids[end]) == end_id
        ):
            result.append((start_pos, start_pos + 1, end))
    return result
