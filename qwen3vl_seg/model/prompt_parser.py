from __future__ import annotations

import json
import re
from typing import Any


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _candidate_blocks(text: str) -> list[str]:
    candidates = [m.group(1) for m in _CODE_FENCE_RE.finditer(text)]
    if not candidates:
        for pattern in (_JSON_ARRAY_RE, _JSON_OBJECT_RE):
            match = pattern.search(text)
            if match:
                candidates.append(match.group(0))
    return candidates


def parse_generated_boxes_and_masks(text: str) -> list[dict[str, Any]]:
    """Parse the model's structured JSON into bbox/mask records.

    Returns [] if parsing fails so callers can mark the sample as empty.
    """
    if not text:
        return []
    for block in _candidate_blocks(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        records = payload if isinstance(payload, list) else [payload]
        normalized: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if "bbox_2d" not in record or "mask" not in record:
                continue
            bbox = record["bbox_2d"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            normalized.append(
                {
                    "bbox_2d": [float(v) for v in bbox],
                    "label": str(record.get("label", "")),
                    "mask": str(record["mask"]),
                }
            )
        if normalized:
            return normalized
    return []

