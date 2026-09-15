#!/usr/bin/env python3
"""Validate GRPO JSONL records generated from PhraseCut."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from PIL import Image

TOP_KEYS = {
    "messages", "response", "solution", "image_path", "mask_paths",
    "bboxes", "image_size", "source", "kind",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Validate first N rows; 0 means all")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--check-masks", action="store_true")
    args = parser.parse_args()

    rows = 0
    errors = collections.Counter()
    sources = collections.Counter()
    kinds = collections.Counter()
    instances = collections.Counter()
    seen_task_paths = set()
    with open(args.jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            if args.limit and rows >= args.limit:
                break
            rows += 1
            try:
                d = json.loads(line)
                solution = json.loads(d["solution"])
                response = json.loads(d["response"])
            except Exception as exc:
                errors[f"json_{type(exc).__name__}"] += 1
                continue

            missing = TOP_KEYS - set(d)
            if missing:
                errors["missing_top_keys"] += 1
            sources[str(d.get("source", ""))] += 1
            kinds[str(d.get("kind", ""))] += 1

            mask_paths = list(solution.get("mask_paths") or [])
            boxes = list(solution.get("bbox_1000") or [])
            image_size = solution.get("image_size") or d.get("image_size")
            instances[len(mask_paths)] += 1
            if len(mask_paths) != len(boxes) or len(mask_paths) != len(response):
                errors["mask_box_response_count_mismatch"] += 1
            if not mask_paths:
                errors["empty_masks"] += 1
            if not isinstance(image_size, list) or len(image_size) != 2:
                errors["bad_image_size"] += 1
                continue

            h, w = int(image_size[0]), int(image_size[1])
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    errors["bad_bbox"] += 1
                    continue
                if not all(0 <= int(v) <= 1000 for v in box):
                    errors["bbox_out_of_range"] += 1

            key = tuple(mask_paths)
            if key in seen_task_paths:
                errors["duplicate_sample"] += 1
            seen_task_paths.add(key)

            if args.check_images:
                try:
                    with Image.open(d["image_path"]) as im:
                        if list(im.size) != [w, h]:
                            errors["image_size_mismatch"] += 1
                except Exception:
                    errors["image_open_failed"] += 1

            if args.check_masks:
                for mask_path in mask_paths:
                    try:
                        with Image.open(mask_path) as im:
                            if im.mode != "L" or list(im.size) != [w, h]:
                                errors["mask_format_or_size_mismatch"] += 1
                    except Exception:
                        errors["mask_open_failed"] += 1

    print(json.dumps({
        "rows": rows,
        "sources": dict(sources),
        "kinds": dict(kinds),
        "instances_per_sample": dict(sorted(instances.items())),
        "errors": dict(errors),
    }, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
