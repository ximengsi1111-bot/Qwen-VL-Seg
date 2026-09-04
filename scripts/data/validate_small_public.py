#!/usr/bin/env python3
"""Validate the built small public dataset on the server or a local staging root."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    samples_path = root / "datasets" / "small" / "samples.jsonl"
    rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_counts = collections.Counter(row["split"] for row in rows)
    missing_images: list[str] = []
    missing_masks: list[str] = []
    bad_lengths: list[str] = []
    bad_labels: list[str] = []
    bad_mask_size: list[str] = []
    for row in rows:
        if not (root / row["image_path"]).exists():
            missing_images.append(row["sample_id"])
        if len(row["bboxes"]) != len(row["mask_paths"]):
            bad_lengths.append(row["sample_id"])
        if row["label"].isdigit():
            bad_labels.append(row["sample_id"])
        for mask_rel in row["mask_paths"]:
            mask_path = root / mask_rel
            if not mask_path.exists():
                missing_masks.append(row["sample_id"])
                continue
            image = Image.open(root / row["image_path"])
            mask = Image.open(mask_path)
            if mask.size != image.size:
                bad_mask_size.append(row["sample_id"])
    print(
        {
            "samples": len(rows),
            "splits": dict(split_counts),
            "images_checked": len({row["image_path"] for row in rows}),
            "missing_images": len(missing_images),
            "missing_masks": len(missing_masks),
            "bad_lengths": len(bad_lengths),
            "bad_labels": len(bad_labels),
            "bad_mask_size": len(bad_mask_size),
        }
    )
    if any(
        (
            missing_images,
            missing_masks,
            bad_lengths,
            bad_labels,
            bad_mask_size,
        )
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
