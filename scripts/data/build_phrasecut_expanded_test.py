#!/usr/bin/env python3
"""Build a clean expanded PhraseCut held-out test manifest.

All images used by the training manifest are excluded. Legacy test images are
also excluded for benchmark independence. One uniformly sampled phrase per image
is retained; no difficulty filtering is performed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_phrasecut_jsonl import _process_image, _quantiles, _valid_task_structure


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _image_id(record: dict[str, Any]) -> int:
    return int(Path(record["image_path"]).stem)


def _union_area_ratio(record: dict[str, Any]) -> float:
    height, width = (int(v) for v in record["image_size"])
    union = np.zeros((height, width), dtype=np.uint8)
    for path in record.get("mask_paths") or []:
        with Image.open(path) as im:
            mask = np.asarray(im.convert("L"), dtype=np.uint8) > 127
        union |= mask.astype(np.uint8)
    return float(union.sum()) / float(max(width * height, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--base-test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--max-phrase-chars", type=int, default=200)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    args = parser.parse_args()

    root = Path(args.root)
    train_manifest = Path(args.train_manifest)
    base_manifest = Path(args.base_test_manifest)
    output = Path(args.output)
    stats_path = Path(args.stats)
    mask_dir = Path(args.mask_dir) / "train"

    image_info_path = root / "image_data_split.json"
    refer_path = root / "refer_train.json"
    images_dir = root / "images"
    for path in (image_info_path, refer_path, images_dir, train_manifest, base_manifest):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    train_records = _load_jsonl(train_manifest)
    base_records = _load_jsonl(base_manifest)
    train_image_ids = {_image_id(r) for r in train_records}
    base_image_ids = {_image_id(r) for r in base_records}
    base_train_overlap = train_image_ids & base_image_ids

    with image_info_path.open(encoding="utf-8") as f:
        image_info = json.load(f)
    with refer_path.open(encoding="utf-8") as f:
        tasks = json.load(f)

    info_by_id = {
        int(img["image_id"]): img
        for img in image_info
        if str(img.get("split", "")) == "train"
    }
    by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    structural_reject = collections.Counter()
    for task in tasks:
        try:
            image_id = int(task["image_id"])
        except (KeyError, TypeError, ValueError):
            structural_reject["invalid_image_id"] += 1
            continue
        if image_id not in info_by_id:
            structural_reject["not_in_train_split"] += 1
            continue
        if not _valid_task_structure(task, args.max_instances, args.max_phrase_chars):
            structural_reject["invalid_task_structure"] += 1
            continue
        by_image[image_id].append(task)

    excluded_ids = train_image_ids | base_image_ids
    rng = random.Random(args.seed)
    candidate_ids = [
        image_id
        for image_id, local_tasks in by_image.items()
        if local_tasks and image_id not in excluded_ids and (images_dir / f"{image_id}.jpg").is_file()
    ]
    rng.shuffle(candidate_ids)

    payloads: list[dict[str, Any]] = []
    for image_id in candidate_ids:
        local_tasks = list(by_image[image_id])
        rng.shuffle(local_tasks)
        payloads.append(
            {
                "image_id": image_id,
                "image_path": str(images_dir / f"{image_id}.jpg"),
                "tasks": local_tasks,
                "mask_dir": str(mask_dir),
                "min_area_ratio": args.min_area_ratio,
                "max_area_ratio": args.max_area_ratio,
                "split": "train",
            }
        )

    records: list[dict[str, Any]] = []
    rejects = collections.Counter()
    attempted = 0
    max_attempts = min(len(payloads), max(args.num_samples * 3, args.num_samples + 1000))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {}
        payload_iter = iter(payloads[:max_attempts])
        while len(records) < args.num_samples and (pending or True):
            while len(pending) < max(1, args.workers * 2):
                try:
                    payload = next(payload_iter)
                except StopIteration:
                    break
                future = executor.submit(_process_image, payload)
                pending[future] = payload["image_id"]
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                attempted += 1
                result = future.result()
                rejects.update(result.get("reject") or {})
                if result.get("record") is not None and len(records) < args.num_samples:
                    records.append(result["record"])
            if len(records) >= args.num_samples:
                for future in pending:
                    future.cancel()
                break

    if len(records) != args.num_samples:
        raise SystemExit(f"generated {len(records)} records, expected {args.num_samples}")
    record_image_ids = {_image_id(r) for r in records}
    if len({r["sample_id"] for r in records}) != len(records):
        raise SystemExit("duplicate sample_id in generated test manifest")
    if len(record_image_ids) != len(records):
        raise SystemExit("duplicate image in generated test manifest")
    if record_image_ids & train_image_ids:
        raise SystemExit("generated test manifest overlaps training images")
    if record_image_ids & base_image_ids:
        raise SystemExit("generated test manifest overlaps legacy test images")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            clean = {k: v for k, v in record.items() if not k.startswith("_audit")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    tmp.replace(output)

    audits: list[dict[str, Any]] = []
    for record in records:
        audit = dict(record["_audit"])
        audit["union_area_ratio"] = _union_area_ratio(record)
        audits.append(audit)
    phrase_types = collections.Counter(str(a["phrase_type"]) for a in audits)
    instance_counts = collections.Counter(int(a["instances"]) for a in audits)
    relation_counts = collections.Counter("relation" if a["relations"] else "no_relation" for a in audits)
    attr_counts = collections.Counter("attribute" if a["attributes"] else "no_attribute" for a in audits)
    stats = {
        "source": "VGPhraseCut",
        "split": "train",
        "root": str(root),
        "output": str(output),
        "train_manifest": str(train_manifest),
        "base_test_manifest": str(base_manifest),
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "output_samples": len(records),
        "unique_images": len(record_image_ids),
        "train_images_excluded": len(train_image_ids),
        "base_test_images_excluded": len(base_image_ids),
        "base_test_overlap_train_images": len(base_train_overlap),
        "base_test_clean_images": len(base_image_ids - train_image_ids),
        "eligible_images_after_exclusion": len(payloads),
        "attempted_images": attempted,
        "structural_reject": dict(structural_reject),
        "record_reject": dict(rejects),
        "final_overlap_train_images": len(record_image_ids & train_image_ids),
        "final_overlap_base_test_images": len(record_image_ids & base_image_ids),
        "phrase_type": dict(phrase_types),
        "instance_count": dict(sorted(instance_counts.items())),
        "relation_presence": dict(relation_counts),
        "attribute_presence": dict(attr_counts),
        "phrase_chars": _quantiles([float(a["phrase_chars"]) for a in audits]),
        "union_area_ratio": _quantiles([float(a["union_area_ratio"]) for a in audits]),
        "selection_rule": "new clean test set; one uniformly sampled eligible phrase per image; excludes all train and legacy test images; no difficulty filtering",
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_tmp = Path(str(stats_path) + ".tmp")
    stats_tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_tmp.replace(stats_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
