#!/usr/bin/env python3
"""Build GRPO JSONL records from the official VGPhraseCut dataset.

One output record = one image + one referring phrase + all referred instances.
The default pilot keeps at most one phrase per image, so image diversity is not
inflated by augmentations. Samples are selected uniformly from eligible official
train images; no difficulty-based selection is performed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

MASK_TEXT = "<mask_start><mask_token><mask_end>"
SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    xs = sorted(values)

    def q(p: float) -> float:
        pos = (len(xs) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {
        "count": len(xs),
        "min": xs[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": xs[-1],
        "mean": statistics.fmean(xs),
    }


def _polygon_mask(instance_polygons: Any, width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    drew = False
    for polygon in instance_polygons or []:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            continue
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if len(points) < 3:
            continue
        draw.polygon(points, fill=1, outline=1)
        drew = True
    if not drew:
        return np.zeros((height, width), dtype=np.uint8)
    return (np.asarray(canvas, dtype=np.uint8) > 0).astype(np.uint8)


def _xywh_to_xyxy(box: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x, y, w, h = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x1 = min(max(x, 0.0), max(width - 1.0, 0.0))
    y1 = min(max(y, 0.0), max(height - 1.0, 0.0))
    x2 = min(max(x + w - 1.0, 0.0), max(width - 1.0, 0.0))
    y2 = min(max(y + h - 1.0, 0.0), max(height - 1.0, 0.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _norm_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    vals = [
        x1 / max(width - 1.0, 1.0) * 1000.0,
        y1 / max(height - 1.0, 1.0) * 1000.0,
        x2 / max(width - 1.0, 1.0) * 1000.0,
        y2 / max(height - 1.0, 1.0) * 1000.0,
    ]
    return [min(1000, max(0, int(round(v)))) for v in vals]


def _valid_task_structure(task: dict[str, Any], max_instances: int, max_phrase_chars: int) -> bool:
    phrase = str(task.get("phrase", "")).strip()
    boxes = task.get("instance_boxes")
    polygons = task.get("Polygons")
    if not phrase or len(phrase) > max_phrase_chars:
        return False
    if not isinstance(boxes, list) or not isinstance(polygons, list):
        return False
    if not boxes or len(boxes) != len(polygons) or len(boxes) > max_instances:
        return False
    return True


def _process_image(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(payload["image_path"])
    tasks = payload["tasks"]
    mask_dir = Path(payload["mask_dir"])
    min_area_ratio = float(payload["min_area_ratio"])
    max_area_ratio = float(payload["max_area_ratio"])
    reject = collections.Counter()

    try:
        with Image.open(image_path) as im:
            width, height = im.size
            im.verify()
    except Exception as exc:
        return {"record": None, "reject": {"image_open_failed": 1}, "error": str(exc), "image_id": payload["image_id"]}

    for task in tasks:
        phrase = str(task.get("phrase", "")).strip()
        boxes = task.get("instance_boxes") or []
        polygons = task.get("Polygons") or []

        masks: list[np.ndarray] = []
        boxes_px: list[list[float]] = []
        all_valid = True
        for polygons_i, box_i in zip(polygons, boxes):
            mask = _polygon_mask(polygons_i, width, height)
            if int(mask.sum()) == 0:
                reject["empty_mask"] += 1
                all_valid = False
                break
            area_ratio = float(mask.sum()) / float(max(width * height, 1))
            if area_ratio < min_area_ratio:
                reject["tiny_mask"] += 1
                all_valid = False
                break
            box_px = _xywh_to_xyxy(box_i, width, height)
            if box_px is None:
                reject["invalid_box"] += 1
                all_valid = False
                break
            masks.append(mask)
            boxes_px.append(box_px)

        if not all_valid or not masks:
            continue

        union = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            union |= mask
        union_ratio = float(union.sum()) / float(max(width * height, 1))
        if union_ratio > max_area_ratio:
            reject["huge_union_mask"] += 1
            continue

        image_id = payload["image_id"]
        task_id = str(task.get("task_id", "task"))
        token = hashlib.sha1(f"{image_id}:{task_id}".encode("utf-8")).hexdigest()[:12]
        sample_id = f"phrasecut_{image_id}_{token}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_paths: list[str] = []
        for instance_i, mask in enumerate(masks):
            mask_path = mask_dir / f"phrasecut_{image_id}_{token}_{instance_i}.png"
            Image.fromarray(mask * 255).save(mask_path, format="PNG", optimize=False)
            mask_paths.append(str(mask_path.resolve()))

        bbox_1000 = [_norm_box(box, width, height) for box in boxes_px]
        label = str((task.get("phrase_structure") or {}).get("name") or phrase)
        response = [
            {"bbox_2d": bbox, "label": label, "mask": MASK_TEXT}
            for bbox in bbox_1000
        ]
        if len(mask_paths) == 1:
            instruction = (
                f'Locate and segment the object that matches the description "{phrase}", '
                "report bbox coordinates and masks in JSON format."
            )
        else:
            instruction = (
                f'Locate and segment every object that matches the description "{phrase}", '
                "report bbox coordinates and masks in JSON format."
            )

        absolute_image = str(image_path.resolve())
        solution = {
            "bbox_1000": bbox_1000,
            "image_path": absolute_image,
            "mask_paths": mask_paths,
            "image_size": [height, width],
        }
        record = {
            "sample_id": sample_id,
            "split": f"phrasecut_{payload['split']}",
            "expression": phrase,
            "label": label,
            "created_at": "",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": absolute_image},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            "response": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            "solution": json.dumps(solution, ensure_ascii=False, separators=(",", ":")),
            "image_path": absolute_image,
            "mask_paths": mask_paths,
            "bboxes": boxes_px,
            "image_size": [height, width],
            "source": f"phrasecut:{payload['split']}",
            "kind": "phrasal",
        }
        structure = task.get("phrase_structure") or {}
        record["_audit"] = {
            "phrase": phrase,
            "phrase_type": str(structure.get("type", "")),
            "attributes": list(structure.get("attributes") or []),
            "relations": list(structure.get("relation_descriptions") or []),
            "instances": len(masks),
            "union_area_ratio": union_ratio,
            "phrase_chars": len(phrase),
        }
        return {"record": record, "reject": dict(reject), "image_id": image_id}

    return {"record": None, "reject": dict(reject), "image_id": payload["image_id"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extracted VGPhraseCut_v0 directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "miniv"])
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--max-phrase-chars", type=int, default=200)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    args = parser.parse_args()

    root = Path(args.root)
    image_info_path = root / "image_data_split.json"
    refer_path = root / f"refer_{args.split}.json"
    images_dir = root / "images"
    for p in (image_info_path, refer_path, images_dir):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    with image_info_path.open(encoding="utf-8") as f:
        image_info = json.load(f)
    with refer_path.open(encoding="utf-8") as f:
        tasks = json.load(f)

    info_by_id = {
        int(img["image_id"]): img
        for img in image_info
        if str(img.get("split", "")) == args.split
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
            structural_reject["not_in_split"] += 1
            continue
        if not _valid_task_structure(task, args.max_instances, args.max_phrase_chars):
            structural_reject["invalid_task_structure"] += 1
            continue
        by_image[image_id].append(task)

    rng = random.Random(args.seed)
    image_ids = [i for i, ts in by_image.items() if ts]
    rng.shuffle(image_ids)
    payloads: list[dict[str, Any]] = []
    for image_id in image_ids:
        img = info_by_id[image_id]
        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            structural_reject["missing_image_file"] += 1
            continue
        local_tasks = list(by_image[image_id])
        rng.shuffle(local_tasks)
        payloads.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "tasks": local_tasks,
                "mask_dir": str(Path(args.mask_dir) / args.split),
                "min_area_ratio": args.min_area_ratio,
                "max_area_ratio": args.max_area_ratio,
                "split": args.split,
            }
        )

    records: list[dict[str, Any]] = []
    rejected = collections.Counter()
    attempted = 0
    max_attempts = min(len(payloads), max(args.num_samples * 3, args.num_samples + 1000))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {}
        payload_iter = iter(payloads[:max_attempts])
        while (len(records) < args.num_samples) and (pending or True):
            while len(pending) < max(1, args.workers * 2):
                try:
                    payload = next(payload_iter)
                except StopIteration:
                    break
                fut = executor.submit(_process_image, payload)
                pending[fut] = payload["image_id"]
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut, None)
                attempted += 1
                result = fut.result()
                rejected.update(result.get("reject") or {})
                if result.get("record") is not None and len(records) < args.num_samples:
                    records.append(result["record"])
            if len(records) >= args.num_samples:
                for fut in pending:
                    fut.cancel()
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = Path(str(out_path) + ".tmp")
    with out_tmp.open("w", encoding="utf-8") as f:
        for record in records:
            clean = {k: v for k, v in record.items() if not k.startswith("_audit")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    out_tmp.replace(out_path)

    audit = [r["_audit"] for r in records]
    phrase_types = collections.Counter(str(a["phrase_type"]) for a in audit)
    instance_counts = collections.Counter(int(a["instances"]) for a in audit)
    relation_counts = collections.Counter("relation" if a["relations"] else "no_relation" for a in audit)
    attr_counts = collections.Counter("attribute" if a["attributes"] else "no_attribute" for a in audit)
    stats = {
        "source": "VGPhraseCut",
        "split": args.split,
        "root": str(root),
        "output": str(out_path),
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "output_samples": len(records),
        "unique_images": len({r["image_path"] for r in records}),
        "eligible_images": len(payloads),
        "eligible_tasks": sum(len(v) for v in by_image.values()),
        "attempted_images": attempted,
        "structural_reject": dict(structural_reject),
        "record_reject": dict(rejected),
        "phrase_type": dict(phrase_types),
        "instance_count": dict(sorted(instance_counts.items())),
        "relation_presence": dict(relation_counts),
        "attribute_presence": dict(attr_counts),
        "phrase_chars": _quantiles([float(a["phrase_chars"]) for a in audit]),
        "union_area_ratio": _quantiles([float(a["union_area_ratio"]) for a in audit]),
        "selection_rule": "one uniformly sampled eligible phrase per image; no difficulty filtering",
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_tmp = Path(str(stats_path) + ".tmp")
    stats_tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_tmp.replace(stats_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
