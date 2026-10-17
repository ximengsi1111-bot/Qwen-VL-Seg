#!/usr/bin/env python3
"""Build a stratified 50k GRPO mixture.

Composition:
  PhraseCut              30,000
  RefCOCO                 4,000
  RefCOCO+                4,000
  RefCOCOg                4,000
  COCO                    4,000
  LVIS                    4,000
  Total                  50,000

The replay records are sampled from the parent rl_hard.jsonl so the requested
counts do not require repeating records that were already dropped from
rl_hard_70k.jsonl. Sampling is base-image aware: a base image contributes one
augmentation before any base contributes a second, and so on.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any

AUG_RE = re.compile(r"_(hflip|color|noise|orig|rot\d*|crop\d*)$")
REPLAY_SOURCES = {
    "refcoco:train": 4_000,
    "refcocoplus:train": 4_000,
    "refcocog:train": 4_000,
    "coco:train": 4_000,
    "lvis:train": 4_000,
}


def base_and_aug(image_path: str) -> tuple[str, str]:
    stem = Path(image_path).stem
    match = AUG_RE.search(stem)
    if match:
        return stem[: match.start()], match.group(1)
    return stem, "none"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_base_balanced(
    records: list[dict[str, Any]],
    target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_base: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        base_id, _ = base_and_aug(str(record.get("image_path", "")))
        by_base[base_id].append(record)

    rng = random.Random(seed)
    bases = list(by_base)
    rng.shuffle(bases)
    for base_id in bases:
        rng.shuffle(by_base[base_id])

    selected: list[dict[str, Any]] = []
    layer = 0
    while len(selected) < target:
        order = list(bases)
        rng.shuffle(order)
        added = 0
        for base_id in order:
            variants = by_base[base_id]
            if layer >= len(variants):
                continue
            selected.append(variants[layer])
            added += 1
            if len(selected) >= target:
                break
        if added == 0:
            break
        layer += 1

    if len(selected) != target:
        raise SystemExit(
            f"requested {target} records, but only {len(selected)} could be selected"
        )

    selected_bases = collections.Counter(
        base_and_aug(str(record.get("image_path", "")))[0]
        for record in selected
    )
    aug_counts = collections.Counter(
        base_and_aug(str(record.get("image_path", "")))[1]
        for record in selected
    )
    audit = {
        "input_rows": len(records),
        "input_bases": len(by_base),
        "selected_rows": len(selected),
        "selected_bases": len(selected_bases),
        "max_variants_per_base": max(selected_bases.values()),
        "aug_selected": dict(sorted(aug_counts.items())),
    }
    return selected, audit


def validate_record(record: dict[str, Any], index: int, check_files: bool) -> None:
    image_path = record.get("image_path")
    mask_paths = record.get("mask_paths")
    if not isinstance(image_path, str) or not image_path:
        raise SystemExit(f"record {index}: missing image_path")
    if not isinstance(mask_paths, list) or not mask_paths:
        raise SystemExit(f"record {index}: missing mask_paths")

    solution_raw = record.get("solution")
    if not isinstance(solution_raw, str):
        raise SystemExit(f"record {index}: solution is not a JSON string")
    solution = json.loads(solution_raw)
    boxes = solution.get("bbox_1000")
    if not isinstance(boxes, list) or len(boxes) != len(mask_paths):
        raise SystemExit(
            f"record {index}: bbox/mask count mismatch "
            f"({len(boxes) if isinstance(boxes, list) else None}/{len(mask_paths)})"
        )

    response_raw = record.get("response")
    if not isinstance(response_raw, str):
        raise SystemExit(f"record {index}: response is not a JSON string")
    response = json.loads(response_raw)
    if not isinstance(response, list) or len(response) != len(mask_paths):
        raise SystemExit(f"record {index}: response/mask count mismatch")

    if check_files:
        if not Path(image_path).is_file():
            raise SystemExit(f"record {index}: image does not exist: {image_path}")
        missing = [path for path in mask_paths if not Path(path).is_file()]
        if missing:
            raise SystemExit(f"record {index}: mask does not exist: {missing[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phrasecut-input",
        default="/mingli01/data/xyk/grpo/newdata/phrasecut/processed/phrasecut_50k_formal.jsonl",
    )
    parser.add_argument(
        "--replay-input",
        default="/mingli01/data/xyk/grpo/rl_hard.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/mingli01/data/xyk/grpo/mixed_50k_phrasecut_refcoco_coco_lvis.jsonl",
    )
    parser.add_argument(
        "--stats",
        default="/mingli01/data/xyk/grpo/mixed_50k_phrasecut_refcoco_coco_lvis_stats.json",
    )
    parser.add_argument("--phrasecut-count", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()

    phrasecut_path = Path(args.phrasecut_input)
    replay_path = Path(args.replay_input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    phrasecut_records = load_jsonl(phrasecut_path)
    replay_records = load_jsonl(replay_path)

    phrasecut_selected, phrasecut_audit = select_base_balanced(
        phrasecut_records,
        args.phrasecut_count,
        args.seed + 1,
    )

    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in replay_records:
        by_source[str(record.get("source", ""))].append(record)

    selected: list[dict[str, Any]] = list(phrasecut_selected)
    replay_audits: dict[str, dict[str, Any]] = {}
    for source_index, (source, target) in enumerate(REPLAY_SOURCES.items()):
        source_records = by_source.get(source, [])
        chosen, audit = select_base_balanced(
            source_records,
            target,
            args.seed + 10 + source_index,
        )
        selected.extend(chosen)
        replay_audits[source] = audit

    rng = random.Random(args.seed)
    rng.shuffle(selected)

    if len(selected) != 50_000:
        raise SystemExit(f"expected 50000 final rows, got {len(selected)}")
    if len({str(record.get("image_path", "")) for record in selected}) != len(selected):
        raise SystemExit("duplicate image_path in final mixture")

    for index, record in enumerate(selected):
        validate_record(record, index, check_files=not args.skip_file_check)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = Path(str(output_path) + ".tmp")
    with output_tmp.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_tmp.replace(output_path)

    source_counts = collections.Counter(str(record.get("source", "")) for record in selected)
    kind_counts = collections.Counter(str(record.get("kind", "")) for record in selected)
    aug_counts = collections.Counter(
        base_and_aug(str(record.get("image_path", "")))[1]
        for record in selected
    )
    stats = {
        "output": str(output_path),
        "phrasecut_input": str(phrasecut_path),
        "replay_input": str(replay_path),
        "seed": args.seed,
        "requested_total": 50_000,
        "output_rows": len(selected),
        "unique_images": len({str(record["image_path"]) for record in selected}),
        "source_output": dict(sorted(source_counts.items())),
        "kind_output": dict(sorted(kind_counts.items())),
        "aug_output": dict(sorted(aug_counts.items())),
        "phrasecut": phrasecut_audit,
        "replay": replay_audits,
        "file_check": not args.skip_file_check,
        "selection_rule": (
            "30k random PhraseCut + balanced 4k each RefCOCO/RefCOCO+/RefCOCOg/"
            "COCO/LVIS; base-aware replay sampling from rl_hard.jsonl"
        ),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_tmp = Path(str(stats_path) + ".tmp")
    stats_tmp.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_tmp.replace(stats_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
