#!/usr/bin/env python3
"""Build a ~70k GRPO subset keeping 2/3 of each augmentation type."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

AUG_RE = re.compile(r"_(hflip|color|noise|orig|rot\d*|crop\d*)$")


def base_and_aug(image_path: str) -> tuple[str, str]:
    stem = Path(image_path).stem
    m = AUG_RE.search(stem)
    if m:
        return stem[:m.start()], m.group(1)
    return stem, "none"


def load_scores(path: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    p = Path(path)
    if not p.is_file():
        return scores
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sid = d.get("sample_id") or d.get("id")
            if sid is None:
                continue
            m = d.get("miou", d.get("mask_iou"))
            if isinstance(m, (int, float)):
                scores[str(sid)] = float(m)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/mingli01/data/xyk/grpo/rl_hard.jsonl")
    parser.add_argument("--score-index", default="/mingli01/data/xyk/runs/eval/train248k-score-index.jsonl")
    parser.add_argument("--output", default="/mingli01/data/xyk/grpo/rl_hard_70k.jsonl")
    parser.add_argument("--stats", default="/mingli01/data/xyk/grpo/rl_hard_70k_stats.json")
    parser.add_argument("--keep-ratio", type=float, default=2.0 / 3.0)
    args = parser.parse_args()

    scores = load_scores(args.score_index)
    records: list[dict] = []
    total = 0
    bases: set[str] = set()
    aug_counts: collections.Counter = collections.Counter()
    missing_score = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base_id, aug = base_and_aug(d.get("image_path", ""))
            source = str(d.get("source", ""))
            kind = str(d.get("kind", ""))
            miou = scores.get(base_id)
            if miou is None:
                missing_score += 1
                miou = 0.5
            d["_base_id"] = base_id
            d["_aug"] = aug
            d["_miou"] = float(miou)
            records.append(d)
            total += 1
            bases.add(base_id)
            aug_counts[aug] += 1

    groups: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    for d in records:
        groups[(d["source"], d["kind"], d["_aug"])].append(d)

    selected: list[dict] = []
    for items in groups.values():
        k = int(round(len(items) * args.keep_ratio))
        items.sort(key=lambda d: (d["_miou"], d["_base_id"]))
        selected.extend(items[:k])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in selected:
            out = {k: v for k, v in d.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    sel_aug = collections.Counter(d["_aug"] for d in selected)
    sel_source = collections.Counter(d["source"] for d in selected)
    sel_kind = collections.Counter(d["kind"] for d in selected)
    sel_bases = {d["_base_id"] for d in selected}
    sel_mious = [d["_miou"] for d in selected]
    stats = {
        "input": args.input,
        "score_index": args.score_index,
        "output": str(out_path),
        "keep_ratio": args.keep_ratio,
        "input_rows": total,
        "input_bases": len(bases),
        "output_rows": len(selected),
        "output_bases": len(sel_bases),
        "missing_score_rows": missing_score,
        "aug_input": dict(aug_counts),
        "aug_output": dict(sel_aug),
        "source_output": dict(sel_source),
        "kind_output": dict(sel_kind),
        "miou_output": {
            "count": len(sel_mious),
            "mean": sum(sel_mious) / len(sel_mious) if sel_mious else 0.0,
            "min": min(sel_mious) if sel_mious else None,
            "max": max(sel_mious) if sel_mious else None,
        },
    }
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
