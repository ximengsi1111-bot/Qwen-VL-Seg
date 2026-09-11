#!/usr/bin/env python3
"""Build a ~70k GRPO subset by randomly dropping one augmentation per base.

Input construction:
  35,045 base samples x {hflip, color, noise} = 105,135 records.

Output construction:
  Keep exactly 2 augmentations per base => 70,090 records.
  The dropped augmentation type is randomly assigned and globally balanced,
  so each augmentation type is retained for about 2/3 of the bases.
  No difficulty-based prioritization is used. The base-level difficulty
  distribution is therefore preserved; miou is only reported for QA.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
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


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def miou_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/mingli01/data/xyk/grpo/rl_hard.jsonl")
    parser.add_argument("--score-index", default="/mingli01/data/xyk/runs/eval/train248k-score-index.jsonl")
    parser.add_argument("--output", default="/mingli01/data/xyk/grpo/rl_hard_70k.jsonl")
    parser.add_argument("--stats", default="/mingli01/data/xyk/grpo/rl_hard_70k_stats.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scores = load_scores(args.score_index)
    by_base: dict[str, dict[str, dict]] = {}
    input_rows = 0
    missing_score = 0
    aug_input: collections.Counter = collections.Counter()
    source_input: collections.Counter = collections.Counter()
    kind_input: collections.Counter = collections.Counter()

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base_id, aug = base_and_aug(d.get("image_path", ""))
            d["_base_id"] = base_id
            d["_aug"] = aug
            d["_miou"] = scores.get(base_id)
            by_base.setdefault(base_id, {})[aug] = d
            input_rows += 1
            aug_input[aug] += 1
            source_input[str(d.get("source", ""))] += 1
            kind_input[str(d.get("kind", ""))] += 1
            if d["_miou"] is None:
                missing_score += 1

    bases = list(by_base)
    for base_id, variants in by_base.items():
        if set(variants) != {"hflip", "color", "noise"}:
            raise SystemExit(f"unexpected augmentation set for {base_id}: {sorted(variants)}")

    rng = random.Random(args.seed)
    rng.shuffle(bases)

    n = len(bases)
    n_hflip = n // 3 + (1 if n % 3 >= 1 else 0)
    n_color = n // 3 + (1 if n % 3 >= 2 else 0)
    n_noise = n // 3
    drop_types = ["hflip"] * n_hflip + ["color"] * n_color + ["noise"] * n_noise
    rng.shuffle(drop_types)

    selected: list[dict] = []
    drop_counts: collections.Counter = collections.Counter()
    for base_id, drop_aug in zip(bases, drop_types):
        drop_counts[drop_aug] += 1
        for aug, rec in by_base[base_id].items():
            if aug != drop_aug:
                selected.append(rec)

    selected.sort(key=lambda d: (d["_base_id"], d["_aug"]))
    out_path = Path(args.output)
    out_tmp = Path(str(out_path) + ".tmp")
    out_tmp.parent.mkdir(parents=True, exist_ok=True)
    with out_tmp.open("w", encoding="utf-8") as f:
        for d in selected:
            out = {k: v for k, v in d.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    out_tmp.replace(out_path)

    sel_aug = collections.Counter(d["_aug"] for d in selected)
    sel_source = collections.Counter(str(d.get("source", "")) for d in selected)
    sel_kind = collections.Counter(str(d.get("kind", "")) for d in selected)
    base_mious = [float(d["_miou"]) for d in by_base.values() for d in d.values() if d["_miou"] is not None]
    unique_base_mious = [float(next(iter(v.values()))["_miou"]) for v in by_base.values() if next(iter(v.values()))["_miou"] is not None]

    stats = {
        "input": args.input,
        "score_index": args.score_index,
        "output": str(out_path),
        "seed": args.seed,
        "input_rows": input_rows,
        "input_bases": len(by_base),
        "output_rows": len(selected),
        "output_bases": len({d["_base_id"] for d in selected}),
        "missing_score_rows": missing_score,
        "aug_input": dict(aug_input),
        "aug_output": dict(sel_aug),
        "drop_counts": dict(drop_counts),
        "source_input": dict(source_input),
        "source_output": dict(sel_source),
        "kind_input": dict(kind_input),
        "kind_output": dict(sel_kind),
        "miou_input_record_level": miou_stats(base_mious),
        "miou_unique_base_level": miou_stats(unique_base_mious),
        "selection_rule": "one randomly dropped augmentation per base; drop type globally balanced; no miou prioritization",
    }

    stats_path = Path(args.stats)
    stats_tmp = Path(str(stats_path) + ".tmp")
    stats_tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_tmp.replace(stats_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
