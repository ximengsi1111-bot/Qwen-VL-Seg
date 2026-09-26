"""Merge per-rank partial eval metrics into a single metrics.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main() -> int:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    parts = sorted(output_dir.glob("metrics.partial.*.json"))
    if not parts:
        print("no partial metrics found", file=sys.stderr)
        return 1

    totals = {
        "n_samples": 0,
        "sum_miou": 0.0,
        "sum_ciou": 0.0,
        "tp@0.5": 0,
        "fp@0.5": 0,
        "fn@0.5": 0,
        "diagnostic_matched_correct": 0,
        "diagnostic_matched_total": 0,
    }
    for path in parts:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in totals:
            totals[key] += data.get(key, 0)

    n = int(totals["n_samples"])
    tp = int(totals["tp@0.5"])
    fp = int(totals["fp@0.5"])
    fn = int(totals["fn@0.5"])
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    matched_total = int(totals["diagnostic_matched_total"])
    metrics = {
        "n_samples": n,
        "miou": _ratio(float(totals["sum_miou"]), n),
        "ciou": _ratio(float(totals["sum_ciou"]), n),
        "precision@0.5": precision,
        "recall@0.5": recall,
        "f1@0.5": f1,
        "diagnostic_matched_p@0.5": _ratio(
            int(totals["diagnostic_matched_correct"]), matched_total
        ),
        "num_parts": len(parts),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
