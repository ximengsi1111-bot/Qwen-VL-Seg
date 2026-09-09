"""Merge per-rank partial eval metrics into a single metrics.json."""
from __future__ import annotations
import json, sys
from pathlib import Path

def main() -> int:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    parts = sorted(output_dir.glob("metrics.partial.*.json"))
    if not parts:
        print("no partial metrics found", file=sys.stderr)
        return 1
    tot = {
        "n_samples": 0,
        "sum_strict": 0.0,
        "sum_inter": 0.0,
        "sum_union": 0.0,
        "matched_correct": 0,
        "matched_total": 0,
        "strict_correct": 0,
        "strict_total": 0,
    }
    for p in parts:
        d = json.loads(p.read_text(encoding="utf-8"))
        for k in tot:
            tot[k] += d.get(k, 0)
    n = tot["n_samples"]
    metrics = {
        "n_samples": n,
        "miou": float(tot["sum_strict"] / n) if n else 0.0,
        "ciou": float(tot["sum_inter"] / tot["sum_union"]) if tot["sum_union"] > 0 else 0.0,
        "matched_p@0.5": float(tot["matched_correct"] / tot["matched_total"]) if tot["matched_total"] > 0 else 0.0,
        "strict_p@0.5": float(tot["strict_correct"] / tot["strict_total"]) if tot["strict_total"] > 0 else 0.0,
        "num_parts": len(parts),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())