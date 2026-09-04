#!/usr/bin/env python3
"""Download COCO2017 images listed in image_manifest.jsonl."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path


def _jpeg_ok(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(2) == b"\xff\xd8"
    except OSError:
        return False


def download_one(row: dict, root: Path, curl: str) -> tuple[str, bool, str]:
    rel = Path(row["image_path"])
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and _jpeg_ok(out):
        return rel.as_posix(), True, "skip"
    for attempt in range(4):
        result = subprocess.run(
            [
                curl,
                "-sS",
                "-L",
                "-f",
                "-k",
                "--connect-timeout",
                "20",
                "--max-time",
                "90",
                "--retry",
                "2",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "-C",
                "-",
                "-o",
                str(out),
                row["url"],
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0 and _jpeg_ok(out):
            return rel.as_posix(), True, f"ok_{attempt}"
    return rel.as_posix(), False, "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    curl = shutil.which("curl") or "curl"
    rows = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_one, row, args.data_root.resolve(), curl)
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    failed = [rel for rel, ok, _ in results if not ok]
    print(f"images ok={len(results) - len(failed)} failed={len(failed)}")
    log_path = args.data_root / "image_download.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for rel, ok, note in results:
            fh.write(
                json.dumps({"image_path": rel, "ok": ok, "note": note})
                + "\n"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
