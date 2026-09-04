#!/usr/bin/env python3
"""Download public raw sources for the small reproduction dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCES = [
    (
        "coco",
        "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_trainval2017.zip",
    ),
    (
        "lvis",
        "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "lvis_v1_train.json.zip",
    ),
    (
        "lvis",
        "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
        "lvis_v1_val.json.zip",
    ),
]

REFCOCO_FILES = {
    "refcoco": [
        "test-00000-of-00001-82af0c1b600890ac.parquet",
        "testB-00000-of-00001-60990e4598892dc1.parquet",
        "train-00000-of-00001-94431d5f4bd5b93f.parquet",
        "validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
    ],
    "refcocoplus": [
        "test-00000-of-00001-2b8e5d26906553b9.parquet",
        "testB-00000-of-00001-4f1178d399f1874a.parquet",
        "train-00000-of-00001-7294665695c630ee.parquet",
        "validation-00000-of-00001-8c57d66282bc60c9.parquet",
    ],
    "refcocog": [
        "test-00000-of-00001-2316f36b19cd7f72.parquet",
        "train-00000-of-00001-4fe3e6340cfb69ed.parquet",
        "validation-00000-of-00001-15168dfe7b5961e5.parquet",
    ],
}


def make_sources() -> list[tuple[str, str, str]]:
    all_sources = list(SOURCES)
    for dataset, files in REFCOCO_FILES.items():
        for filename in files:
            url = (
                "https://huggingface.co/datasets/"
                f"jxu124/{dataset}/resolve/main/data/{filename}"
            )
            all_sources.append((dataset, url, filename))
    return all_sources


def download_one(task: tuple[str, str, str, Path, list[dict]]) -> bool:
    folder, url, filename, dest, log = task
    out = dest / folder / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        if out.exists() and out.stat().st_size > 0:
            log.append({"url": url, "path": str(out), "status": "skip"})
            return True
        with requests.get(url, stream=True, timeout=120, verify=False) as resp:
            resp.raise_for_status()
            with out.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
        log.append(
            {
                "url": url,
                "path": str(out),
                "size": out.stat().st_size,
                "seconds": round(time.time() - start, 1),
                "status": "ok",
            }
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.append({"url": url, "path": str(out), "status": "error", "error": str(exc)})
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    tasks = [
        (folder, url, filename, dest, [])
        for folder, url, filename in make_sources()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(download_one, tasks))
    log: list[dict] = []
    for task, ok in zip(tasks, results, strict=True):
        log.extend(task[-1])
    log_path = dest / "download.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in log:
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
    failed = sum(1 for ok in results if not ok)
    print(f"download complete: ok={len(results) - failed} failed={failed} log={log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
