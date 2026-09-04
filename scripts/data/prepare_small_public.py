#!/usr/bin/env python3
"""Build the small public dataset for the Qwen3-VL-Seg reproduction.

The script reads raw RefCOCO/RefCOCO+/RefCOCOg parquet files plus COCO2017 and
LVIS annotation JSON files, samples a deterministic small set, renders binary
masks, and writes a final samples.jsonl. Raw image files are not downloaded
here; download_coco_images.py consumes image_manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


DATASETS = ("refcoco", "refcocoplus", "refcocog")

PARQUET_FILES = {
    "refcoco": {
        "train": "train-00000-of-00001-94431d5f4bd5b93f.parquet",
        "val": "validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
        "testA": "test-00000-of-00001-82af0c1b600890ac.parquet",
        "testB": "testB-00000-of-00001-60990e4598892dc1.parquet",
    },
    "refcocoplus": {
        "train": "train-00000-of-00001-7294665695c630ee.parquet",
        "val": "validation-00000-of-00001-8c57d66282bc60c9.parquet",
        "testA": "test-00000-of-00001-2b8e5d26906553b9.parquet",
        "testB": "testB-00000-of-00001-4f1178d399f1874a.parquet",
    },
    "refcocog": {
        "train": "train-00000-of-00001-4fe3e6340cfb69ed.parquet",
        "val": "validation-00000-of-00001-15168dfe7b5961e5.parquet",
        "test": "test-00000-of-00001-2316f36b19cd7f72.parquet",
    },
}


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_zip_json(zip_path: Path, member: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            return json.loads(fh.read().decode("utf-8"))


def load_coco_annotations(source_root: Path) -> tuple[dict, dict]:
    zip_path = source_root / "coco2017" / "annotations_trainval2017.zip"
    if zip_path.exists():
        return (
            load_zip_json(zip_path, "annotations/instances_train2017.json"),
            load_zip_json(zip_path, "annotations/instances_val2017.json"),
        )
    return (
        load_json(source_root / "annotations" / "instances_train2017.json"),
        load_json(source_root / "annotations" / "instances_val2017.json"),
    )


def load_lvis_annotations(source_root: Path, split: str) -> dict:
    zip_path = source_root / "lvis" / f"lvis_v1_{split}.json.zip"
    if zip_path.exists():
        return load_zip_json(zip_path, f"lvis_v1_{split}.json")
    return load_json(source_root / f"lvis_v1_{split}.json")


def segmentation_to_mask(height: int, width: int, segmentation: object) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    if isinstance(segmentation, list) and segmentation:
        for polygon in segmentation:
            points = [
                (round(float(x)), round(float(y)))
                for x, y in np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            ]
            draw.polygon(points, fill=255)
    elif isinstance(segmentation, dict) and "counts" in segmentation:
        raise NotImplementedError("RLE segmentation is not implemented")
    return np.asarray(image, dtype=np.uint8)


def mask_nonempty(mask: np.ndarray) -> bool:
    return bool(np.any(mask))


def sample_rows_with_seed(
    rows: list[dict], count: int, rng: random.Random
) -> list[dict]:
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[i] for i in indices[:count]]


def sample_refcoco(
    source_root: Path,
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> tuple[list[dict], set[int]]:
    records: list[dict] = []
    train_image_ids: set[int] = set()
    sample_counter = 0

    for dataset_index, dataset in enumerate(DATASETS):
        files = PARQUET_FILES[dataset]
        for split_key in ("train", "val", "testA", "testB", "test"):
            filename = files.get(split_key)
            if filename is None:
                continue
            if split_key == "testA":
                count = test_count // 2
            elif split_key == "testB":
                count = test_count - test_count // 2
            elif split_key == "train":
                count = train_count
            elif split_key == "val":
                count = val_count
            else:
                count = test_count
            rows = pq.read_table(source_root / dataset / filename).to_pylist()
            rng = random.Random(seed * 100_003 + dataset_index * 1000 + len(split_key))
            for row in sample_rows_with_seed(rows, count, rng):
                ann = json.loads(row["raw_anns"])
                image_info = json.loads(row["raw_image_info"])
                sentences = row.get("sentences") or []
                if not sentences or "segmentation" not in ann:
                    continue
                sentence = sentences[rng.randrange(len(sentences))]["sent"]
                split_out = "train" if split_key == "train" else (
                    "val" if split_key == "val" else split_key
                )
                if split_key == "train":
                    train_image_ids.add(int(row["image_id"]))
                records.append(
                    {
                        "sample_id": f"{dataset}_{split_out}_{sample_counter:06d}",
                        "dataset": dataset,
                        "split": split_out,
                        "source": f"{dataset}:{split_out}",
                        "image_id": int(row["image_id"]),
                        "expression": sentence,
                        "label": str(row.get("category_id", "")),
                        "kind": "phrasal",
                        "image_size": [
                            int(image_info["height"]),
                            int(image_info["width"]),
                        ],
                        "target_anns": [
                            {
                                "id": int(ann["id"]),
                                "segmentation": ann.get("segmentation", []),
                                "bbox_xywh": ann.get("bbox", []),
                                "category_id": int(ann.get("category_id", -1)),
                            }
                        ],
                    }
                )
                sample_counter += 1
    return records, train_image_ids


def select_coco_category_samples(
    train_image_ids: set[int],
    coco_train: dict,
    max_images: int,
    target_samples: int,
    seed: int,
) -> list[dict]:
    images_by_id = {im["id"]: im for im in coco_train["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco_train["categories"]}
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco_train["annotations"]:
        if ann["image_id"] in train_image_ids and not ann.get("iscrowd"):
            ann_by_image[ann["image_id"]].append(ann)

    candidate_images = sorted(
        im_id
        for im_id, anns in ann_by_image.items()
        if any(ann.get("segmentation") for ann in anns)
    )
    rng = random.Random(seed + 10_001)
    selected_images = rng.sample(
        candidate_images, min(max_images, len(candidate_images))
    )
    chosen: list[tuple[int, int]] = []
    pair_set: set[tuple[int, int]] = set()
    for image_id in sorted(selected_images):
        cats = sorted({ann["category_id"] for ann in ann_by_image[image_id]})
        if not cats:
            continue
        rng.shuffle(cats)
        chosen.append((image_id, cats[0]))
        pair_set.add((image_id, cats[0]))
        for cat in cats[1:]:
            pair_set.add((image_id, cat))
    extra_pairs = sorted(pair_set - set(chosen))
    rng.shuffle(extra_pairs)
    chosen.extend(extra_pairs)
    chosen = chosen[:target_samples]

    records: list[dict] = []
    for rank, (image_id, category_id) in enumerate(chosen):
        image = images_by_id[image_id]
        anns = [
            ann
            for ann in ann_by_image[image_id]
            if ann["category_id"] == category_id and ann.get("segmentation")
        ]
        if not anns:
            continue
        records.append(
            {
                "sample_id": f"coco_train_{rank:06d}",
                "dataset": "coco",
                "split": "train",
                "source": "coco:train",
                "image_id": int(image_id),
                "expression": categories.get(category_id, "object"),
                "label": categories.get(category_id, "object"),
                "kind": "category_single" if len(anns) == 1 else "category_multi",
                "image_size": [int(image["height"]), int(image["width"])],
                "target_anns": [
                    {
                        "id": int(ann["id"]),
                        "segmentation": ann.get("segmentation", []),
                        "bbox_xywh": ann.get("bbox", []),
                        "category_id": int(ann["category_id"]),
                    }
                    for ann in anns
                ],
            }
        )
    return records


def select_lvis_category_samples(
    train_image_ids: set[int],
    lvis_train: dict,
    max_images: int,
    target_samples: int,
    seed: int,
) -> list[dict]:
    images_by_id = {im["id"]: im for im in lvis_train["images"]}
    cat_by_id = {
        cat["id"]: {"name": cat["name"], "frequency": cat["frequency"]}
        for cat in lvis_train["categories"]
    }
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in lvis_train["annotations"]:
        if ann["image_id"] in train_image_ids:
            ann_by_image[ann["image_id"]].append(ann)

    candidate_images = sorted(
        im_id
        for im_id, anns in ann_by_image.items()
        if any(ann.get("segmentation") for ann in anns)
    )
    rng = random.Random(seed + 20_001)
    selected_images = rng.sample(
        candidate_images, min(max_images, len(candidate_images))
    )
    high_pairs: list[tuple[int, int]] = []
    low_pairs: list[tuple[int, int]] = []
    for image_id in sorted(selected_images):
        cats = {
            ann["category_id"]
            for ann in ann_by_image[image_id]
            if ann.get("segmentation")
        }
        for cat in sorted(cats):
            freq = cat_by_id[cat]["frequency"] if cat in cat_by_id else "c"
            (high_pairs if freq in ("c", "f") else low_pairs).append((image_id, cat))
    high_pairs = sorted(set(high_pairs))
    low_pairs = sorted(set(low_pairs))

    high_n = min(1200, len(high_pairs))
    low_n = min(300, len(low_pairs))
    if high_n + low_n < target_samples:
        high_n = min(target_samples - low_n, len(high_pairs))
        low_n = min(target_samples - high_n, len(low_pairs))
    chosen = rng.sample(high_pairs, high_n) + rng.sample(low_pairs, low_n)

    records: list[dict] = []
    for rank, (image_id, category_id) in enumerate(chosen):
        image = images_by_id[image_id]
        meta = cat_by_id.get(category_id, {"name": "object", "frequency": "r"})
        anns = [
            ann
            for ann in ann_by_image[image_id]
            if ann["category_id"] == category_id and ann.get("segmentation")
        ]
        if not anns:
            continue
        records.append(
            {
                "sample_id": f"lvis_train_{rank:06d}",
                "dataset": "lvis",
                "split": "train",
                "source": "lvis:train",
                "image_id": int(image_id),
                "expression": meta["name"],
                "label": meta["name"],
                "kind": "category_single" if len(anns) == 1 else "category_multi",
                "image_size": [int(image["height"]), int(image["width"])],
                "target_anns": [
                    {
                        "id": int(ann["id"]),
                        "segmentation": ann.get("segmentation", []),
                        "bbox_xywh": ann.get("bbox", []),
                        "category_id": int(ann["category_id"]),
                    }
                    for ann in anns
                ],
            }
        )
    return records


def finalize_records(
    records: list[dict],
    coco_image_splits: dict[int, str],
    mask_root: Path,
    created_at: str,
) -> list[dict]:
    samples: list[dict] = []
    seen_masks: dict[str, str] = {}
    for rec in records:
        image_id = rec["image_id"]
        subdir = coco_image_splits.get(image_id)
        if subdir is None:
            continue
        rel_image = f"datasets/raw/coco2017/{subdir}/{image_id:012d}.jpg"
        masks: list[str] = []
        boxes: list[list[float]] = []
        height, width = rec["image_size"]
        for ann in rec["target_anns"]:
            mask_key = f"{rec['dataset']}_{ann['id']}"
            rel_mask = seen_masks.get(mask_key)
            if rel_mask is None:
                mask_array = segmentation_to_mask(height, width, ann["segmentation"])
                if not mask_nonempty(mask_array):
                    continue
                filename = f"mask_{mask_key}.png"
                mask_root.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask_array).save(mask_root / filename, compress_level=1)
                rel_mask = f"datasets/small/public/masks/{filename}"
                seen_masks[mask_key] = rel_mask
            masks.append(rel_mask)
            boxes.append(xywh_to_xyxy(ann["bbox_xywh"]))
        if not masks or len(masks) != len(boxes):
            continue
        samples.append(
            {
                "sample_id": rec["sample_id"],
                "split": rec["split"],
                "image_path": rel_image,
                "expression": rec["expression"],
                "kind": rec["kind"],
                "label": rec["label"],
                "bboxes": boxes,
                "mask_paths": masks,
                "image_size": rec["image_size"],
                "source": rec["source"],
                "created_at": created_at,
            }
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refcoco-train", type=int, default=1000)
    parser.add_argument("--refcoco-val", type=int, default=200)
    parser.add_argument("--refcoco-test", type=int, default=200)
    parser.add_argument("--coco-samples", type=int, default=1500)
    parser.add_argument("--lvis-samples", type=int, default=1500)
    parser.add_argument("--max-category-images", type=int, default=1000)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    build_root = args.build_root.resolve()
    datasets_root = build_root / "datasets"
    masks_root = datasets_root / "small" / "public" / "masks"
    created_at = "2026-09-03"

    records, train_image_ids = sample_refcoco(
        source_root,
        args.seed,
        args.refcoco_train,
        args.refcoco_val,
        args.refcoco_test,
    )
    print(f"refcoco records={len(records)} train_images={len(train_image_ids)}")

    coco_train, coco_val = load_coco_annotations(source_root)
    coco_category_names = {
        cat["id"]: cat["name"] for cat in coco_train["categories"]
    }
    for record in records:
        if record["dataset"].startswith("refcoco") and record["target_anns"]:
            cat_id = record["target_anns"][0]["category_id"]
            record["label"] = coco_category_names.get(
                cat_id, record["label"]
            )
    coco_image_splits: dict[int, str] = {}
    for split, payload in (("train2017", coco_train), ("val2017", coco_val)):
        for image in payload["images"]:
            coco_image_splits[int(image["id"])] = split

    coco_samples = select_coco_category_samples(
        train_image_ids,
        coco_train,
        args.max_category_images,
        args.coco_samples,
        args.seed,
    )
    print(f"coco category records={len(coco_samples)}")

    lvis_train = load_lvis_annotations(source_root, "train")
    lvis_samples = select_lvis_category_samples(
        train_image_ids,
        lvis_train,
        args.max_category_images,
        args.lvis_samples,
        args.seed,
    )
    print(f"lvis category records={len(lvis_samples)}")

    samples = finalize_records(
        records + coco_samples + lvis_samples,
        coco_image_splits,
        masks_root,
        created_at,
    )
    samples_path = datasets_root / "small" / "samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=True) + "\n")

    image_paths = sorted({sample["image_path"] for sample in samples})
    image_manifest = []
    for rel in image_paths:
        image_manifest.append(
            {
                "image_path": rel,
                "image_id": int(Path(rel).stem),
                "url": (
                    "https://images.cocodataset.org/"
                    f"{Path(rel).parent.name}/{Path(rel).name}"
                ),
            }
        )
    manifest_path = build_root / "image_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in image_manifest:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    stats = Counter((s["split"], s["source"].split(":")[0]) for s in samples)
    stats_path = build_root / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "samples": len(samples),
                "images": len(image_manifest),
                "by_split": {
                    split: sum(1 for s in samples if s["split"] == split)
                    for split in sorted({s["split"] for s in samples})
                },
                "details": {
                    f"{k[0]}|{k[1]}": v for k, v in sorted(stats.items())
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"final samples={len(samples)} images={len(image_manifest)} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
