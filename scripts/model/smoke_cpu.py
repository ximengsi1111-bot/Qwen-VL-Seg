#!/usr/bin/env python3
"""Small end-to-end CPU smoke for the Qwen3-VL-Seg wrapper."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoProcessor

from qwen3vl_seg.data.dataset import SegmentationDataset
from qwen3vl_seg.model.model_wrapper import Qwen3VLSegForSegmentation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mingli01/models/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--manifest",
        default="/file_storage01/home/mingli/data/xyk/datasets/small/samples.jsonl",
    )
    parser.add_argument("--data-root", default="/file_storage01/home/mingli/data/xyk")
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(32)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dataset = SegmentationDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        processor=processor,
        split=args.split,
        limit=args.limit,
        max_pixels=args.max_pixels,
    )
    batch = dataset[args.index]
    print(
        {
            "sample_id": batch["sample_id"],
            "num_instances": batch["num_instances"],
            "image_grid": batch["image_grid"],
            "seq_len": batch["input_ids"].shape[1],
        },
        flush=True,
    )

    start = time.time()
    wrapper = Qwen3VLSegForSegmentation.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    wrapper.eval()
    print(f"model_load_seconds={time.time() - start:.1f}", flush=True)

    start = time.time()
    output = wrapper(batch)
    print(
        {
            "text_loss": None if output["text_loss"] is None else float(output["text_loss"]),
            "seg_loss": None if output.get("seg_loss") is None else float(output["seg_loss"]),
            "iou_mse_loss": (
                None if output.get("iou_mse_loss") is None else float(output["iou_mse_loss"])
            ),
            "total_loss": float(output["total_loss"]),
            "mask_logits_shape": tuple(output["mask_logits"].shape),
            "forward_seconds": round(time.time() - start, 1),
        },
        flush=True,
    )
    start = time.time()
    output["total_loss"].backward()
    print(f"backward_seconds={time.time() - start:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
