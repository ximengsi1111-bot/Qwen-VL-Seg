"""Bucket-batched oracle-box inference smoke with original-order restore."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen3vl_seg.data.buckets import choose_bucket
from qwen3vl_seg.data.collator import SegmentationCollator
from qwen3vl_seg.data.dataset import SegmentationDataset
from qwen3vl_seg.eval.metrics import mask_iou
from qwen3vl_seg.model.lora import apply_lora
from qwen3vl_seg.model.model_wrapper import Qwen3VLSegForSegmentation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-bucket", action="store_true")
    parser.add_argument("--bucket-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--split", default=None)
    return parser.parse_args()


def _load_stage2(wrapper: Qwen3VLSegForSegmentation, checkpoint: str) -> None:
    ckpt = Path(checkpoint)
    ds_dir = ckpt / "ds"
    if ds_dir.exists():
        latest = (ds_dir / "latest").read_text(encoding="utf-8").strip()
        state = torch.load(
            ds_dir / latest / "mp_rank_00_model_states.pt",
            map_location="cpu",
            weights_only=False,
        )
        module = state.get("module", state)
    else:
        native = ckpt / "latest" / "model_trainable.pt"
        if not native.exists():
            native = ckpt / "model_trainable.pt"
        module = torch.load(native, map_location="cpu", weights_only=False)
    result = wrapper.load_state_dict(module, strict=False)
    if result.unexpected_keys:
        raise RuntimeError("unexpected keys: " + ", ".join(result.unexpected_keys[:8]))


def main() -> int:
    args = _parse_args()
    bucket_sizes = tuple(args.bucket_sizes) if args.bucket_sizes else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = Qwen3VLSegForSegmentation.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    apply_lora(
        wrapper.base.model.language_model,
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        rank=32,
        alpha=32,
    )
    _load_stage2(wrapper, args.checkpoint)
    wrapper.to(device, dtype=torch.bfloat16)
    wrapper.eval()

    dataset = SegmentationDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        processor=wrapper.processor,
        split=args.split,
        max_pixels=args.max_pixels,
        use_bucket=not args.no_bucket,
        bucket_sizes=bucket_sizes,
    )
    n = min(args.max_samples, len(dataset))
    items = [dataset[i] for i in range(n)]
    groups: dict[int, list[tuple[int, dict]]] = {}
    for idx, item in enumerate(items):
        h, w = item["orig_size"]
        groups.setdefault(choose_bucket(h, w, bucket_sizes), []).append((idx, item))

    results: list[dict] = [None] * n
    collate = SegmentationCollator(wrapper.tokenizer.pad_token_id or 0)
    chunk_size = 1 if args.no_bucket else 2
    out_dir = Path(args.output_dir or "preds_bucket")
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for bucket in sorted(groups):
            entries = groups[bucket]
            # Micro-batch within bucket.
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                item_batch = [item for _, item in chunk]
                batch = collate(item_batch)
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                out = wrapper(batch)
                mask_logits = out["mask_logits"]  # (B, maxN, H, W)
                for j, (orig_idx, item) in enumerate(chunk):
                    num = int(item["num_instances"])
                    pred = (torch.sigmoid(mask_logits[j][:num]) > 0.5).float().cpu().numpy()
                    gt = item["masks"].numpy()
                    ious = []
                    for k in range(num):
                        if k < gt.shape[0]:
                            ious.append(
                                float(
                                    mask_iou(
                                        np.expand_dims(pred[k], 0),
                                        np.expand_dims(gt[k], 0),
                                    )
                                )
                            )
                        else:
                            ious.append(0.0)
                    results[orig_idx] = {
                        "sample_id": item["sample_id"],
                        "bucket": bucket,
                        "num_instances": num,
                        "miou": float(np.mean(ious)) if ious else 0.0,
                        "mask_ious": ious,
                    }

    ordered = [r for r in results if r is not None]
    out_path = out_dir / "predictions_bucket.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in ordered:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"done {len(ordered)} -> {out_path}")
    for row in ordered:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
