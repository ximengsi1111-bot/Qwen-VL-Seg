"""Sharded multi-GPU inference runner: shard samples across ranks, output partial metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from qwen3vl_seg.data.dataset import SegmentationDataset, normalize_decoder_image
from qwen3vl_seg.eval.metrics import mask_iou, match_masks
from qwen3vl_seg.model.model_wrapper import Qwen3VLSegForSegmentation
from qwen3vl_seg.model.prompt_format import (
    MASK_END,
    MASK_START,
    MASK_TOKEN,
    build_user_prompt,
)
from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--splits", default="val,testA,testB,test")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-bucket", action="store_true")
    parser.add_argument("--bucket-sizes", default="256,384,512,640,768")

    return parser.parse_args()


def _load_stage2(wrapper: Qwen3VLSegForSegmentation, checkpoint: str) -> None:
    ckpt = Path(checkpoint)
    ds_dir = ckpt / "ds"
    if not ds_dir.exists():
        ds_dir = ckpt
    latest = (ds_dir / "latest").read_text(encoding="utf-8").strip()
    model_states = ds_dir / latest / "mp_rank_00_model_states.pt"
    state = torch.load(model_states, map_location="cpu", weights_only=False)
    module = state.get("module", state)
    result = wrapper.load_state_dict(module, strict=False)
    if result.unexpected_keys:
        raise RuntimeError("unexpected keys: " + ", ".join(result.unexpected_keys[:8]))


def _generation_messages(sample: Any) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": build_user_prompt(sample)},
            ],
        }
    ]


def _encode(
    wrapper: Qwen3VLSegForSegmentation,
    messages: list[dict[str, Any]],
    image: Image.Image,
    max_pixels: int,
    device: torch.device,
    add_generation_prompt: bool,
):
    content = messages[0]["content"]
    content[0]["image"] = image
    encoded = wrapper.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        return_dict=True,
        processor_kwargs={
            "images_kwargs": {
                "max_pixels": max_pixels,
                "min_pixels": 56 * 56,
            }
        },
    )
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in encoded.items()}


def _inject_mask_placeholders(text: str) -> str:
    return re.sub(
        r'("mask"\s*:\s*")[^"]*(")',
        lambda m: (
            m.group(1) + MASK_START + MASK_TOKEN + MASK_END + m.group(2)
        ),
        text,
    )


def _panel(
    image: Image.Image,
    mask: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    color: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    canvas = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (width, height), Image.BILINEAR
    )
    base = np.array(image.convert("RGB"), dtype=np.float32)
    alpha = np.array(canvas, dtype=np.float32) / 255.0
    overlay = base * (1.0 - 0.6 * alpha[..., None]) + (
        np.array(color, dtype=np.float32) * 0.6 * alpha[..., None]
    )
    out = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 90, 255), width=3)
    return out


def _save_mask_viz(
    image: Image.Image,
    pred_mask: np.ndarray,
    pred_bbox_1000: list[float],
    gt_masks: np.ndarray,
    gt_boxes_norm: np.ndarray,
    path: Path,
) -> None:
    width, height = image.size
    pred_boxes = [
        (
            float(pred_bbox_1000[0]) / 1000.0 * width,
            float(pred_bbox_1000[1]) / 1000.0 * height,
            float(pred_bbox_1000[2]) / 1000.0 * width,
            float(pred_bbox_1000[3]) / 1000.0 * height,
        )
    ]
    gt_boxes = []
    if gt_boxes_norm is not None and gt_boxes_norm.ndim == 2:
        for box in gt_boxes_norm:
            gt_boxes.append((box[0] * width, box[1] * height, box[2] * width, box[3] * height))
    gt_mask = np.asarray(gt_masks)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.max(axis=0)
    left = _panel(image, np.asarray(pred_mask), pred_boxes, (255, 0, 0))
    right = _panel(image, gt_mask, gt_boxes, (0, 200, 0))
    combined = Image.new("RGB", (left.width * 2, left.height))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    combined.save(path)


def main() -> int:
    args = _parse_args()
    # Read torchrun-injected rank info; fall back to CLI/defaults.
    shard_rank = int(os.environ.get("LOCAL_RANK", str(args.shard_rank)))
    shard_size = int(os.environ.get("WORLD_SIZE", str(args.shard_size)))
    device = torch.device(f"cuda:{shard_rank}" if torch.cuda.is_available() else "cpu")

    wrapper = Qwen3VLSegForSegmentation.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    _load_stage2(wrapper, args.checkpoint)
    if args.adapter:
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(wrapper.base, args.adapter)
        wrapper.base = peft_model.merge_and_unload()
        print(f"[runner_multi] merged LoRA adapter: {args.adapter}", flush=True)
    wrapper.to(device, dtype=torch.bfloat16)
    wrapper.eval()
    # Qwen3-VL is decoder-only: batch generation requires LEFT padding.
    wrapper.processor.tokenizer.padding_side = "left"

    _bucket_sizes = tuple(int(x) for x in args.bucket_sizes.split(",")) if args.bucket_sizes else None
    dataset = SegmentationDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        processor=wrapper.processor,
        split=None,
        max_pixels=args.max_pixels,
        use_bucket=args.use_bucket,
        bucket_sizes=_bucket_sizes,
    )
    if args.splits:
        keep = {s.strip() for s in args.splits.split(",") if s.strip()}
        dataset.samples = [s for s in dataset.samples if s.split in keep]
    output_dir = Path(args.output_dir or "preds")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_viz = args.save_viz or not args.no_viz
    if save_viz:
        (output_dir / "viz").mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions.rank{shard_rank}.jsonl"
    results: list[dict[str, Any]] = []
    all_ious: list[float] = []
    sum_inter = 0.0
    sum_union = 0.0
    strict_values: list[float] = []
    strict_correct = 0
    strict_total = 0

    def _pad_img(img, bucket):
        if bucket:
            from qwen3vl_seg.data.buckets import pad_right_bottom
            img, _, _ = pad_right_bottom(img, bucket)
        return img

    def _batch_encode(msgs, add_gen):
        enc = wrapper.processor.apply_chat_template(
            msgs, tokenize=True, padding=True, add_generation_prompt=add_gen,
            return_tensors="pt", return_dict=True,
            processor_kwargs={"images_kwargs": {"max_pixels": args.max_pixels, "min_pixels": 56 * 56}},
        )
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

    def _decode_text(out_row, input_len_b):
        raw = wrapper.processor.decode(out_row[input_len_b:], skip_special_tokens=False)
        return raw.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

    def process_group(indices, bucket):
        nonlocal sum_inter, sum_union, strict_correct, strict_total
        chunk_s = [dataset.samples[i] for i in indices]
        chunk_item = [dataset[i] for i in indices]
        chunk_img = []
        msgs = []
        for s in chunk_s:
            img = Image.open(Path(args.data_root) / s.image_path).convert("RGB")
            img = _pad_img(img, bucket)
            chunk_img.append(img)
            um = _generation_messages(s)
            um[0]["content"][0]["image"] = img
            msgs.append(um)
        enc = _batch_encode(msgs, True)
        input_len = enc["input_ids"].shape[1]
        out = wrapper.base.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"], image_grid_thw=enc["image_grid_thw"],
            mm_token_type_ids=enc["mm_token_type_ids"],
            max_new_tokens=256, do_sample=False, use_cache=True,
        )
        # per-sample decode + parse
        rows = []
        valid = []
        for b, (sample, item, image) in enumerate(zip(chunk_s, chunk_item, chunk_img)):
            assistant_text = _decode_text(out[b], input_len)
            records = parse_generated_boxes_and_masks(assistant_text)
            decoder_text = assistant_text
            if records and assistant_text.count(MASK_TOKEN) == 0:
                decoder_text = _inject_mask_placeholders(assistant_text)
            row = {"sample_id": sample.sample_id, "generated_text": assistant_text,
                   "num_records": len(records), "bbox_2d": [r["bbox_2d"] for r in records],
                   "mask_tokens": decoder_text.count(MASK_TOKEN)}
            rows.append((row, sample, item, image, records, decoder_text))
            if records and decoder_text.count(MASK_TOKEN) == len(records):
                valid.append((row, sample, item, image, records, decoder_text))

        # BATCH decoder for valid samples (bucket mode: same grid).
        if valid and bucket is not None:
            # Encode each sample individually (no padding) so the image grid respects
            # max_pixels (matches the dataset item), then pad text manually and concat
            # pixel_values so the decoder forward is batched.
            encs = []
            for (row, sample, item, image, records, decoder_text) in valid:
                um = _generation_messages(sample)
                um[0]["content"][0]["image"] = image
                fm = [um[0], {"role": "assistant", "content": decoder_text}]
                e = wrapper.processor.apply_chat_template(
                    fm, tokenize=True, add_generation_prompt=False,
                    return_tensors="pt", return_dict=True,
                    processor_kwargs={"images_kwargs": {"max_pixels": args.max_pixels, "min_pixels": 56 * 56}},
                )
                encs.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in e.items()})
            grid = encs[0]["image_grid_thw"][0]
            grid_h, grid_w = int(grid[1]), int(grid[2])
            stride_h, stride_w = grid_h * 2, grid_w * 2
            max_len = max(e["input_ids"].shape[1] for e in encs)

            def _pad(key_, padval):
                return torch.stack([
                    F.pad(e[key_].detach().squeeze(0), (max_len - e[key_].shape[1], 0), value=padval)
                    for e in encs
                ], dim=0)

            input_ids = _pad("input_ids", 0)
            attention_mask = _pad("attention_mask", 0)
            mm_tok = _pad("mm_token_type_ids", 0)
            pixel_values = torch.cat([e["pixel_values"].detach() for e in encs], dim=0)
            image_grid_thw = torch.stack([e["image_grid_thw"][0].detach() for e in encs], dim=0)
            max_inst = max(len(r[4]) for r in valid)
            boxes_list = []
            masks_list = []
            image_list = []
            num_inst_list = []
            for (row, sample, item, image, records, decoder_text) in valid:
                num = len(records)
                boxes = torch.tensor([[v / 1000.0 for v in r["bbox_2d"]] for r in records], dtype=torch.float32)
                if num < max_inst:
                    pad = torch.zeros(max_inst - num, 4, dtype=torch.float32)
                    boxes = torch.cat([boxes, pad], dim=0)
                boxes_list.append(boxes)
                masks_list.append(torch.zeros(num, stride_h, stride_w, dtype=torch.float32).to(device))
                image_list.append(item["image"].detach().to(device, dtype=torch.float32))
                num_inst_list.append(num)
            batch = {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "mm_token_type_ids": mm_tok,
                "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
                "boxes": torch.stack(boxes_list, dim=0).to(device),
                "masks": masks_list, "num_instances": num_inst_list,
                "has_seg": [True] * len(valid), "image": image_list,
            }
            out_dec = wrapper(batch)
            logits_all = out_dec["mask_logits"]           # [V, max_inst, stride_h, stride_w]
            iou_all = out_dec["iou_scores"]
            for vi, (row, sample, item, image, records, decoder_text) in enumerate(valid):
                num = len(records)
                logits = logits_all[vi][:num]
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                iou_preds = iou_all[vi][:num].float().cpu().numpy()
                gt = item["masks"].cpu().numpy()
                pred_list = [preds[j] for j in range(num)]
                gt_list = [np.asarray(gt[j]) for j in range(gt.shape[0])]
                pred_idx, gt_idx, total = match_masks(pred_list, gt_list)
                matched_ious = []
                for pj, gj in zip(pred_idx, gt_idx):
                    iou = float(mask_iou(np.expand_dims(pred_list[pj], 0), np.expand_dims(gt_list[gj], 0)))
                    matched_ious.append(iou)
                    inter = np.logical_and(pred_list[pj] > 0, gt_list[gj] > 0).sum()
                    union = np.logical_or(pred_list[pj] > 0, gt_list[gj] > 0).sum()
                    sum_inter += float(inter)
                    sum_union += float(union)
                all_ious.extend(matched_ious)
                strict_correct += sum(1 for iou in matched_ious if iou >= 0.5)
                strict_total += len(gt_list)
                strict = total / max(len(gt_list), 1)
                row["mask_ious"] = matched_ious
                row["iou_scores"] = [float(v) for v in iou_preds]
                row["miou"] = strict
                strict_values.append(strict)
                if save_viz:
                    _save_mask_viz(image, preds[0], records[0]["bbox_2d"], gt,
                                   item["boxes"].detach().cpu().numpy(),
                                   output_dir / "viz" / f"{sample.sample_id}_cmp.png")
        elif valid:
            for (row, sample, item, image, records, decoder_text) in valid:
                user_msg = _generation_messages(sample)
                user_msg[0]["content"][0]["image"] = image
                full_messages = [user_msg[0], {"role": "assistant", "content": decoder_text}]
                full = wrapper.processor.apply_chat_template(
                    full_messages, tokenize=True, add_generation_prompt=False,
                    return_tensors="pt", return_dict=True,
                    processor_kwargs={"images_kwargs": {"max_pixels": args.max_pixels, "min_pixels": 56 * 56}},
                )
                full = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in full.items()}
                grid = full["image_grid_thw"][0]
                grid_h, grid_w = int(grid[1]), int(grid[2])
                stride_h, stride_w = grid_h * 2, grid_w * 2
                num = len(records)
                boxes = torch.tensor([[v / 1000.0 for v in r["bbox_2d"]] for r in records], dtype=torch.float32).to(device)
                masks = torch.zeros(num, stride_h, stride_w, dtype=torch.float32)
                image_tensor = normalize_decoder_image(image, grid_h * 16, grid_w * 16).unsqueeze(0)
                batch = {
                    "input_ids": full["input_ids"], "attention_mask": full["attention_mask"],
                    "mm_token_type_ids": full["mm_token_type_ids"], "pixel_values": full["pixel_values"],
                    "image_grid_thw": full["image_grid_thw"], "boxes": boxes.unsqueeze(0),
                    "masks": masks.unsqueeze(0).to(device), "num_instances": [num],
                    "has_seg": True, "image": image_tensor.to(device),
                }
                out_dec = wrapper(batch)
                logits = out_dec["mask_logits"][0]
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                iou_preds = out_dec["iou_scores"][0].float().cpu().numpy()
                gt = item["masks"].cpu().numpy()
                pred_list = [preds[i] for i in range(num)]
                gt_list = [np.asarray(gt[j]) for j in range(gt.shape[0])]
                pred_idx, gt_idx, total = match_masks(pred_list, gt_list)
                matched_ious = []
                for pj, gj in zip(pred_idx, gt_idx):
                    iou = float(mask_iou(np.expand_dims(pred_list[pj], 0), np.expand_dims(gt_list[gj], 0)))
                    matched_ious.append(iou)
                    inter = np.logical_and(pred_list[pj] > 0, gt_list[gj] > 0).sum()
                    union = np.logical_or(pred_list[pj] > 0, gt_list[gj] > 0).sum()
                    sum_inter += float(inter)
                    sum_union += float(union)
                all_ious.extend(matched_ious)
                strict_correct += sum(1 for iou in matched_ious if iou >= 0.5)
                strict_total += len(gt_list)
                strict = total / max(len(gt_list), 1)
                row["mask_ious"] = matched_ious
                row["iou_scores"] = [float(v) for v in iou_preds]
                row["miou"] = strict
                strict_values.append(strict)
                if save_viz:
                    _save_mask_viz(image, preds[0], records[0]["bbox_2d"], gt,
                                   item["boxes"].detach().cpu().numpy(),
                                   output_dir / "viz" / f"{sample.sample_id}_cmp.png")

        # non-valid rows get miou 0
        for (row, sample, item, image, records, decoder_text) in rows:
            if "miou" not in row:
                row["miou"] = 0.0
                row["mask_ious"] = []
                strict_values.append(0.0)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    with torch.inference_mode():
        total = len(dataset)
        limit = args.max_samples if args.max_samples > 0 else total
        sample_indices = list(range(shard_rank, min(limit, total), shard_size))
        batch_size = max(1, int(args.batch_size))
        if args.use_bucket:
            from qwen3vl_seg.data.buckets import choose_bucket
            groups: dict[int, list[int]] = {}
            for i in sample_indices:
                s = dataset.samples[i]
                h, w = (int(v) for v in s.image_size)
                b = choose_bucket(h, w, _bucket_sizes)
                groups.setdefault(b, []).append(i)
            for b in sorted(groups):
                process_group(groups[b], b)
        else:
            for bstart in range(0, len(sample_indices), batch_size):
                process_group(sample_indices[bstart:bstart + batch_size], None)

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_samples = len(results)
    partial = {
        "rank": shard_rank,
        "shard_size": shard_size,
        "n_samples": n_samples,
        "sum_strict": float(np.sum(strict_values)) if strict_values else 0.0,
        "sum_inter": float(sum_inter),
        "sum_union": float(sum_union),
        "matched_correct": int(np.sum(np.asarray(all_ious) >= 0.5)) if all_ious else 0,
        "matched_total": int(len(all_ious)),
        "strict_correct": int(strict_correct),
        "strict_total": int(strict_total),
        "predictions": str(out_path.name),
    }
    (output_dir / f"metrics.partial.{shard_rank}.json").write_text(
        json.dumps(partial, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rank": shard_rank, "n_samples": n_samples}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())