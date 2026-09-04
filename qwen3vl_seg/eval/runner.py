"""Minimal inference runner: generate JSON -> parse -> decode masks -> metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from qwen3vl_seg.data.dataset import SegmentationDataset, normalize_decoder_image
from qwen3vl_seg.eval.metrics import mask_iou
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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-viz", action="store_true")
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


def _save_mask_viz(
    image: Image.Image,
    mask: np.ndarray,
    bbox: list[float],
    path: Path,
) -> None:
    width, height = image.size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (width, height), Image.BILINEAR
    )
    base = np.array(image.convert("RGB"), dtype=np.float32)
    alpha = np.array(mask_img, dtype=np.float32) / 255.0
    overlay = base * (1.0 - 0.6 * alpha[..., None]) + (
        np.array([255, 0, 0], dtype=np.float32) * 0.6 * alpha[..., None]
    )
    out = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [float(v) / 1000.0 * s for v, s in zip(bbox, (width, height, width, height))]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 90, 255), width=3)
    out.save(path)


def main() -> int:
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = Qwen3VLSegForSegmentation.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    _load_stage2(wrapper, args.checkpoint)
    wrapper.to(device, dtype=torch.bfloat16)
    wrapper.eval()

    dataset = SegmentationDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        processor=wrapper.processor,
        split=None,
        max_pixels=args.max_pixels,
    )
    output_dir = Path(args.output_dir or "preds")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        (output_dir / "viz").mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.jsonl"
    results: list[dict[str, Any]] = []

    with torch.inference_mode():
        for idx in range(min(args.max_samples, len(dataset))):
            sample = dataset.samples[idx]
            item = dataset[idx]
            image_path = Path(args.data_root) / sample.image_path
            image = Image.open(image_path).convert("RGB")
            user_msg = _generation_messages(sample)
            enc = _encode(
                wrapper, user_msg, image, args.max_pixels, device,
                add_generation_prompt=True,
            )
            input_len = enc["input_ids"].shape[1]
            out = wrapper.base.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
                mm_token_type_ids=enc["mm_token_type_ids"],
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )
            raw = wrapper.processor.decode(
                out[0][input_len:], skip_special_tokens=False
            )
            assistant_text = (
                raw.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            )
            records = parse_generated_boxes_and_masks(assistant_text)
            decoder_text = assistant_text
            if records and assistant_text.count(MASK_TOKEN) == 0:
                decoder_text = _inject_mask_placeholders(assistant_text)
            row: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "generated_text": assistant_text,
                "num_records": len(records),
                "bbox_2d": [r["bbox_2d"] for r in records],
                "mask_tokens": decoder_text.count(MASK_TOKEN),
            }

            if records and decoder_text.count(MASK_TOKEN) == len(records):
                full_messages = [
                    user_msg[0],
                    {"role": "assistant", "content": decoder_text},
                ]
                # Re-encode with the actual image object.
                full_messages[0]["content"][0]["image"] = image
                full = wrapper.processor.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    return_dict=True,
                    processor_kwargs={
                        "images_kwargs": {
                            "max_pixels": args.max_pixels,
                            "min_pixels": 56 * 56,
                        }
                    },
                )
                full = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in full.items()
                }
                grid = full["image_grid_thw"][0]
                grid_h, grid_w = int(grid[1]), int(grid[2])
                stride_h, stride_w = grid_h * 2, grid_w * 2
                num = len(records)
                boxes = torch.tensor(
                    [[v / 1000.0 for v in r["bbox_2d"]] for r in records],
                    dtype=torch.float32,
                ).to(device)
                masks = torch.zeros(num, stride_h, stride_w, dtype=torch.float32)
                image_tensor = normalize_decoder_image(
                    image, grid_h * 16, grid_w * 16
                ).unsqueeze(0)
                batch = {
                    "input_ids": full["input_ids"],
                    "attention_mask": full["attention_mask"],
                    "mm_token_type_ids": full["mm_token_type_ids"],
                    "pixel_values": full["pixel_values"],
                    "image_grid_thw": full["image_grid_thw"],
                    "boxes": boxes.unsqueeze(0),
                    "masks": masks.unsqueeze(0).to(device),
                    "num_instances": [num],
                    "has_seg": True,
                    "image": image_tensor.to(device),
                }
                out_dec = wrapper(batch)
                logits = out_dec["mask_logits"][0]
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                iou_preds = out_dec["iou_scores"][0].float().cpu().numpy()
                gt = item["masks"].cpu().numpy()
                ious = []
                for j in range(num):
                    if j < gt.shape[0]:
                        ious.append(
                            float(
                                mask_iou(
                                    np.expand_dims(preds[j], 0),
                                    np.expand_dims(gt[j], 0),
                                )
                            )
                        )
                    else:
                        ious.append(0.0)
                row["mask_ious"] = ious
                row["iou_scores"] = [float(v) for v in iou_preds]
                row["miou"] = float(np.mean(ious)) if ious else 0.0
                if args.save_viz:
                    _save_mask_viz(
                        image,
                        preds[0],
                        records[0]["bbox_2d"],
                        output_dir / "viz" / f"{sample.sample_id}_pred.png",
                    )
            else:
                row["miou"] = 0.0
                row["mask_ious"] = []

            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"done {len(results)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
