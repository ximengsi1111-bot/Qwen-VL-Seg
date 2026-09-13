"""Helpers for computing GRPO rewards from the live policy's features.

The reward itself is non-differentiable.  These helpers consume hidden states
captured from the same forward pass that computes ``old_per_token_logps`` and
run only the frozen mask decoder under ``torch.no_grad()``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from qwen3vl_seg.data.dataset import normalize_decoder_image
from qwen3vl_seg.eval.metrics import match_masks
from qwen3vl_seg.model.mask_decoder import BoxGuidedMaskDecoder, MaskDecoderConfig
from qwen3vl_seg.model.prompt_format import MASK_TOKEN
from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks
from qwen3vl_seg.model.vision_hooks import ModelFeatureHooks, extract_per_image_features


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _parse_solution(solution: Any) -> dict[str, Any]:
    if isinstance(solution, dict):
        return solution
    if not solution:
        return {}
    try:
        value = json.loads(solution)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _completion_text(sample: Any) -> str:
    messages = getattr(sample, "messages", None) or []
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    return content if isinstance(content, str) else str(content)


def _box_reward(records: list[dict[str, Any]], gt: dict[str, Any]) -> float:
    gt_boxes = gt.get("bbox_1000", [])
    pred_boxes = [r.get("bbox_2d") for r in records]
    pred_boxes = [b for b in pred_boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not gt_boxes or not pred_boxes:
        return 0.0
    return max(_box_iou(pred, gt_box) for pred in pred_boxes for gt_box in gt_boxes)


def _load_gt_masks(gt: dict[str, Any], stride_h: int, stride_w: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for mask_path in gt.get("mask_paths", []):
        with Image.open(mask_path) as handle:
            mask = handle.convert("L").resize((stride_w, stride_h), Image.BILINEAR)
        masks.append(np.asarray(mask, dtype=np.uint8))
    return masks


def _extract_decoder_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    module = state.get("module", state)
    if not isinstance(module, dict):
        raise RuntimeError("stage2 checkpoint does not contain a state dict")

    prefixes = ("decoder.", "model.decoder.", "base.decoder.", "module.decoder.")
    for prefix in prefixes:
        selected = {k[len(prefix):]: v for k, v in module.items() if k.startswith(prefix)}
        if selected:
            return selected

    selected = {}
    for key, value in module.items():
        marker = "decoder."
        if marker in key:
            selected[key.split(marker, 1)[1]] = value
    if not selected:
        raise RuntimeError(
            "could not find decoder weights in stage2 checkpoint; "
            f"sample keys={list(module)[:8]}"
        )
    return selected


def load_frozen_mask_decoder(checkpoint: str, device: torch.device, dtype: torch.dtype) -> BoxGuidedMaskDecoder:
    """Load only the trained ``decoder.*`` weights, not the 4B base model."""
    ckpt = Path(checkpoint)
    ds_dir = ckpt / "ds"
    if not ds_dir.exists():
        ds_dir = ckpt
    latest = (ds_dir / "latest").read_text(encoding="utf-8").strip()
    state_path = ds_dir / latest / "mp_rank_00_model_states.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"stage2 model state not found: {state_path}")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    decoder_state = _extract_decoder_state_dict(state)
    decoder = BoxGuidedMaskDecoder(MaskDecoderConfig())
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "decoder checkpoint mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    decoder.to(device=device, dtype=dtype)
    decoder.eval()
    decoder.requires_grad_(False)
    return decoder


def resolve_policy_base(model: Any) -> Any:
    """Resolve a DDP/DeepSpeed/PEFT wrapper to the Qwen3-VL base module."""
    import torch.nn as nn

    candidate = model
    while hasattr(candidate, "module") and isinstance(candidate.module, nn.Module):
        candidate = candidate.module

    if hasattr(candidate, "get_base_model"):
        base = candidate.get_base_model()
        if base is not candidate:
            candidate = base

    if not (
        hasattr(candidate, "model")
        and hasattr(candidate.model, "visual")
        and hasattr(candidate.model, "language_model")
    ):
        raise TypeError(
            "could not resolve Qwen3-VL base module from "
            f"{type(candidate).__module__}.{type(candidate).__name__}"
        )
    return candidate


def score_masks_from_policy_forward(
    *,
    decoder: BoxGuidedMaskDecoder,
    tokenizer: Any,
    hooks: ModelFeatureHooks,
    model_inputs: dict[str, Any],
    samples: Sequence[Any],
    device: torch.device,
    decoder_dtype: torch.dtype,
) -> list[dict[str, float]]:
    """Decode masks from the hidden states of one current-policy forward batch."""
    hidden_states = hooks.text_last_hidden
    if hidden_states is None:
        raise RuntimeError("policy forward did not expose text_last_hidden")

    input_ids = model_inputs["input_ids"]
    mm_token_type_ids = model_inputs["mm_token_type_ids"]
    image_grid_thw = model_inputs.get("image_grid_thw")
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        raise RuntimeError("policy forward did not contain image_grid_thw")

    per_image = extract_per_image_features(
        hooks, hidden_states, mm_token_type_ids, image_grid_thw
    )
    image_rows = [
        row
        for row in range(mm_token_type_ids.shape[0])
        if bool((mm_token_type_ids[row] == 1).any())
    ]
    if len(image_rows) != len(samples):
        raise RuntimeError(
            f"expected one image per sample, got {len(image_rows)} image rows for {len(samples)} samples"
        )

    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    if mask_token_id is None:
        raise RuntimeError(f"tokenizer does not define {MASK_TOKEN}")

    image_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    gt_mask_cache: dict[tuple[tuple[str, ...], int, int], list[np.ndarray]] = {}
    results: list[dict[str, float]] = []

    for image_index, row in enumerate(image_rows):
        sample = samples[row]
        completion = _completion_text(sample)
        gt = _parse_solution((getattr(sample, "extra", None) or {}).get("solution"))
        records = parse_generated_boxes_and_masks(completion)
        box_iou = _box_reward(records, gt)
        stat = {
            "box_iou": float(box_iou),
            "mask_iou": 0.0,
            "parse_fail": float(not records),
            "missing_mask_token": 0.0,
            "mask_count_mismatch": 0.0,
            "decode_fail": 0.0,
        }

        mask_positions = (input_ids[row] == mask_token_id).nonzero(as_tuple=False).flatten()
        if mask_positions.numel() == 0:
            stat["missing_mask_token"] = 1.0
            results.append(stat)
            continue
        if mask_positions.numel() != len(records):
            stat["mask_count_mismatch"] = 1.0
            results.append(stat)
            continue

        try:
            grid = image_grid_thw[row]
            grid_h, grid_w = int(grid[1]), int(grid[2])
            stride_h, stride_w = grid_h * 2, grid_w * 2

            gt_paths = tuple(str(mask_path) for mask_path in gt.get("mask_paths", []))
            gt_key = (gt_paths, stride_h, stride_w)
            gt_masks = gt_mask_cache.get(gt_key)
            if gt_masks is None:
                gt_masks = _load_gt_masks(gt, stride_h, stride_w)
                gt_mask_cache[gt_key] = gt_masks
            if not gt_masks:
                stat["decode_fail"] = 1.0
                results.append(stat)
                continue

            image_path = str(gt.get("image_path", ""))
            image_key = (image_path, grid_h, grid_w)
            image_tensor = image_cache.get(image_key)
            if image_tensor is None:
                with Image.open(image_path) as image_handle:
                    image_tensor = normalize_decoder_image(
                        image_handle.convert("RGB"), grid_h * 16, grid_w * 16
                    ).unsqueeze(0)
                image_tensor = image_tensor.to(device=device, dtype=decoder_dtype)
                image_cache[image_key] = image_tensor

            boxes = torch.tensor(
                [[float(v) / 1000.0 for v in record["bbox_2d"]] for record in records],
                dtype=torch.float32,
                device=device,
            )
            feature = per_image[image_index]
            seg_features = hidden_states[row, mask_positions, :].unsqueeze(0)
            with torch.no_grad():
                decoded = decoder(
                    visual_features=feature["visual"],
                    mm_features=feature["mm"].to(seg_features.dtype),
                    seg_features=seg_features.to(decoder_dtype),
                    boxes=boxes.unsqueeze(0).to(decoder_dtype),
                    image=image_tensor,
                )
            logits = decoded["mask_logits"][0]
            preds = (torch.sigmoid(logits[: len(records)]) > 0.5).float().cpu().numpy()
            pred_list = [preds[index] for index in range(len(preds))]
            _, _, total = match_masks(pred_list, gt_masks)
            stat["mask_iou"] = float(total / max(len(gt_masks), 1))
        except Exception:
            stat["decode_fail"] = 1.0
        results.append(stat)

    return results
