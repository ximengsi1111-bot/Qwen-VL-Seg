"""Joint token+mask GRPO policy for Qwen3-VL-Seg.

Factorised joint policy::

    pi_theta(y, m | x) = pi_theta(y | x) * pi_phi(m | x, y)

* ``y`` -- the generated text (boxes + ``<mask_token>`` placeholders).  Its action
  space is the LM vocabulary and it is scored by ms-swift's stock GRPO loss.
* ``m`` -- the binary mask decoded by :class:`BoxGuidedMaskDecoder` from the
  ``<mask_token>`` hidden states of the *current* policy.  Its action space is one
  Bernoulli variable per decoder pixel; randomness comes from sampling ``m``.

``log pi_phi(m | x, y)`` is differentiable w.r.t. both the decoder parameters and
the Qwen3-VL hidden states that feed it, so a single GRPO objective updates the
whole joint policy instead of only the language branch.  The reward itself
(box IoU / mask IoU) stays non-differentiable and is always detached.

Mask resolution follows SFT exactly: the GT mask is bilinearly resized to the
decoder output resolution and binarised with ``> 127.5``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qwen3vl_seg.data.dataset import normalize_decoder_image
from qwen3vl_seg.eval.metrics import match_masks
from qwen3vl_seg.grpo.on_policy_reward import (
    _box_reward,
    _completion_text,
    _extract_decoder_state_dict,
    _parse_solution,
)
from qwen3vl_seg.model.mask_decoder import BoxGuidedMaskDecoder, MaskDecoderConfig
from qwen3vl_seg.model.prompt_format import MASK_TOKEN
from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks
from qwen3vl_seg.model.vision_hooks import ModelFeatureHooks, extract_per_image_features

LOGPROB_REDUCTIONS = ("mean", "sum")
REWARD_MODES = ("deterministic", "sampled")


def resolve_mask_decoder_dtype(name: str | None) -> torch.dtype:
    """Map an env override to a dtype (defaults to fp32 for trainable decoding)."""
    value = (name or "float32").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if value not in mapping:
        raise ValueError(f"unsupported mask policy dtype: {name!r}")
    return mapping[value]


def load_mask_decoder(
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    trainable: bool = True,
    init_state: str | None = None,
) -> BoxGuidedMaskDecoder:
    """Load the mask decoder weights, optionally from a joint-training snapshot."""
    decoder_state: dict[str, torch.Tensor] | None = None
    if init_state:
        state_path = Path(init_state)
        if not state_path.exists():
            raise FileNotFoundError(f"joint decoder init state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "decoder" in state:
            state = state["decoder"]
        if not isinstance(state, dict):
            raise RuntimeError(f"unexpected joint decoder state format in {state_path}")
        decoder_state = dict(state)

    if decoder_state is None:
        ckpt = Path(checkpoint)
        ds_dir = ckpt / "ds"
        if not ds_dir.exists():
            ds_dir = ckpt
        latest = (ds_dir / "latest").read_text(encoding="utf-8").strip()
        model_state = ds_dir / latest / "mp_rank_00_model_states.pt"
        if not model_state.exists():
            raise FileNotFoundError(f"stage2 model state not found: {model_state}")
        state = torch.load(model_state, map_location="cpu", weights_only=False)
        decoder_state = _extract_decoder_state_dict(state)

    decoder = BoxGuidedMaskDecoder(MaskDecoderConfig())
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "decoder checkpoint mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    decoder.to(device=device, dtype=dtype)
    decoder.train(trainable)
    decoder.requires_grad_(trainable)
    return decoder


def load_gt_masks(gt: dict[str, Any], stride_h: int, stride_w: int) -> list[np.ndarray]:
    """Binarised GT masks at the decoder output resolution (SFT-identical)."""
    masks: list[np.ndarray] = []
    for mask_path in gt.get("mask_paths", []):
        with Image.open(mask_path) as handle:
            resized = handle.convert("L").resize((stride_w, stride_h), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        masks.append((array > 127.5).astype(np.uint8))
    return masks


def mask_logprob_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Bernoulli log-probability of ``target`` under ``sigmoid(logits)``.

    ``logits`` / ``target`` have shape ``(N, H, W)``.  The returned scalar is the
    differentiable ``log pi_phi(m | x, y)`` term, averaged over instances.
    """
    if reduction not in LOGPROB_REDUCTIONS:
        raise ValueError(f"unknown mask log-prob reduction: {reduction!r}")
    per_pixel = -F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    flat = per_pixel.flatten(1)
    per_instance = flat.sum(dim=1) if reduction == "sum" else flat.mean(dim=1)
    return per_instance.mean()


@dataclass
class MaskActionRecord:
    """One sampled mask action plus everything needed to re-score it."""

    row: int
    image_index: int
    boxes: torch.Tensor | None = None
    mask_positions: torch.Tensor | None = None
    image: torch.Tensor | None = None
    sampled_mask: torch.Tensor | None = None
    gt_masks: list[np.ndarray] = field(default_factory=list)
    old_mask_logp: float = 0.0
    mask_iou: float = 0.0
    deterministic_mask_iou: float = 0.0
    box_iou: float = 0.0
    advantage: float = 0.0
    error: str | None = None
    failure: str | None = None

    @property
    def valid(self) -> bool:
        return self.failure is None and self.sampled_mask is not None

    @property
    def num_instances(self) -> int:
        return 0 if self.sampled_mask is None else int(self.sampled_mask.shape[0])


def _empty_stat() -> dict[str, float]:
    return {
        "box_iou": 0.0,
        "mask_iou": 0.0,
        "sampled_mask_iou": 0.0,
        "deterministic_mask_iou": 0.0,
        "parse_fail": 0.0,
        "missing_mask_token": 0.0,
        "mask_count_mismatch": 0.0,
        "decode_fail": 0.0,
    }


def capture_mask_actions_from_policy_forward(
    *,
    decoder: BoxGuidedMaskDecoder,
    tokenizer: Any,
    hooks: ModelFeatureHooks,
    model_inputs: dict[str, Any],
    samples: Sequence[Any],
    device: torch.device,
    sample_masks: bool = True,
    reward_mode: str = "deterministic",
    logprob_reduction: str = "mean",
) -> tuple[list[MaskActionRecord], list[dict[str, float]]]:
    """Sample one mask action per sample from the live policy's hidden states.

    Must be called inside the old-policy forward (``torch.no_grad()``): the logits
    here define the behaviour policy ``mu`` used as the denominator of the GRPO
    importance ratio.
    """
    hidden_states = hooks.text_last_hidden
    if hidden_states is None:
        raise RuntimeError("joint mask policy: forward did not expose text_last_hidden")

    input_ids = model_inputs["input_ids"]
    mm_token_type_ids = model_inputs["mm_token_type_ids"]
    image_grid_thw = model_inputs.get("image_grid_thw")
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        raise RuntimeError("joint mask policy: forward did not contain image_grid_thw")

    per_image = extract_per_image_features(hooks, hidden_states, mm_token_type_ids, image_grid_thw)
    image_rows = [
        row
        for row in range(mm_token_type_ids.shape[0])
        if bool((mm_token_type_ids[row] == 1).any())
    ]
    if len(image_rows) != len(samples):
        raise RuntimeError(
            f"joint mask policy: expected one image per sample, got {len(image_rows)} rows "
            f"for {len(samples)} samples"
        )

    if reward_mode not in REWARD_MODES:
        raise ValueError(f"unknown mask reward mode: {reward_mode!r}")

    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    if mask_token_id is None:
        raise RuntimeError(f"tokenizer does not define {MASK_TOKEN}")

    decode_dtype = next(decoder.parameters()).dtype
    image_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    gt_cache: dict[tuple[tuple[str, ...], int, int], list[np.ndarray]] = {}
    records: list[MaskActionRecord] = []
    stats: list[dict[str, float]] = []

    for image_index, row in enumerate(image_rows):
        sample = samples[row]
        completion = _completion_text(sample)
        gt = _parse_solution((getattr(sample, "extra", None) or {}).get("solution"))
        parsed = parse_generated_boxes_and_masks(completion)
        record = MaskActionRecord(row=row, image_index=image_index)
        stat = _empty_stat()
        record.box_iou = float(_box_reward(parsed, gt))
        stat["box_iou"] = record.box_iou

        mask_positions = (input_ids[row] == mask_token_id).nonzero(as_tuple=False).flatten()
        if not parsed:
            record.failure = "parse_fail"
            stat["parse_fail"] = 1.0
        elif mask_positions.numel() == 0:
            record.failure = "missing_mask_token"
            stat["missing_mask_token"] = 1.0
        elif mask_positions.numel() != len(parsed):
            record.failure = "mask_count_mismatch"
            stat["mask_count_mismatch"] = 1.0

        if record.failure is None:
            try:
                grid = image_grid_thw[row]
                grid_h, grid_w = int(grid[1]), int(grid[2])
                stride_h, stride_w = grid_h * 2, grid_w * 2

                gt_paths = tuple(str(path) for path in gt.get("mask_paths", []))
                gt_key = (gt_paths, stride_h, stride_w)
                gt_masks = gt_cache.get(gt_key)
                if gt_masks is None:
                    gt_masks = load_gt_masks(gt, stride_h, stride_w)
                    gt_cache[gt_key] = gt_masks
                if not gt_masks:
                    raise ValueError("sample has no ground-truth mask")

                image_path = str(gt.get("image_path", ""))
                image_key = (image_path, grid_h, grid_w)
                image_tensor = image_cache.get(image_key)
                if image_tensor is None:
                    with Image.open(image_path) as handle:
                        image_tensor = normalize_decoder_image(
                            handle.convert("RGB"), grid_h * 16, grid_w * 16
                        ).unsqueeze(0)
                    image_cache[image_key] = image_tensor

                boxes = torch.tensor(
                    [[float(v) / 1000.0 for v in item["bbox_2d"]] for item in parsed],
                    dtype=torch.float32,
                    device=device,
                )
                feature = per_image[image_index]
                seg_features = hidden_states[row, mask_positions, :].unsqueeze(0)
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    decoded = decoder(
                        visual_features=[item.float() for item in feature["visual"]],
                        mm_features=feature["mm"].to(decode_dtype),
                        seg_features=seg_features.to(decode_dtype),
                        boxes=boxes.unsqueeze(0).to(decode_dtype),
                        image=image_tensor.to(device=device, dtype=decode_dtype),
                    )
                logits = decoded["mask_logits"][0][: len(parsed)].float()
                probabilities = torch.sigmoid(logits)
                if sample_masks:
                    sampled = torch.bernoulli(probabilities)
                else:
                    sampled = (probabilities > 0.5).float()
                record.old_mask_logp = float(
                    mask_logprob_from_logits(logits, sampled, logprob_reduction).item()
                )
                record.sampled_mask = sampled.detach().to("cpu")
                record.mask_positions = mask_positions.detach().to("cpu")
                record.boxes = boxes.detach().to("cpu")
                record.image = image_tensor.detach().squeeze(0).to("cpu")
                record.gt_masks = gt_masks

                sampled_np = record.sampled_mask.numpy() > 0.5
                pred_list = [sampled_np[index] for index in range(sampled_np.shape[0])]
                _, _, total = match_masks(pred_list, gt_masks)
                record.mask_iou = float(total / max(len(gt_masks), 1))

                det_np = (probabilities > 0.5).detach().to("cpu").numpy()
                det_list = [det_np[index] for index in range(det_np.shape[0])]
                _, _, det_total = match_masks(det_list, gt_masks)
                record.deterministic_mask_iou = float(det_total / max(len(gt_masks), 1))
            except Exception as exc:  # noqa: BLE001
                record.failure = "decode_fail"
                record.error = str(exc)
                stat["decode_fail"] = 1.0

        stat["sampled_mask_iou"] = record.mask_iou
        # Token-branch reward keeps the eval/SFT-identical deterministic mask by default
        # so the text advantage is not polluted by mask-sampling noise; the mask
        # action advantage below always uses the sampled mask.
        stat["mask_iou"] = (
            record.mask_iou if reward_mode == "sampled" else record.deterministic_mask_iou
        )
        stat["deterministic_mask_iou"] = record.deterministic_mask_iou
        records.append(record)
        stats.append(stat)

    return records, stats


def joint_mask_policy_loss(
    *,
    decoder: BoxGuidedMaskDecoder,
    hooks: ModelFeatureHooks,
    model_inputs: dict[str, Any],
    records: Sequence[MaskActionRecord],
    device: torch.device,
    logprob_reduction: str = "mean",
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """GRPO objective for the mask action, differentiable w.r.t. the policy.

    ``L_mask = -E[ min(r * A, clip(r, 1-e, 1+e) * A) ]`` with
    ``r = exp(log pi_phi(m|.) - log mu(m|.))`` and ``A`` the group-normalised mask
    IoU advantage of the sampled mask.
    """
    usable = [record for record in records if record.valid]
    metrics: dict[str, float] = {
        "joint_masks/valid_ratio": (len(usable) / len(records)) if records else 0.0,
        "joint_masks/sampled_iou": (
            sum(record.mask_iou for record in usable) / len(usable) if usable else 0.0
        ),
        "joint_masks/deterministic_iou": (
            sum(record.deterministic_mask_iou for record in usable) / len(usable) if usable else 0.0
        ),
        "joint_masks/instances": (
            sum(record.num_instances for record in usable) / len(usable) if usable else 0.0
        ),
    }
    if not usable:
        return None, metrics

    hidden_states = hooks.text_last_hidden
    if hidden_states is None:
        raise RuntimeError("joint mask policy loss: forward did not expose text_last_hidden")

    image_grid_thw = model_inputs.get("image_grid_thw")
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        raise RuntimeError("joint mask policy loss: forward did not contain image_grid_thw")
    per_image = extract_per_image_features(
        hooks, hidden_states, model_inputs["mm_token_type_ids"], image_grid_thw
    )

    decode_dtype = next(decoder.parameters()).dtype
    objectives: list[torch.Tensor] = []
    ratios: list[float] = []
    logps: list[float] = []
    advantages: list[float] = []
    entropies: list[float] = []
    for record in usable:
        feature = per_image[record.image_index]
        positions = record.mask_positions.to(device)
        seg_features = hidden_states[record.row, positions, :].unsqueeze(0)
        target = record.sampled_mask.to(device=device, dtype=torch.float32)
        boxes = record.boxes.to(device=device, dtype=decode_dtype)
        if boxes.ndim == 2:  # (N, 4) -> (1, N, 4): the decoder expects a batch dim
            boxes = boxes.unsqueeze(0)
        with torch.autocast(device_type=device.type, enabled=False):
            decoded = decoder(
                visual_features=[item.float() for item in feature["visual"]],
                mm_features=feature["mm"].to(decode_dtype),
                seg_features=seg_features.to(decode_dtype),
                boxes=boxes,
                image=record.image.to(device=device, dtype=decode_dtype).unsqueeze(0),
            )
            logits = decoded["mask_logits"][0][: target.shape[0]].float()
            new_logp = mask_logprob_from_logits(logits, target, logprob_reduction)

        advantage = float(record.advantage)
        ratio = torch.exp((new_logp - float(record.old_mask_logp)).clamp(min=-6.0, max=6.0))
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = -torch.minimum(ratio * advantage, clipped * advantage)
        objectives.append(objective)

        with torch.no_grad():
            ratio_value = ratio.detach()
            ratios.append(float(ratio_value.item()))
            logps.append(float(new_logp.detach().item()))
            advantages.append(advantage)
            probability = torch.sigmoid(logits)
            entropy = -(
                probability * torch.log(probability.clamp_min(1e-6))
                + (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-6))
            )
            entropies.append(float(entropy.mean().item()))

    loss = torch.stack([item.float() for item in objectives]).mean()
    metrics.update(
        {
            "joint_masks/aux_loss": float(loss.detach().item()),
            "joint_masks/logp_mean": sum(logps) / len(logps),
            "joint_masks/ratio_mean": sum(ratios) / len(ratios),
            "joint_masks/adv_mean": sum(advantages) / len(advantages),
            "joint_masks/adv_abs_mean": sum(abs(item) for item in advantages) / len(advantages),
            "joint_masks/entropy": sum(entropies) / len(entropies),
            "joint_masks/grad_to_llm": 1.0 if hidden_states.requires_grad else 0.0,
        }
    )
    return loss, metrics



