"""GRPO trainer whose mask reward sees the live policy hidden states."""
from __future__ import annotations

import os
from typing import Any

import torch
from accelerate.utils import gather

from swift.rlhf_trainers import GRPOTrainer

from qwen3vl_seg.grpo.on_policy_reward import (
    load_frozen_mask_decoder,
    resolve_policy_base,
    score_masks_from_policy_forward,
)
from qwen3vl_seg.model.vision_hooks import ModelFeatureHooks


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class OnPolicySegGRPOTrainer(GRPOTrainer):
    """Compute mask reward from the same forward pass as old log-probabilities.

    This keeps the mask decoder frozen while ensuring its ``seg_features`` come
    from the current policy rather than a separate fixed base model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_policy_enabled = _env_flag("GRPO_ON_POLICY_REWARD", False)
        self._on_policy_mode = os.environ.get("GRPO_ON_POLICY_MODE", "reuse").strip().lower()
        self._on_policy_debug = _env_flag("GRPO_ON_POLICY_DEBUG", False)
        self._on_policy_box_weight = float(os.environ.get("GRPO_ON_POLICY_BOX_WEIGHT", "0.3"))
        self._on_policy_mask_weight = float(os.environ.get("GRPO_ON_POLICY_MASK_WEIGHT", "0.7"))
        self._on_policy_decoder = None
        self._on_policy_decoder_dtype = torch.bfloat16
        self._on_policy_pending_chunks: list[list[Any]] | None = None
        self._on_policy_seen_inputs: set[int] = set()
        self._on_policy_stats: dict[int, dict[str, float]] = {}
        self._on_policy_capture_active = False

        if self._on_policy_enabled:
            if self._on_policy_mode != "reuse":
                raise ValueError(
                    "only GRPO_ON_POLICY_MODE=reuse is implemented; "
                    f"got {self._on_policy_mode!r}"
                )
            if getattr(self, "dynamic_sample", False) or getattr(self, "dynamic_num_samples", False):
                raise ValueError(
                    "on-policy reward v1 requires dynamic_sample=false and dynamic_num_samples=false"
                )
            padding_free = bool(getattr(self.template, "padding_free", False))
            sequence_parallel_size = int(getattr(self.template, "sequence_parallel_size", 1))
            if padding_free or sequence_parallel_size != 1:
                raise ValueError(
                    "on-policy reward v1 requires padding_free=false and sequence_parallel_size=1"
                )
            if self.template.is_encoder_decoder:
                raise ValueError("on-policy segmentation reward does not support encoder-decoder templates")
            print(
                "[OnPolicySegGRPOTrainer] enabled "
                f"mode={self._on_policy_mode} box_weight={self._on_policy_box_weight} "
                f"mask_weight={self._on_policy_mask_weight}",
                flush=True,
            )

    def _get_decoder(self):
        if self._on_policy_decoder is None:
            checkpoint = os.environ.get("GRPO_STAGE2")
            if not checkpoint:
                raise ValueError("GRPO_STAGE2 must be set for on-policy reward")
            self._on_policy_decoder = load_frozen_mask_decoder(
                checkpoint,
                device=self.accelerator.device,
                dtype=self._on_policy_decoder_dtype,
            )
        return self._on_policy_decoder

    def _score_completions(self, samples):
        if not self._on_policy_enabled:
            return super()._score_completions(samples)
        # Reward computation is deferred until the current-policy forward exists.
        return samples

    def _prepare_batch_inputs(self, samples):
        if not self._on_policy_enabled:
            return super()._prepare_batch_inputs(samples)

        chunks = list(self.split_by_mini_batches(samples))
        self._on_policy_pending_chunks = list(chunks)
        self._on_policy_seen_inputs = set()
        self._on_policy_stats = {}
        self._on_policy_capture_active = True
        try:
            batch_encoded_inputs = super()._prepare_batch_inputs(samples)
        finally:
            self._on_policy_capture_active = False
            self._on_policy_pending_chunks = None

        missing = [index for index, sample in enumerate(samples) if id(sample) not in self._on_policy_stats]
        if missing:
            raise RuntimeError(
                "on-policy reward did not capture every sample; "
                f"missing={len(missing)} first_index={missing[0]}"
            )

        device = self.accelerator.device
        rewards = torch.tensor(
            [
                self._on_policy_box_weight * self._on_policy_stats[id(sample)]["box_iou"]
                + self._on_policy_mask_weight * self._on_policy_stats[id(sample)]["mask_iou"]
                for sample in samples
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        self._rewards_per_func = gather(rewards).detach()

        mode = "train" if self.model.training else "eval"
        stats = list(self._on_policy_stats.values())
        self._metrics[mode]["onpolicy/box_iou"].append(
            sum(item["box_iou"] for item in stats) / len(stats)
        )
        self._metrics[mode]["onpolicy/mask_iou"].append(
            sum(item["mask_iou"] for item in stats) / len(stats)
        )
        self._metrics[mode]["onpolicy/reward"].append(float(rewards.mean().item()))
        self._metrics[mode]["onpolicy/missing_mask_token_ratio"].append(
            sum(item["missing_mask_token"] for item in stats) / len(stats)
        )
        self._metrics[mode]["onpolicy/mask_count_mismatch_ratio"].append(
            sum(item["mask_count_mismatch"] for item in stats) / len(stats)
        )
        self._metrics[mode]["onpolicy/parse_fail_ratio"].append(
            sum(item["parse_fail"] for item in stats) / len(stats)
        )
        self._metrics[mode]["onpolicy/decode_fail_ratio"].append(
            sum(item["decode_fail"] for item in stats) / len(stats)
        )

        if self._on_policy_debug and self.accelerator.is_main_process:
            print(
                "[OnPolicySegGRPOTrainer] "
                f"step={self.state.global_step} "
                f"reward={rewards.mean().item():.6f} "
                f"box={sum(item['box_iou'] for item in stats) / len(stats):.6f} "
                f"mask={sum(item['mask_iou'] for item in stats) / len(stats):.6f} "
                f"missing_mask={sum(item['missing_mask_token'] for item in stats):.0f} "
                f"mismatch={sum(item['mask_count_mismatch'] for item in stats):.0f} "
                f"decode_fail={sum(item['decode_fail'] for item in stats):.0f}",
                flush=True,
            )
        return batch_encoded_inputs

    def _get_per_token_logps_and_entropies(
        self,
        model,
        model_inputs,
        grpo_batch,
        compute_entropy=False,
        origin_data=None,
    ):
        if not self._on_policy_capture_active or model is not self.model:
            return super()._get_per_token_logps_and_entropies(
                model,
                model_inputs,
                grpo_batch,
                compute_entropy=compute_entropy,
                origin_data=origin_data,
            )

        input_key = id(model_inputs)
        if input_key in self._on_policy_seen_inputs:
            return super()._get_per_token_logps_and_entropies(
                model,
                model_inputs,
                grpo_batch,
                compute_entropy=compute_entropy,
                origin_data=origin_data,
            )
        self._on_policy_seen_inputs.add(input_key)

        if not self._on_policy_pending_chunks:
            raise RuntimeError("on-policy capture received more batches than expected")
        chunk = self._on_policy_pending_chunks.pop(0)
        if len(chunk) != int(model_inputs["input_ids"].shape[0]):
            raise RuntimeError(
                "on-policy sample/forward row mismatch: "
                f"samples={len(chunk)} rows={int(model_inputs['input_ids'].shape[0])}"
            )

        policy_base = resolve_policy_base(self.accelerator.unwrap_model(self.model))
        with ModelFeatureHooks(policy_base) as hooks:
            output = super()._get_per_token_logps_and_entropies(
                model,
                model_inputs,
                grpo_batch,
                compute_entropy=compute_entropy,
                origin_data=origin_data,
            )
            stats = score_masks_from_policy_forward(
                decoder=self._get_decoder(),
                tokenizer=self.tokenizer,
                hooks=hooks,
                model_inputs=model_inputs,
                samples=chunk,
                device=self.accelerator.device,
                decoder_dtype=self._on_policy_decoder_dtype,
            )
        if len(stats) != len(chunk):
            raise RuntimeError(
                f"on-policy reward returned {len(stats)} rows for {len(chunk)} samples"
            )
        for sample, item in zip(chunk, stats):
            self._on_policy_stats[id(sample)] = item
        return output
