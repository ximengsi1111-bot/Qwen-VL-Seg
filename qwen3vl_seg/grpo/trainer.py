"""GRPO trainer whose mask reward/actions come from the live policy.

Two modes share this implementation:

* ``GRPO_ON_POLICY_REWARD=1`` (v1, frozen decoder) -- the mask decoder is frozen
  and only turns the live hidden states into a non-differentiable IoU reward.
* ``GRPO_JOINT_MASK_POLICY=1`` (v2, joint policy) -- the decoder is trainable and
  the mask is treated as a second action of the policy::

      pi_theta(y, m | x) = pi_theta(y | x) * pi_phi(m | x, y)
      L_total = L_token + beta * L_mask

  ``L_mask`` is a GRPO/PPO-style objective on the sampled mask, so its gradient
  flows into the decoder parameters *and* into the Qwen3-VL hidden states that
  feed the decoder.  The IoU reward stays detached.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from accelerate.utils import gather

from swift.rlhf_trainers import GRPOTrainer

from qwen3vl_seg.grpo.joint_mask_policy import (
    MaskActionRecord,
    capture_mask_actions_from_policy_forward,
    joint_mask_policy_loss,
    load_mask_decoder,
    resolve_mask_decoder_dtype,
)
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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


class OnPolicySegGRPOTrainer(GRPOTrainer):
    """Compute/optimise mask actions from the same forward as the token policy."""

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

        self._joint_enabled = _env_flag("GRPO_JOINT_MASK_POLICY", False)
        self._joint_beta = _env_float("GRPO_MASK_POLICY_BETA", 0.01)
        self._joint_lr = _env_float("GRPO_MASK_POLICY_LR", 1e-5)
        self._joint_clip = _env_float("GRPO_MASK_POLICY_CLIP", 0.2)
        self._joint_logprob = os.environ.get("GRPO_MASK_POLICY_LOGPROB", "mean").strip().lower()
        self._joint_dtype = resolve_mask_decoder_dtype(os.environ.get("GRPO_MASK_POLICY_DTYPE"))
        self._joint_sample_masks = _env_flag("GRPO_MASK_POLICY_SAMPLE", True)
        self._joint_reward_mode = os.environ.get("GRPO_MASK_POLICY_REWARD", "deterministic").strip().lower()
        self._joint_pending: list[tuple[int, dict[str, Any], list[MaskActionRecord]]] = []
        self._joint_batches: dict[int, list[MaskActionRecord]] = {}
        self._joint_optimizer_attached = False
        self._joint_warned_no_grad = False

        if self._joint_enabled and not self._on_policy_enabled:
            raise ValueError("GRPO_JOINT_MASK_POLICY requires GRPO_ON_POLICY_REWARD=1")
        if self._joint_reward_mode not in {"deterministic", "sampled"}:
            raise ValueError(f"unsupported GRPO_MASK_POLICY_REWARD={self._joint_reward_mode!r}")
        if self._joint_logprob not in {"mean", "sum"}:
            raise ValueError(f"unsupported GRPO_MASK_POLICY_LOGPROB={self._joint_logprob!r}")

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
            if self._joint_enabled:
                self._on_policy_decoder_dtype = self._joint_dtype
            print(
                "[OnPolicySegGRPOTrainer] enabled "
                f"mode={self._on_policy_mode} box_weight={self._on_policy_box_weight} "
                f"mask_weight={self._on_policy_mask_weight} joint_mask_policy={self._joint_enabled} "
                f"beta={self._joint_beta} mask_lr={self._joint_lr} "
                f"mask_dtype={self._joint_dtype} sample_masks={self._joint_sample_masks} "
                f"logprob_reduction={self._joint_logprob} reward={self._joint_reward_mode}",
                flush=True,
            )

    # ------------------------------------------------------------------ decoder
    def _get_decoder(self):
        if self._on_policy_decoder is None:
            checkpoint = os.environ.get("GRPO_STAGE2")
            if not checkpoint:
                raise ValueError("GRPO_STAGE2 must be set for on-policy reward")
            if self._joint_enabled:
                init_state = os.environ.get("GRPO_JOINT_DECODER_INIT", "").strip() or None
                self._on_policy_decoder = load_mask_decoder(
                    checkpoint,
                    device=self.accelerator.device,
                    dtype=self._joint_dtype,
                    trainable=True,
                    init_state=init_state,
                )
            else:
                self._on_policy_decoder = load_frozen_mask_decoder(
                    checkpoint,
                    device=self.accelerator.device,
                    dtype=self._on_policy_decoder_dtype,
                )
        return self._on_policy_decoder

    # ------------------------------------------------------------------- rollout
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
        self._joint_pending = []
        self._joint_batches = {}
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
        if self._joint_enabled:
            self._metrics[mode]["onpolicy/sampled_mask_iou"].append(
                sum(item.get("sampled_mask_iou", 0.0) for item in stats) / len(stats)
            )
            self._metrics[mode]["onpolicy/deterministic_mask_iou"].append(
                sum(item.get("deterministic_mask_iou", 0.0) for item in stats) / len(stats)
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

        if self._joint_enabled:
            self._register_joint_batches(samples)
        return batch_encoded_inputs

    def _register_joint_batches(self, samples) -> None:
        """Attach group-normalised mask advantages and index records per micro-batch."""
        flat: list[tuple[int, dict[str, Any], MaskActionRecord]] = [
            (key, model_inputs, record)
            for key, model_inputs, records in self._joint_pending
            for record in records
        ]
        if len(flat) != len(samples):
            raise RuntimeError(
                "joint mask policy captured "
                f"{len(flat)} actions for {len(samples)} samples"
            )
        if not flat:
            return

        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        values = torch.tensor(
            [record.mask_iou if record.valid else 0.0 for _, _, record in flat],
            dtype=torch.float32,
            device=device,
        )
        advantages = self._mask_group_advantage(values)
        for (_, _, record), advantage in zip(flat, advantages.tolist()):
            record.advantage = float(advantage) if record.valid else 0.0

        valid = [record for _, _, record in flat if record.valid]
        self._metrics[mode]["joint_masks/advantage_abs_mean"].append(
            sum(abs(record.advantage) for record in valid) / len(valid) if valid else 0.0
        )
        for key, model_inputs, records in self._joint_pending:
            self._joint_batches[key] = records
            assert model_inputs is not None
        if self._on_policy_debug and self.accelerator.is_main_process:
            print(
                f"[joint-mask] step={self.state.global_step} "
                f"actions={len(flat)} valid={len(valid)} "
                f"adv_abs={sum(abs(record.advantage) for record in valid) / max(len(valid), 1):.4f}",
                flush=True,
            )

    def _mask_group_advantage(self, values: torch.Tensor) -> torch.Tensor:
        """Group-normalised mask IoU advantage, matching ms-swift's group scaling."""
        num_generations = int(self.num_generations if self.model.training else self.num_generations_eval)
        local_count = int(values.shape[0])
        gathered = self.accelerator.gather(values)
        world = self.accelerator.num_processes
        if gathered.numel() != local_count * world:
            # Uneven local batches: fall back to a plain local standardisation.
            return (values - values.mean()) / values.std().clamp_min(1e-6)
        if num_generations > 1 and gathered.numel() % num_generations == 0:
            groups = gathered.view(-1, num_generations)
            mean = groups.mean(dim=1, keepdim=True)
            std = groups.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            advantage = ((groups - mean) / std).reshape(-1)
        else:
            advantage = (gathered - gathered.mean()) / gathered.std().clamp_min(1e-6)
        start = self.accelerator.process_index * local_count
        return advantage[start : start + local_count]

    # -------------------------------------------------------------- policy logps
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
        joint_records: list[MaskActionRecord] = []
        with ModelFeatureHooks(policy_base) as hooks:
            output = super()._get_per_token_logps_and_entropies(
                model,
                model_inputs,
                grpo_batch,
                compute_entropy=compute_entropy,
                origin_data=origin_data,
            )
            if self._joint_enabled:
                joint_records, stats = capture_mask_actions_from_policy_forward(
                    decoder=self._get_decoder(),
                    tokenizer=self.tokenizer,
                    hooks=hooks,
                    model_inputs=model_inputs,
                    samples=chunk,
                    device=self.accelerator.device,
                    sample_masks=self._joint_sample_masks,
                    reward_mode=self._joint_reward_mode,
                    logprob_reduction=self._joint_logprob,
                )
            else:
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
        if self._joint_enabled:
            self._joint_pending.append((input_key, model_inputs, joint_records))
        return output

    # --------------------------------------------------------------------- loss
    def _compute_loss_and_metrics(self, model, model_inputs, grpo_batch):
        if not (self._joint_enabled and self.model.training):
            return super()._compute_loss_and_metrics(model, model_inputs, grpo_batch)

        records = self._joint_batches.get(id(model_inputs))
        if records is None:
            records = self._joint_batches.get(id(grpo_batch))
        policy_base = resolve_policy_base(self.accelerator.unwrap_model(self.model))
        with ModelFeatureHooks(policy_base) as hooks:
            loss, metrics_data = super()._compute_loss_and_metrics(model, model_inputs, grpo_batch)
            if records is not None:
                aux_loss, aux_metrics = joint_mask_policy_loss(
                    decoder=self._get_decoder(),
                    hooks=hooks,
                    model_inputs=model_inputs,
                    records=records,
                    device=self.accelerator.device,
                    logprob_reduction=self._joint_logprob,
                    clip_epsilon=self._joint_clip,
                )
            else:
                aux_loss, aux_metrics = None, {}

        if records is None:
            if not self._joint_warned_no_grad and self.accelerator.is_main_process:
                self._joint_warned_no_grad = True
                print(
                    "[joint-mask] no sampled mask action found for this micro-batch; "
                    "skipping the mask policy term",
                    flush=True,
                )
        elif aux_loss is not None:
            loss = loss + self._joint_beta * aux_loss
            aux_metrics["joint_masks/beta"] = self._joint_beta
            aux_metrics["joint_masks/weighted_aux_loss"] = float(
                (self._joint_beta * aux_loss).detach().item()
            )
        self._log_joint_metrics(aux_metrics)
        return loss, metrics_data

    def _log_joint_metrics(self, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        mode = "train" if self.model.training else "eval"
        for key, value in metrics.items():
            self._metrics[mode][key].append(float(value))

    # ---------------------------------------------------------------- optimizer
    def create_optimizer(self, model=None):
        """Attach the mask decoder to the actor optimizer.

        ``transformers>=5.8`` builds the optimizer through ``create_optimizer``
        (``create_optimizer_and_scheduler`` is no longer called), so this is the
        only reliable hook for extra parameter groups.
        """
        optimizer = super().create_optimizer(model=model)
        self._attach_joint_decoder_optimizer(optimizer)
        return optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self._attach_joint_decoder_optimizer()

    def _attach_joint_decoder_optimizer(self, optimizer=None) -> None:
        """Add the (already instantiated) mask decoder to the actor optimizer."""
        if not self._joint_enabled:
            return
        if optimizer is None:
            optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        # ``accelerator.prepare`` may wrap the optimizer; add the group to the
        # wrapped torch optimizer so that ``step()`` actually sees it.
        inner = getattr(optimizer, "optimizer", optimizer)
        existing = {id(param) for group in inner.param_groups for param in group["params"]}
        parameters = [
            param
            for param in self._get_decoder().parameters()
            if param.requires_grad and id(param) not in existing
        ]
        if not parameters:
            return
        inner.add_param_group(
            {"params": parameters, "lr": self._joint_lr, "weight_decay": 0.0}
        )
        self._joint_optimizer_attached = True
        if self.accelerator.is_main_process:
            print(
                "[joint-mask] added mask decoder to the optimizer: "
                f"params={sum(param.numel() for param in parameters)} lr={self._joint_lr} "
                f"param_groups={len(inner.param_groups)}",
                flush=True,
            )

    # ------------------------------------------------------------------ backward
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._joint_enabled:
            self._sync_joint_decoder_grads()
        return loss

    def _sync_joint_decoder_grads(self) -> None:
        """Average mask decoder gradients across ranks (decoder sits outside DDP)."""
        decoder = self._on_policy_decoder
        if decoder is None:
            return
        parameters = [param for param in decoder.parameters() if param.requires_grad]
        if not parameters:
            return
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        grad_norm = 0.0
        for param in parameters:
            if param.grad is None:
                # Keep every rank on the same collective schedule even if this rank
                # produced no mask action for the micro-batch.
                if world > 1:
                    param.grad = torch.zeros_like(param)
                continue
            if world > 1:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad /= world
            grad_norm += float(param.grad.detach().pow(2).sum().item())
        if world > 1:
            for param in parameters:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
        self._metrics["train"]["joint_masks/decoder_grad_norm"].append(grad_norm**0.5)

    # ---------------------------------------------------------------- checkpoint
    def _save_checkpoint(self, model, trial, metrics=None):
        output = super()._save_checkpoint(model, trial)
        if self._joint_enabled and self.accelerator.is_main_process:
            self._save_joint_decoder_snapshot()
        return output

    def _save_joint_decoder_snapshot(self) -> None:
        decoder = self._on_policy_decoder
        if decoder is None:
            return
        directory = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, "joint_mask_decoder.pt")
            state = {key: value.detach().to("cpu") for key, value in decoder.state_dict().items()}
            torch.save({"decoder": state, "step": int(self.state.global_step)}, path)
            if self.accelerator.is_main_process:
                print(f"[joint-mask] saved mask decoder -> {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[joint-mask] failed to save mask decoder snapshot: {exc}", flush=True)