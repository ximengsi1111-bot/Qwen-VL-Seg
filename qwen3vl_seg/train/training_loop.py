"""Distributed training loop for Qwen3-VL-Seg Stage1/Stage2.

The default launcher is ``torchrun`` because the server's ``segment`` conda
environment currently has a broken ``accelerate`` import. The loop is plain
PyTorch DDP with deterministic resumable sample ordering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from qwen3vl_seg.data.collator import SegmentationCollator
from qwen3vl_seg.data.bucket_sampler import BucketOrderSampler, BucketProcessSampler
from qwen3vl_seg.data.dataset import SegmentationDataset
from qwen3vl_seg.data.samplers import StepIndexSampler
from qwen3vl_seg.model.lora import apply_lora, merge_lora
from qwen3vl_seg.model.model_wrapper import Qwen3VLSegForSegmentation
from qwen3vl_seg.train.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--data-limit", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--deepspeed-config", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _init_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return rank, world_size, device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def _model_trainable_groups(
    wrapper: Qwen3VLSegForSegmentation,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    base_lr = float(config["learning_rate"])
    scale = float(config.get("vision_learning_rate_scale", 1.0))
    vision_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("base.model.visual."):
            vision_params.append(parameter)
        else:
            other_params.append(parameter)
    groups: list[dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr})
    if vision_params:
        groups.append({"params": vision_params, "lr": base_lr * scale})
    return groups


class _CosineWithWarmup:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_learning_rate: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = int(warmup_steps)
        self.min_learning_rate = float(min_learning_rate)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.min_lrs = [
            min(self.min_learning_rate, base_lr * 0.01) for base_lr in self.base_lrs
        ]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        step = self.step_count
        if step <= self.warmup_steps:
            factor = step / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base_lr, min_lr in zip(
            self.optimizer.param_groups, self.base_lrs, self.min_lrs
        ):
            group["lr"] = min_lr + (base_lr - min_lr) * factor

    def set_step(self, step: int) -> None:
        target = int(step)
        self.step_count = 0
        for _ in range(target):
            self.step()
        self.step_count = target

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state["step_count"])


def _configure_trainable(
    wrapper: Qwen3VLSegForSegmentation,
    config: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    method = str(config.get("method", "full"))
    if method == "lora":
        replaced = apply_lora(
            wrapper.base.model.language_model,
            list(config.get("lora_modules", [])),
            rank=int(config.get("lora_rank", 32)),
            alpha=int(config.get("lora_rank", 32)),
        )
        _rank0(rank, f"LoRA wrapped modules={replaced}")

    for name, parameter in wrapper.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in wrapper.named_parameters():
        if bool(config.get("train_decoder", True)) and name.startswith("decoder."):
            parameter.requires_grad_(True)
        if bool(config.get("train_vision_encoder", False)) and name.startswith(
            "base.model.visual."
        ):
            parameter.requires_grad_(True)
        if method == "lora" and name.startswith("base.model.language_model.") and ".lora_" in name:
            parameter.requires_grad_(True)
        if method == "full" and name.startswith("base.model.language_model."):
            parameter.requires_grad_(True)

    if method == "lora":
        _enable_special_token_training(wrapper)

    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def _enable_special_token_training(
    wrapper: Qwen3VLSegForSegmentation,
) -> None:
    """Make only the added seg-token rows of embedding/head trainable in Stage1."""
    targets: list[torch.nn.Parameter] = []
    for name, parameter in wrapper.named_parameters():
        if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
            targets.append(parameter)
    if not targets:
        raise RuntimeError("could not find embed_tokens.weight / lm_head.weight")
    device = targets[0].device
    ids = torch.as_tensor(list(wrapper.special_token_ids), device=device)
    row_mask = torch.zeros(
        targets[0].shape[0], dtype=torch.bool, device=device
    )
    row_mask[ids] = True

    def _masked_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad * row_mask.unsqueeze(-1)

    for parameter in targets:
        parameter.requires_grad_(True)
        parameter.register_hook(_masked_grad)


def _run_dir(data_root: str, checkpoint_dir: str) -> Path:
    root = Path(data_root)
    checkpoint = Path(checkpoint_dir)
    try:
        rel = checkpoint.relative_to(root)
    except ValueError:
        return root / "runs" / "small"
    if rel.parts and rel.parts[0] == "checkpoints":
        return root / "runs" / Path(*rel.parts[1:])
    return root / "runs" / "small"


def _save_rng(rank: int, directory: Path) -> None:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    torch.save(state, directory / f"rng-rank{rank}.pt")


def _load_rng(rank: int, directory: Path) -> None:
    state = torch.load(directory / f"rng-rank{rank}.pt")
    torch.random.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def _save_checkpoint(
    wrapper: Qwen3VLSegForSegmentation,
    optimizer: torch.optim.Optimizer,
    scheduler: _CosineWithWarmup,
    trainer_state: dict[str, Any],
    rank: int,
    world_size: int,
    checkpoint_dir: Path,
) -> None:
    _barrier(world_size)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainable_state = {
            name: parameter.detach().to("cpu")
            for name, parameter in wrapper.named_parameters()
            if parameter.requires_grad
        }
        torch.save(trainable_state, checkpoint_dir / "model_trainable.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2), encoding="utf-8"
        )
    _barrier(world_size)
    _save_rng(rank, checkpoint_dir)
    _barrier(world_size)


def _load_checkpoint(
    wrapper: Qwen3VLSegForSegmentation,
    optimizer: torch.optim.Optimizer,
    scheduler: _CosineWithWarmup,
    rank: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    trainable_state = torch.load(
        checkpoint_dir / "model_trainable.pt",
        map_location=device,
    )
    result = wrapper.load_state_dict(trainable_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "unexpected checkpoint keys: "
            + ", ".join(result.unexpected_keys[:8])
        )
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=device))
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location=device))
    scheduler.set_step(scheduler.step_count)
    _load_rng(rank, checkpoint_dir)
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return state


def _ds_save_checkpoint(
    engine: Any,
    scheduler: _CosineWithWarmup,
    trainer_state: dict[str, Any],
    rank: int,
    checkpoint_dir: Path,
) -> None:
    save_dir = checkpoint_dir / "ds"
    tag = f"step-{trainer_state['step']}"
    engine.save_checkpoint(
        save_dir=str(save_dir),
        tag=tag,
        client_state={"trainer_state": trainer_state, "scheduler": scheduler.state_dict()},
        save_latest=True,
        exclude_frozen_parameters=True,
    )
    for child in save_dir.glob("step-*"):
        if child.name != tag:
            shutil.rmtree(child, ignore_errors=True)
    _save_rng(rank, save_dir)
    if rank == 0:
        (save_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2), encoding="utf-8"
        )


def _ds_load_checkpoint(
    engine: Any,
    scheduler: _CosineWithWarmup,
    rank: int,
    load_dir: Path,
) -> dict[str, Any]:
    load_path, client_state = engine.load_checkpoint(
        str(load_dir),
        load_module_strict=False,
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
    )
    if load_path is None or client_state is None:
        raise RuntimeError(f"failed to load DeepSpeed checkpoint from {load_dir}")
    scheduler.load_state_dict(client_state["scheduler"])
    scheduler.set_step(scheduler.step_count)
    _load_rng(rank, load_dir)
    return client_state["trainer_state"]


def _warmstart_stage1(
    wrapper: Qwen3VLSegForSegmentation,
    config: dict[str, Any],
    rank: int,
) -> None:
    ckpt = Path(str(config.get("stage1_checkpoint", "")))
    if not ckpt.exists():
        raise FileNotFoundError(f"stage1_checkpoint not found: {ckpt}")
    ds_dir = ckpt / "ds"
    if not ds_dir.exists():
        ds_dir = ckpt
    latest_file = ds_dir / "latest"
    if not latest_file.exists():
        raise FileNotFoundError(f"latest checkpoint not found: {latest_file}")
    tag = latest_file.read_text(encoding="utf-8").strip()
    model_states = ds_dir / tag / "mp_rank_00_model_states.pt"
    if not model_states.exists():
        raise FileNotFoundError(f"Stage1 model states not found: {model_states}")

    lora_modules = [str(m) for m in config.get("stage1_lora_modules") or []]
    lora_rank = int(config.get("stage1_lora_rank", config.get("lora_rank", 32)))
    if lora_modules:
        apply_lora(
            wrapper.base.model.language_model,
            lora_modules,
            rank=lora_rank,
            alpha=lora_rank,
        )

    state = torch.load(model_states, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "module" in state and isinstance(
        state["module"], dict
    ):
        state = state["module"]
    result = wrapper.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            "unexpected Stage1 checkpoint keys: "
            + ", ".join(result.unexpected_keys[:8])
        )
    merge_lora(wrapper.base.model.language_model)
    del state
    _rank0(rank, f"Stage2 warm-start from Stage1 {model_states}, LoRA merged")


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = parameter.grad.float().norm()
            total += value.item() ** 2
    return math.sqrt(total)


def _log_metrics(
    metrics_path: Path,
    row: dict[str, Any],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["max_steps"] = int(config["smoke_steps"])
        config["max_pixels"] = int(config["smoke_max_pixels"])
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.max_pixels:
        config["max_pixels"] = args.max_pixels
    if args.num_workers >= 0:
        config["num_workers"] = args.num_workers
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.model_path:
        config["model_path"] = args.model_path
    if args.data_root:
        config["data_root"] = args.data_root
    if args.manifest:
        config["manifest"] = args.manifest
    if args.log_interval:
        config["log_interval"] = args.log_interval

    rank, world_size, device = _init_distributed()
    seed = int(config.get("seed", 42))
    _set_seed(seed + rank)
    _rank0(rank, f"rank={rank} world={world_size} device={device}")

    dtype = _dtype(str(config["dtype"]))
    _rank0(rank, f"loading {config['model_path']} ({config['dtype']})")
    wrapper = Qwen3VLSegForSegmentation.from_pretrained(
        str(config["model_path"]),
        torch_dtype=dtype,
    )
    if int(config.get("stage", 1)) == 2 and config.get("stage1_checkpoint"):
        _warmstart_stage1(wrapper, config, rank)
    stats = _configure_trainable(wrapper, config, rank)
    _rank0(rank, f"trainable_parameters={stats['trainable_parameters']}")
    wrapper.to(device, dtype=dtype)
    if bool(config.get("gradient_checkpointing", False)):
        wrapper.base.gradient_checkpointing_enable()

    param_groups = _model_trainable_groups(wrapper, config)
    if not param_groups:
        raise RuntimeError("no trainable parameters after applying training config")
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    total_steps = int(config["max_steps"])
    warmup_steps = int(config.get("warmup_steps", 0))

    max_pixels = int(config["max_pixels"])
    bucket_sizes_cfg = config.get("bucket_sizes")
    bucket_sizes = (
        tuple(int(v) for v in bucket_sizes_cfg) if bucket_sizes_cfg else None
    )
    processor = wrapper.processor
    dataset = SegmentationDataset(
        manifest_path=config["manifest"],
        data_root=config["data_root"],
        processor=processor,
        split="train",
        limit=args.data_limit or None,
        max_pixels=max_pixels,
        use_bucket=bool(config.get("use_bucket", False)),
        bucket_sizes=bucket_sizes,
    )
    accum = int(config["gradient_accumulation_steps"])
    use_deepspeed = bool(args.deepspeed_config)
    deepspeed_config: dict[str, Any] = {}
    if use_deepspeed:
        deepspeed_config = json.loads(
            Path(args.deepspeed_config).read_text(encoding="utf-8")
        )
        from deepspeed import initialize as ds_initialize

    train_model: torch.nn.Module
    if use_deepspeed:
        engine, optimizer, _, _ = ds_initialize(
            model=wrapper,
            optimizer=optimizer,
            config=deepspeed_config,
            dist_init_required=False,
        )
        wrapper = engine.module
        train_model = engine
    elif world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("distributed training requires CUDA")
        train_model = torch.nn.parallel.DistributedDataParallel(
            wrapper,
            device_ids=[device.index],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    else:
        train_model = wrapper

    scheduler = _CosineWithWarmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_learning_rate=float(config.get("min_learning_rate", 0.0)),
    )

    start_step = 0
    checkpoint_dir = Path(config["checkpoint_dir"])
    if use_deepspeed:
        resume_dir = Path(args.resume) if args.resume else checkpoint_dir / "ds"
        auto_resume = not args.no_resume and (resume_dir / "latest").exists()
    else:
        resume_dir = Path(args.resume) if args.resume else checkpoint_dir / "latest"
        auto_resume = not args.no_resume and (
            checkpoint_dir / "latest" / "trainer_state.json"
        ).exists()
    if args.resume or auto_resume:
        if use_deepspeed:
            if not (resume_dir / "latest").exists():
                raise FileNotFoundError(f"checkpoint not found: {resume_dir}/latest")
            trainer_state = _ds_load_checkpoint(
                engine,
                scheduler,
                rank,
                resume_dir,
            )
        else:
            if not (resume_dir / "trainer_state.json").exists():
                raise FileNotFoundError(f"checkpoint not found: {resume_dir}")
            trainer_state = _load_checkpoint(
                wrapper,
                optimizer,
                scheduler,
                rank,
                resume_dir,
                device,
            )
        start_step = int(trainer_state["step"])
        _rank0(rank, f"resumed from step {start_step}")
    if start_step >= total_steps:
        _rank0(rank, f"already at final step {start_step}")
        return 0

    if bool(config.get("use_bucket", False)):
        if world_size > 1:
            sampler = BucketProcessSampler(
                dataset,
                batch_size=int(config["micro_batch_size"]),
                world_size=world_size,
                rank=rank,
                seed=seed,
                bucket_sizes=bucket_sizes,
            )
        else:
            sampler = BucketOrderSampler(
                dataset,
                batch_size=int(config["micro_batch_size"]),
                seed=seed,
                bucket_sizes=bucket_sizes,
            )
    else:
        sampler = StepIndexSampler(
            dataset_size=len(dataset),
            world_size=world_size,
            rank=rank,
            start_step=start_step,
            total_steps=total_steps,
            gradient_accumulation_steps=accum,
            seed=seed,
        )
    collate = SegmentationCollator(wrapper.tokenizer.pad_token_id or 0)
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["micro_batch_size"]),
        sampler=sampler,
        collate_fn=collate,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)) and device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )

    save_interval = int(config.get("save_interval", 50))
    log_interval = int(config.get("log_interval", 10))
    run_directory = _run_dir(str(config["data_root"]), str(config["checkpoint_dir"]))
    metrics_path = run_directory / "metrics.jsonl"
    latest_dir = checkpoint_dir / ("ds" if use_deepspeed else "latest")

    expected_global = int(config["global_batch_size"])
    actual_global = int(config["micro_batch_size"]) * accum * world_size
    if expected_global != actual_global:
        _rank0(
            rank,
            f"warning: global_batch_size={expected_global}, "
            f"actual={actual_global} (micro*accum*world)",
        )

    _rank0(
        rank,
        f"train samples={len(dataset)} max_steps={total_steps} "
        f"grad_accum={accum} max_pixels={max_pixels}",
    )
    _rank0(rank, "starting training")
    micro_count = 0
    optimizer_step = start_step
    epoch_start = time.time()
    step_start = time.time()
    sums = {
        "total_loss": 0.0,
        "text_loss": 0.0,
        "seg_loss": 0.0,
        "iou_mse_loss": 0.0,
        "seg_samples": 0,
    }
    tokens_this_step = 0
    nan_steps = 0

    for batch in dataloader:
        output = train_model(batch)
        loss = output.get("total_loss")
        if loss is None or not torch.isfinite(loss):
            nan_steps += 1
            raise RuntimeError(f"non-finite training loss at global micro-step {optimizer_step * accum + micro_count}")
        if use_deepspeed:
            train_model.backward(loss)
        else:
            (loss / accum).backward()

        sums["total_loss"] += float(loss.detach().float())
        if output.get("text_loss") is not None:
            sums["text_loss"] += float(output["text_loss"].detach().float())
        if output.get("seg_loss") is not None:
            sums["seg_loss"] += float(output["seg_loss"].detach().float())
            sums["iou_mse_loss"] += float(output["iou_mse_loss"].detach().float())
            sums["seg_samples"] += 1
        tokens_this_step += int(batch["input_ids"].numel())
        micro_count += 1

        if use_deepspeed:
            is_opt_step = bool(train_model.is_gradient_accumulation_boundary())
        else:
            is_opt_step = micro_count >= accum

        if use_deepspeed:
            if is_opt_step:
                scheduler.step()
            train_model.step()
            if is_opt_step:
                global_grad_norm = train_model.get_global_grad_norm()
                grad_norm = float(global_grad_norm) if global_grad_norm is not None else 0.0
        else:
            if not is_opt_step:
                continue
            if float(config.get("grad_clip", 0.0)) > 0:
                clip_grad_norm_(
                    (p for p in wrapper.parameters() if p.requires_grad),
                    max_norm=float(config["grad_clip"]),
                )
            grad_norm = _grad_norm(wrapper)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not is_opt_step:
            continue

        optimizer_step += 1

        elapsed = time.time() - step_start
        row = {
            "step": optimizer_step,
            "total_loss": sums["total_loss"] / accum,
            "text_loss": sums["text_loss"] / accum if micro_count else 0.0,
            "seg_loss": sums["seg_loss"] / max(sums["seg_samples"], 1),
            "iou_mse_loss": sums["iou_mse_loss"] / max(sums["seg_samples"], 1),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "tokens": tokens_this_step,
            "elapsed_seconds": round(elapsed, 3),
            "memory_gb": (
                round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                if torch.cuda.is_available()
                else 0.0
            ),
            "epoch_elapsed_seconds": round(time.time() - epoch_start, 3),
        }
        if rank == 0 and optimizer_step % log_interval == 0:
            _log_metrics(metrics_path, row)
            print(json.dumps(row), flush=True)

        if optimizer_step % save_interval == 0 or optimizer_step >= total_steps:
            trainer_state = {
                "step": optimizer_step,
                "total_steps": total_steps,
                "seed": seed,
                "config_path": args.config,
            }
            _rank0(rank, f"saving checkpoint at step {optimizer_step} -> {latest_dir}")
            if use_deepspeed:
                _ds_save_checkpoint(
                    train_model,
                    scheduler,
                    trainer_state,
                    rank,
                    checkpoint_dir,
                )
            else:
                _save_checkpoint(
                    wrapper,
                    optimizer,
                    scheduler,
                    trainer_state,
                    rank,
                    world_size,
                    latest_dir,
                )

        micro_count = 0
        sums = {
            "total_loss": 0.0,
            "text_loss": 0.0,
            "seg_loss": 0.0,
            "iou_mse_loss": 0.0,
            "seg_samples": 0,
        }
        tokens_this_step = 0
        step_start = time.time()

        if optimizer_step >= total_steps:
            break

    if micro_count:
        _rank0(rank, "dataloader ended before optimizer step; incomplete accumulation ignored")
    _rank0(rank, f"training finished at step {optimizer_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
