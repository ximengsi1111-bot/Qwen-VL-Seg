"""Minimal DeepSpeed engine smoke that exercises the real training-loop helpers."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3vl_seg.train.training_loop import (  # noqa: E402
    _CosineWithWarmup,
    _ds_load_checkpoint,
    _ds_save_checkpoint,
)


def main() -> int:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    groups = [
        {"params": list(model[0].parameters()), "lr": 1.0e-4},
        {"params": list(model[2].parameters()), "lr": 1.0e-5},
    ]
    optimizer = torch.optim.AdamW(groups, lr=1.0e-4)
    config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
        "gradient_clipping": 1.0,
        "communication_data_type": "fp32",
    }

    import deepspeed

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
        dist_init_required=False,
    )
    scheduler = _CosineWithWarmup(
        optimizer,
        total_steps=4,
        warmup_steps=1,
        min_learning_rate=0.0,
    )

    for _ in range(2):
        scheduler.step()
        loss = engine(torch.randn(1, 8)).sum()
        engine.backward(loss)
        engine.step()
        engine.zero_grad()

    checkpoint = Path(tempfile.mkdtemp(prefix="ds-engine-smoke-"))
    _ds_save_checkpoint(
        engine,
        scheduler,
        {"step": 2, "total_steps": 4, "seed": 0, "config_path": "test"},
        0,
        checkpoint,
    )
    restored = _ds_load_checkpoint(engine, scheduler, 0, checkpoint / "ds")
    assert restored["step"] == 2, restored
    assert (checkpoint / "ds" / "latest").exists()
    print("deepspeed-engine-smoke-ok", restored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
