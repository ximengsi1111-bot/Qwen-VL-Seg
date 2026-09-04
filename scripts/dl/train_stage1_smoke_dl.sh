#!/bin/bash
# Stage1 smoke launcher for the rented "DL" machine (2 x RTX 5090D 32GB).
# No SLURM: run directly inside a container / interactive shell.

set -e

PROJECT_ROOT="${DL_PROJECT_ROOT:-/root/qwen3vl-seg}"
DATA_ROOT="${DL_DATA_ROOT:-/root/data/xyk}"
MODEL_PATH="${DL_MODEL_PATH:-$DATA_ROOT/models/Qwen3-VL-4B-Instruct}"
HF_HOME="${DL_HF_HOME:-$DATA_ROOT/cache/hf}"
MASTER_PORT="${DL_MASTER_PORT:-29501}"

cd "$PROJECT_ROOT"
export DATA_ROOT
export HF_HOME
export PYTHONPATH="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${DL_CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/checkpoints/small"

deepspeed \
  --num_nodes=1 \
  --num_gpus=2 \
  --master_port="$MASTER_PORT" \
  --module qwen3vl_seg.train.training_loop \
  --config configs/train/stage1_small_2gpu_smoke.yaml \
  --deepspeed-config configs/train/deepspeed_stage1_smoke_2gpu.json \
  --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/datasets/small/samples.jsonl" \
  --checkpoint-dir "$DATA_ROOT/checkpoints/small/stage1-smoke-2gpu" \
  --smoke \
  --no-resume
