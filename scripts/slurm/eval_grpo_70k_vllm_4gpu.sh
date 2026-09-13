#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --job-name=qvlseg-grpo70k-eval
#SBATCH --output=/mingli01/data/xyk/grpo/grpo70k_vllm_eval_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/grpo70k_vllm_eval_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then
  module load conda/3 2>/dev/null || true
  module load cuda/13.0 2>/dev/null || true
fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-vllm
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
export MODEL=/mingli01/data/xyk/model/grpo-policy-248k
export CKPT=/mingli01/data/xyk/checkpoints/small/stage2-248k
export ADAPTER=/mingli01/data/xyk/grpo/train248k_vllm_70k_g6_formal/v3-20260912-111920/checkpoint-2920
export SPLITS=val,testA,testB,test
export OUT_ROOT=${OUT_ROOT:-/mingli01/data/xyk/runs/eval/grpo70k-vllm-final}
mkdir -p "$OUT_ROOT"

run_one () {
  local TAG="$1"
  local ADAPTER_PATH="$2"
  local IDX="$3"
  local OUT="$OUT_ROOT/$TAG-$SLURM_JOB_ID"
  local PORT=$((31000 + SLURM_JOB_ID % 1000 + IDX))
  mkdir -p "$OUT"
  local ADAPTER_ARG=()
  if [ -n "$ADAPTER_PATH" ]; then ADAPTER_ARG=(--adapter "$ADAPTER_PATH"); fi
  echo "=== RUN $TAG adapter='$ADAPTER_PATH' -> $OUT ==="
  torchrun --nproc_per_node=4 --master_port="$PORT" \
    --module qwen3vl_seg.eval.runner_multi \
    --checkpoint "$CKPT" \
    --model-path "$MODEL" \
    "${ADAPTER_ARG[@]}" \
    --data-root "$DATA_ROOT" \
    --manifest "$DATA_ROOT/build100k/datasets/small/samples.jsonl" \
    --splits "$SPLITS" \
    --max-pixels 262144 \
    --max-samples 0 \
    --no-viz \
    --output-dir "$OUT" \
    --shard-size 4
  python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
  echo "=== METRICS $TAG ==="
  cat "$OUT/metrics.json"
}

run_one baseline "" 0
run_one grpo-final "$ADAPTER" 1
echo "ALL DONE"
