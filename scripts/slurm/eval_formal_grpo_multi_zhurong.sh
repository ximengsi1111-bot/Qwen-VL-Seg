#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=qvlseg-grpo-formal-multi
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/eval-grpo-formal-multi-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/eval-grpo-formal-multi-%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then
  module load conda/3 2>/dev/null || true
  module load cuda/13.0 2>/dev/null || true
fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
export MODEL=/mingli01/data/xyk/model/grpo-policy
export CKPT=/file_storage01/home/mingli/data/xyk/checkpoints/small/stage2-100k
export SPLITS=val,testA,testB,test
mkdir -p logs

declare -a RUNS=(
  "baseline|"
  "grpo500|/mingli01/data/xyk/grpo/train_out/v0-20260908-124037/checkpoint-500"
  "grpo1000|/mingli01/data/xyk/grpo/train_out/v0-20260908-124037/checkpoint-1000"
)

run_one () {
  local TAG="$1" ADAPTER="$2"
  local OUT=/mingli01/data/xyk/runs/eval/grpo-formal-$TAG
  mkdir -p "$OUT"
  local ADAPTER_ARG=()
  if [ -n "$ADAPTER" ]; then ADAPTER_ARG=(--adapter "$ADAPTER"); fi
  local PORT=$((31000 + SLURM_JOB_ID % 1000 + $3))
  echo "=== RUN $TAG (adapter='$ADAPTER') -> $OUT ==="
  torchrun --nproc_per_node=4 --master_port=$PORT \
    --module qwen3vl_seg.eval.runner_multi \
    --checkpoint "$CKPT" \
    --model-path "$MODEL" \
    "${ADAPTER_ARG[@]}" \
    --data-root "$DATA_ROOT" \
    --manifest "$DATA_ROOT/build100k/datasets/small/samples.jsonl" \
    --splits "$SPLITS" \
    --max-pixels 262144 \
    --max-samples 0 \
    --output-dir "$OUT" \
    --shard-size 4
  python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
}

IDX=0
for entry in "${RUNS[@]}"; do
  TAG="${entry%%|*}"
  ADAPTER="${entry#*|}"
  run_one "$TAG" "$ADAPTER" "$IDX"
  IDX=$((IDX + 1))
done
echo "ALL DONE"
