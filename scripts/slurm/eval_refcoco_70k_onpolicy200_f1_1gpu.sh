#!/bin/bash
#SBATCH --partition=gre
#SBATCH --exclude=gpu33
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=qvlseg-rc-70k-onpolicy200-f1
#SBATCH --output=/mingli01/data/xyk/runs/eval/refcoco_70k_onpolicy200_f1_%j.out
#SBATCH --error=/mingli01/data/xyk/runs/eval/refcoco_70k_onpolicy200_f1_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-phrasecut
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
export MODEL=/mingli01/data/xyk/model/grpo-policy-248k
export CKPT=/mingli01/data/xyk/checkpoints/small/stage2-248k
export FINAL_ADAPTER=/mingli01/data/xyk/grpo/train248k_vllm_70k_g6_onpolicy_200/v0-20260913-190952/checkpoint-200
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
export MANIFEST_ROOT=/mingli01/data/xyk
export MANIFEST="$DATA_ROOT/build100k/datasets/small/samples.jsonl"
export SPLITS=val,testA,testB,test
export OUT_ROOT=${OUT_ROOT:-/mingli01/data/xyk/runs/eval/refcoco-70k-onpolicy200-f1}
declare -a RUNS=(
  "sft-baseline|"
  "grpo-70k-onpolicy200|$FINAL_ADAPTER"
)
mkdir -p "$OUT_ROOT"

run_one () {
  local TAG="$1"
  local ADAPTER_PATH="$2"
  local IDX="$3"
  local OUT="$OUT_ROOT/$TAG-$SLURM_JOB_ID"
  local PORT=$((37000 + SLURM_JOB_ID % 1000 + IDX))
  mkdir -p "$OUT"
  local ADAPTER_ARG=()
  if [ -n "$ADAPTER_PATH" ]; then ADAPTER_ARG=(--adapter "$ADAPTER_PATH"); fi
  echo "=== RUN $TAG adapter='$ADAPTER_PATH' -> $OUT ==="
  torchrun --nproc_per_node=1 --master_port="$PORT" \
    --module qwen3vl_seg.eval.runner_multi \
    --checkpoint "$CKPT" \
    --model-path "$MODEL" \
    "${ADAPTER_ARG[@]}" \
    --data-root "$MANIFEST_ROOT" \
    --manifest "$MANIFEST" \
    --splits "$SPLITS" \
    --max-pixels 262144 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-samples 0 \
    --no-viz \
    --output-dir "$OUT" \
    --shard-size 1
  python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
  echo "=== METRICS $TAG ==="
  cat "$OUT/metrics.json"
}

IDX=0
for entry in "${RUNS[@]}"; do
  TAG="${entry%%|*}"
  ADAPTER="${entry#*|}"
  run_one "$TAG" "$ADAPTER" "$IDX"
  IDX=$((IDX + 1))
done
echo "ALL DONE"
