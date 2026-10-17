#!/bin/bash
#SBATCH --partition=gre
#SBATCH --exclude=gpu33
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --job-name=qvlseg-phrasecut-f1-200-viz
#SBATCH --output=/mingli01/data/xyk/grpo/newdata/phrasecut/phrasecut_f1_200_viz_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/newdata/phrasecut/phrasecut_f1_200_viz_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-phrasecut
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
export MODEL=/mingli01/data/xyk/model/grpo-policy-248k
export CKPT=/mingli01/data/xyk/checkpoints/small/stage2-248k
export MANIFEST=/mingli01/data/xyk/grpo/newdata/phrasecut/test_processed/phrasecut_1200_f1_visual12.jsonl
export F1_ADAPTER=/mingli01/data/xyk/grpo/newdata/phrasecut/runs/phrasecut_50k_f1_200step/v0-20260926-154707/checkpoint-200
export OUT_ROOT=${OUT_ROOT:-/mingli01/data/xyk/runs/visuals/phrasecut-joint-f1-200-compare-v1}
mkdir -p "$OUT_ROOT"

run_one () {
  local TAG="$1"
  local ADAPTER_PATH="$2"
  local IDX="$3"
  local OUT="$OUT_ROOT/$TAG-$SLURM_JOB_ID"
  local PORT=$((36000 + SLURM_JOB_ID % 1000 + IDX))
  mkdir -p "$OUT"
  local ADAPTER_ARG=()
  if [ -n "$ADAPTER_PATH" ]; then ADAPTER_ARG=(--adapter "$ADAPTER_PATH"); fi
  echo "=== RUN $TAG adapter='$ADAPTER_PATH' -> $OUT ==="
  torchrun --nproc_per_node=1 --master_port="$PORT" \
    --module qwen3vl_seg.eval.runner_multi \
    --checkpoint "$CKPT" \
    --model-path "$MODEL" \
    "${ADAPTER_ARG[@]}" \
    --data-root / \
    --manifest "$MANIFEST" \
    --splits phrasecut_train \
    --max-pixels 262144 \
    --max-new-tokens 512 \
    --max-samples 0 \
    --save-viz \
    --output-dir "$OUT" \
    --shard-size 1
  python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
}

run_one baseline "" 0
run_one grpo-joint-f1-200 "$F1_ADAPTER" 1
echo "ALL DONE"
