#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=qvlseg-phrasecut5k-eval
#SBATCH --output=/mingli01/data/xyk/grpo/newdata/phrasecut/phrasecut5k_eval_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/newdata/phrasecut/phrasecut5k_eval_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then
  module load conda/3 2>/dev/null || true
  module load cuda/13.0 2>/dev/null || true
fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-phrasecut
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
export MODEL=/mingli01/data/xyk/model/grpo-policy-248k
export CKPT=/mingli01/data/xyk/checkpoints/small/stage2-248k
export RUN_ROOT=${RUN_ROOT:-/mingli01/data/xyk/grpo/newdata/phrasecut/runs/phrasecut_5k_200}
export ADAPTER=${ADAPTER:-$(find "$RUN_ROOT" -type d -name checkpoint-200 2>/dev/null | sort | tail -1)}
if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
  echo "checkpoint-200 not found under $RUN_ROOT" >&2
  exit 1
fi
export SPLITS=val,testA,testB,test
export OUT_ROOT=${OUT_ROOT:-/mingli01/data/xyk/runs/eval/phrasecut5k-200}
mkdir -p "$OUT_ROOT"
OUT="$OUT_ROOT/checkpoint-200-$SLURM_JOB_ID"
PORT=$((31000 + SLURM_JOB_ID % 1000))
mkdir -p "$OUT"
echo "=== EVAL adapter=$ADAPTER -> $OUT ==="
torchrun --nproc_per_node=1 --master_port="$PORT" \
  --module qwen3vl_seg.eval.runner_multi \
  --checkpoint "$CKPT" \
  --model-path "$MODEL" \
  --adapter "$ADAPTER" \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/build100k/datasets/small/samples.jsonl" \
  --splits "$SPLITS" \
  --max-pixels 262144 \
  --max-samples 0 \
  --no-viz \
  --output-dir "$OUT" \
  --shard-size 1
python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
echo "=== METRICS ==="
cat "$OUT/metrics.json"
echo "ALL DONE"
