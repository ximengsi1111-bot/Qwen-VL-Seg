#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=00:20:00
#SBATCH --job-name=qvlseg-batch-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/batch_smoke.out
#SBATCH --error=/mingli01/data/xyk/grpo/batch_smoke.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then module load conda/3 2>/dev/null || true; module load cuda/13.0 2>/dev/null || true; fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
OUT=/mingli01/data/xyk/runs/eval/batch-smoke
mkdir -p "$OUT"
PORT=$((31000 + SLURM_JOB_ID % 1000))
torchrun --nproc_per_node=1 --master_port=$PORT \
  --module qwen3vl_seg.eval.runner_multi \
  --checkpoint /file_storage01/home/mingli/data/xyk/checkpoints/small/stage2-100k \
  --model-path /mingli01/data/xyk/model/grpo-policy \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/build100k/datasets/small/samples.jsonl" \
  --splits val \
  --max-pixels 262144 \
  --max-samples 8 \
  --output-dir "$OUT" \
  --shard-size 1 \
  --batch-size 4
python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
