#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=00:20:00
#SBATCH --job-name=qvlseg-nobucket-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/nobucket_smoke.out
#SBATCH --error=/mingli01/data/xyk/grpo/nobucket_smoke.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then module load conda/3 2>/dev/null || true; module load cuda/13.0 2>/dev/null || true; fi
export PATH=/persist_data/home/mingli/.conda/envs/segment/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export DATA_ROOT=/mingli01/data/xyk
export HF_HOME=/mingli01/data/xyk/cache/hf
export PYTHONPATH=.
OUT=/mingli01/data/xyk/runs/eval/nobucket-smoke
mkdir -p "$OUT"
PORT=$((31000 + SLURM_JOB_ID % 1000))
torchrun --nproc_per_node=1 --master_port=$PORT \
  --module qwen3vl_seg.eval.runner_multi \
  --checkpoint /mingli01/data/xyk/checkpoints/small/stage2-248k \
  --model-path /mingli01/models/Qwen3-VL-4B-Instruct \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/datasets/small/samples.cap8.jsonl" \
  --splits val \
  --max-pixels 262144 \
  --max-samples 8 \
  --output-dir "$OUT" \
  --shard-size 1 \
  --batch-size 4
python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"
