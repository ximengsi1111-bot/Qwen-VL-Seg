#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --job-name=qvlseg-train248k-score
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/train248k-score-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/train248k-score-%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then module load conda/3 2>/dev/null || true; module load cuda/13.0 2>/dev/null || true; fi
export PATH=/persist_data/home/mingli/.conda/envs/segment/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export DATA_ROOT=/mingli01/data/xyk
export HF_HOME=/mingli01/data/xyk/cache/hf
export PYTHONPATH=.
mkdir -p logs
OUT=/mingli01/data/xyk/runs/eval/train248k-score
mkdir -p "$OUT"
PORT=$((31000 + SLURM_JOB_ID % 1000))
torchrun --nproc_per_node=4 --master_port=$PORT \
  --module qwen3vl_seg.eval.runner_multi \
  --checkpoint /mingli01/data/xyk/checkpoints/small/stage2-248k \
  --model-path /mingli01/models/Qwen3-VL-4B-Instruct \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/datasets/small/samples.cap8.jsonl" \
  --splits train \
  --max-pixels 589824 \
  --max-samples 0 \
  --output-dir "$OUT" \
  --shard-size 4 \
  --batch-size 4 \
  --use-bucket \
  --no-viz \
  --resume-done "$OUT/predictions.done2.jsonl"
echo DONE
