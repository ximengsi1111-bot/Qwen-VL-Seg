#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=qvlseg-eval-248k-multi
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/eval-formal-248k-multi-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/eval-formal-248k-multi-%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then
  module load conda/3 2>/dev/null || true
  module load cuda/13.0 2>/dev/null || true
fi
export PATH=/persist_data/home/mingli/.conda/envs/segment/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
export OUT=/mingli01/data/xyk/runs/eval/stage2-248k-multi-formal
mkdir -p "$OUT" logs
MASTER_PORT=$((31000 + SLURM_JOB_ID % 1000))
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
  --module qwen3vl_seg.eval.runner_multi \
  --checkpoint /mingli01/data/xyk/checkpoints/small/stage2-248k \
  --model-path /mingli01/models/Qwen3-VL-4B-Instruct \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/build100k/datasets/small/samples.jsonl" \
  --splits val,testA,testB,test \
  --max-pixels 262144 \
  --max-samples 0 \
  --output-dir "$OUT" \
  --shard-size 4
python -m qwen3vl_seg.eval.merge_eval_metrics "$OUT"