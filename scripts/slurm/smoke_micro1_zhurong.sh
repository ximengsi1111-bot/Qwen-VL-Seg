#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu36,gpu37
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --job-name=qvlseg-micro1-smoke
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/micro1-smoke-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/micro1-smoke-%j.err

set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
source /persist_data/apps/miniconda3/etc/profile.d/conda.sh
conda activate segment
cd /mingli01/project/xiyongkai/qwen3vl-seg

export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
mkdir -p logs

MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
deepspeed \
  --num_nodes=1 \
  --num_gpus=4 \
  --master_port=$MASTER_PORT \
  --module qwen3vl_seg.train.training_loop \
  --config configs/train/stage1_small.yaml \
  --deepspeed-config configs/train/deepspeed_stage1_4gpu_zhurong.json \
  --manifest "$DATA_ROOT/datasets/small/samples_smoke.jsonl" \
  --checkpoint-dir "$DATA_ROOT/checkpoints/small/probe-micro1" \
  --smoke \
  --max-steps 2 \
  --max-pixels 262144 \
  --log-interval 1 \
  --no-resume
