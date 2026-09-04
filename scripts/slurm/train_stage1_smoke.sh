#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --job-name=qvlseg-stage1-smoke-2gpu
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/stage1-smoke-2gpu-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/stage1-smoke-2gpu-%j.err

set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
module load conda/3
module load cuda/13.0
source activate segment
cd /mingli01/project/xiyongkai/qwen3vl-seg

export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
mkdir -p logs

MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
deepspeed \
  --num_nodes=1 \
  --num_gpus=2 \
  --master_port=$MASTER_PORT \
  --module qwen3vl_seg.train.training_loop \
  --config configs/train/stage1_small_2gpu_smoke.yaml \
  --deepspeed-config configs/train/deepspeed_stage1_smoke_2gpu.json \
  --smoke \
  --no-resume
