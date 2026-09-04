#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu36,gpu37
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --job-name=qvlseg-stage1-100k
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/stage1-100k-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/stage1-100k-%j.err

set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
source /etc/profile.d/lmod.sh
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
  --num_gpus=4 \
  --master_port=$MASTER_PORT \
  --module qwen3vl_seg.train.training_loop \
  --config configs/train/stage1_100k_zhurong.yaml \
  --deepspeed-config configs/train/deepspeed_stage1_100k_zhurong.json
