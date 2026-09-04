#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=qvlseg-env-smoke
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/env-smoke-%j.out

set -e
export PYTHONNOUSERSITE=1
module load conda/3
module load cuda/13.0
source activate segment
cd /mingli01/project/xiyongkai/qwen3vl-seg

export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"

mkdir -p "$DATA_ROOT/logs"
python scripts/env_check.py
