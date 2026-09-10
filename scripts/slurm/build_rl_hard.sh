#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=qvlseg-build-rlhard
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/build-rlhard-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/build-rlhard-%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export WORKERS=32
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export PYTHONPATH=.
python scripts/data/build_grpo_hard_aug.py
echo DONE
