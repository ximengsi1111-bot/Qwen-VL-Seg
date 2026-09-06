#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:20:00
#SBATCH --job-name=qvlseg-grpo-eval
#SBATCH --output=/mingli01/data/xyk/grpo/eval.out
#SBATCH --error=/mingli01/data/xyk/grpo/eval.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then module load conda/3 2>/dev/null || true; module load cuda/13.0 2>/dev/null || true; fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export PYTHONPATH=.
export HF_HOME=/mingli01/data/xyk/cache/hf
python scripts/eval/grpo_eval_iou.py