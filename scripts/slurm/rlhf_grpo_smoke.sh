#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --job-name=qvlseg-grpo-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/smoke.out
#SBATCH --error=/mingli01/data/xyk/grpo/smoke.err

set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then
  module load conda/3 2>/dev/null || true
  module load cuda/13.0 2>/dev/null || true
fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export PYTHONPATH=.
export HF_HOME=/mingli01/data/xyk/cache/hf
mkdir -p /mingli01/data/xyk/grpo

python qwen3vl_seg/grpo/run_rlhf.py \
  --model /mingli01/data/xyk/model/grpo-policy \
  --model_type qwen3_vl \
  --dataset /mingli01/data/xyk/grpo/rl_smoke.jsonl \
  --reward_funcs seg_iou \
  --rlhf_type grpo \
  --advantage_estimator grpo \
  --num_generations 4 \
  --generation_batch_size 4 \
  --num_ppo_epochs 1 \
  --max_epochs 1 \
  --report_to none \
  --output_dir /mingli01/data/xyk/grpo/smoke_out