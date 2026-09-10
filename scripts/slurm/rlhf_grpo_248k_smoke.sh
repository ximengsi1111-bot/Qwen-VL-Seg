#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=qvlseg-grpo248k-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/grpo248k_smoke.out
#SBATCH --error=/mingli01/data/xyk/grpo/grpo248k_smoke.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module >/dev/null 2>&1; then module load conda/3 2>/dev/null || true; module load cuda/13.0 2>/dev/null || true; fi
export PATH=/mingli01/data/xyk/conda/ms-swift-grpo/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export PYTHONPATH=.
export HF_HOME=/mingli01/data/xyk/cache/hf
# reward uses the 248k policy base + 248k stage2 decoder, aligned resolution
export GRPO_BASE=/mingli01/data/xyk/model/grpo-policy-248k
export GRPO_STAGE2=/mingli01/data/xyk/checkpoints/small/stage2-248k
export GRPO_MAX_PIXELS=589824
python qwen3vl_seg/grpo/run_rlhf.py \
  --model /mingli01/data/xyk/model/grpo-policy-248k \
  --model_type qwen3_vl \
  --dataset /mingli01/data/xyk/grpo/rl_hard.jsonl \
  --reward_funcs seg_iou \
  --rlhf_type grpo \
  --advantage_estimator grpo \
  --num_generations 8 \
  --generation_batch_size 8 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 2 \
  --num_ppo_epochs 1 \
  --max_steps 50 \
  --report_to none \
  --output_dir /mingli01/data/xyk/grpo/train248k_smoke
echo DONE
