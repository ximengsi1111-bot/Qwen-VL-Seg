#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=qvlseg-onpolicy-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/onpolicy_grpo_smoke_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/onpolicy_grpo_smoke_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-29620}
export OPTIM=${OPTIM:-adamw_torch}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export PATH=/mingli01/data/xyk/conda/vllm-qwen3vl/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-vllm
export PYTHONPATH=.
export HF_CACHE=/dev/shm/mingli/qwen3vl-onpolicy-cache-${SLURM_JOB_ID:-$$}
export HF_HOME=$HF_CACHE/hf
export HF_DATASETS_CACHE=$HF_CACHE/datasets
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export TMPDIR=$HF_CACHE/tmp
export MODELSCOPE_CACHE=$HF_CACHE/modelscope
export MS_CACHE_HOME=$MODELSCOPE_CACHE
trap 'case "$HF_CACHE" in /dev/shm/mingli/qwen3vl-onpolicy-cache-*) rm -rf "$HF_CACHE";; esac' EXIT
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$MODELSCOPE_CACHE"

export GRPO_BASE=/mingli01/data/xyk/model/grpo-policy-248k
export GRPO_STAGE2=/mingli01/data/xyk/checkpoints/small/stage2-248k
export GRPO_MAX_PIXELS=589824
export GRPO_ON_POLICY_REWARD=1
export GRPO_ON_POLICY_MODE=reuse
export GRPO_ON_POLICY_DEBUG=1
export GRPO_ON_POLICY_BOX_WEIGHT=0.3
export GRPO_ON_POLICY_MASK_WEIGHT=0.7

DATASET=${DATASET:-/mingli01/data/xyk/grpo/rl_smoke.jsonl}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
GEN_BATCH=${GEN_BATCH:-4}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
MAX_STEPS=${MAX_STEPS:-2}
OUT_DIR=${OUT_DIR:-/mingli01/data/xyk/grpo/onpolicy_reward_smoke_${SLURM_JOB_ID:-manual}}

python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" qwen3vl_seg/grpo/run_rlhf.py \
  --model /mingli01/data/xyk/model/grpo-policy-248k \
  --model_type qwen3_vl \
  --dataset "$DATASET" \
  --reward_funcs seg_iou \
  --rlhf_type grpo \
  --advantage_estimator grpo \
  --num_generations "$NUM_GENERATIONS" \
  --generation_batch_size "$GEN_BATCH" \
  --learning_rate 2e-5 \
  --optim "$OPTIM" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --num_ppo_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_enable_lora true \
  --lora_rank 8 \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --vllm_max_model_len 8192 \
  --sleep_level 1 \
  --report_to none \
  --output_dir "$OUT_DIR"

echo DONE
