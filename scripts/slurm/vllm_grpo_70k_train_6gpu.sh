#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --job-name=qvlseg-vllm-grpo70k-train
#SBATCH --output=/mingli01/data/xyk/grpo/vllm_grpo_70k_train_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/vllm_grpo_70k_train_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-29606}
export OPTIM=${OPTIM:-adamw_torch}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PATH=/mingli01/data/xyk/conda/vllm-qwen3vl/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-vllm
export PYTHONPATH=.
export HF_CACHE=/dev/shm/mingli/qwen3vl-vllm-cache-${SLURM_JOB_ID:-$$}
export HF_HOME=$HF_CACHE/hf
export HF_DATASETS_CACHE=$HF_CACHE/datasets
export TRANSFORMERS_CACHE=$HF_CACHE/transformers
export TMPDIR=$HF_CACHE/tmp
export MODELSCOPE_CACHE=$HF_CACHE/modelscope
export MS_CACHE_HOME=$MODELSCOPE_CACHE
trap 'case "$HF_CACHE" in /dev/shm/mingli/qwen3vl-vllm-cache-*) rm -rf "$HF_CACHE";; esac' EXIT
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$MODELSCOPE_CACHE"
export GRPO_BASE=/mingli01/data/xyk/model/grpo-policy-248k
export GRPO_STAGE2=/mingli01/data/xyk/checkpoints/small/stage2-248k
export GRPO_MAX_PIXELS=589824
DATASET=${DATASET:-/mingli01/data/xyk/grpo/rl_hard_70k.jsonl}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GEN_BATCH=${GEN_BATCH:-96}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
OUT_DIR=${OUT_DIR:-/mingli01/data/xyk/grpo/train248k_vllm_70k_g6_formal}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
LOAD_ARGS=${LOAD_ARGS:-true}
RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-}
if [ -z "$RESUME_ONLY_MODEL" ]; then
  if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ONLY_MODEL=true
  else
    RESUME_ONLY_MODEL=false
  fi
fi
RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

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
  --max_epochs 1 \
  --load_args "$LOAD_ARGS" \
  --resume_only_model "$RESUME_ONLY_MODEL" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  "${RESUME_ARGS[@]}" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_enable_lora true \
  --lora_rank 8 \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_enforce_eager true \
  --vllm_max_model_len 8192 \
  --sleep_level 1 \
  --report_to none \
  --output_dir "$OUT_DIR"

echo DONE




