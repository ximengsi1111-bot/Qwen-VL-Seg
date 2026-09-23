#!/bin/bash
#SBATCH --partition=gre
# gpu33 was occupied by a foreign process (~72.7 GiB) on 2026-09-22; keep it out of the smoke only
#SBATCH --exclude=gpu33
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --job-name=qvlseg-phrasecut-joint-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/newdata/phrasecut/phrasecut_grpo_joint_smoke_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/newdata/phrasecut/phrasecut_grpo_joint_smoke_%j.err
#
# Joint token+mask GRPO smoke test.
#
# pi_theta(y, m | x) = pi_theta(y | x) * pi_phi(m | x, y)
#   * token action y : vLLM rollout of boxes + <mask_token>          (L_token)
#   * mask action  m : Bernoulli sample of the decoder mask logits   (beta * L_mask)
#
# The mask decoder is frozen by default. Its outputs remain differentiable with
# respect to the Qwen hidden states, so beta * L_mask can update the LoRA policy
# while all decoder parameters stay fixed.
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=${MASTER_PORT:-29617}
export OPTIM=${OPTIM:-adamw_torch}
export NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PATH=/mingli01/data/xyk/conda/vllm-qwen3vl/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg-phrasecut
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
# ---- on-policy live-feature reward (frozen mode would set GRPO_JOINT_MASK_POLICY=0) ----
export GRPO_ON_POLICY_REWARD=1
export GRPO_ON_POLICY_MODE=reuse
export GRPO_ON_POLICY_DEBUG=1
export GRPO_ON_POLICY_BOX_WEIGHT=0.3
export GRPO_ON_POLICY_MASK_WEIGHT=0.7
# ---- joint mask policy (decoder frozen by default, mask action in the GRPO objective) ----
export GRPO_JOINT_MASK_POLICY=${GRPO_JOINT_MASK_POLICY:-1}
export GRPO_MASK_POLICY_FREEZE=${GRPO_MASK_POLICY_FREEZE:-1}
export GRPO_MASK_POLICY_BETA=${GRPO_MASK_POLICY_BETA:-0.01}
# decoder LR is only used when GRPO_MASK_POLICY_FREEZE=0
export GRPO_MASK_POLICY_LR=${GRPO_MASK_POLICY_LR:-1e-5}
export GRPO_MASK_POLICY_CLIP=${GRPO_MASK_POLICY_CLIP:-0.2}
export GRPO_MASK_POLICY_DTYPE=${GRPO_MASK_POLICY_DTYPE:-float32}
export GRPO_MASK_POLICY_LOGPROB=${GRPO_MASK_POLICY_LOGPROB:-mean}
export GRPO_MASK_POLICY_SAMPLE=${GRPO_MASK_POLICY_SAMPLE:-1}
# token-branch mask reward: `deterministic` (eval/SFT-identical) or `sampled`
export GRPO_MASK_POLICY_REWARD=${GRPO_MASK_POLICY_REWARD:-deterministic}
# optional: resume the mask decoder from a joint-training snapshot
export GRPO_JOINT_DECODER_INIT=${GRPO_JOINT_DECODER_INIT:-}
DATASET=${DATASET:-/mingli01/data/xyk/grpo/newdata/phrasecut/processed/phrasecut_5k_pilot.jsonl}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GEN_BATCH=${GEN_BATCH:-96}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
TOKEN_LR=${TOKEN_LR:-1e-4}
OUT_DIR=${OUT_DIR:-/mingli01/data/xyk/grpo/newdata/phrasecut/runs/phrasecut_5k_joint_smoke}
SAVE_STEPS=${SAVE_STEPS:-10}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
MAX_STEPS=${MAX_STEPS:-10}
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

echo "=== nvidia-smi: compute apps at job start ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
echo "=== end nvidia-smi ==="

python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" qwen3vl_seg/grpo/run_rlhf.py \
  --model /mingli01/data/xyk/model/grpo-policy-248k \
  --model_type qwen3_vl \
  --dataset "$DATASET" \
  --reward_funcs seg_iou \
  --rlhf_type grpo \
  --advantage_estimator grpo \
  --num_generations "$NUM_GENERATIONS" \
  --generation_batch_size "$GEN_BATCH" \
  --learning_rate "$TOKEN_LR" \
  --optim "$OPTIM" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --num_ppo_epochs 1 \
  --max_epochs 1 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
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
  --vllm_mm_processor_cache_gb 0 \
  --vllm_enforce_eager true \
  --vllm_max_model_len 8192 \
  --sleep_level 1 \
  --report_to none \
  --output_dir "$OUT_DIR"

echo DONE
