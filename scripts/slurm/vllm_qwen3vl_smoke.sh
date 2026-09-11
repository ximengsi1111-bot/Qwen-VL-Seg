#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=qvlseg-vllm-smoke
#SBATCH --output=/mingli01/data/xyk/grpo/vllm_smoke_%j.out
#SBATCH --error=/mingli01/data/xyk/grpo/vllm_smoke_%j.err
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PATH=/mingli01/data/xyk/conda/vllm-qwen3vl/bin:$PATH
export HF_HOME=/mingli01/data/xyk/cache/hf
cd /mingli01/project/xiyongkai/qwen3vl-seg-vllm
python scripts/eval/vllm_qwen3vl_smoke.py
echo DONE
