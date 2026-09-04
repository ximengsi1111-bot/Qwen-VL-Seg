#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=qvlseg-infer-smoke-zr
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/infer-smoke-zr-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/infer-smoke-zr-%j.err

set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
source /etc/profile.d/lmod.sh
module load conda/3
module load cuda/13.0
source activate segment
cd /mingli01/project/xiyongkai/qwen3vl-seg

export DATA_ROOT=/file_storage01/home/mingli/data/xyk
export HF_HOME="$DATA_ROOT/cache/hf"
export PYTHONPATH=.
mkdir -p logs

python3 -m qwen3vl_seg.eval.runner \
  --checkpoint "$DATA_ROOT/checkpoints/small/stage2" \
  --model-path /mingli01/models/Qwen3-VL-4B-Instruct \
  --data-root "$DATA_ROOT" \
  --manifest "$DATA_ROOT/datasets/small/samples_smoke.jsonl" \
  --max-samples 8 \
  --max-pixels 262144 \
  --output-dir "$DATA_ROOT/runs/eval/smoke" \
  --save-viz
