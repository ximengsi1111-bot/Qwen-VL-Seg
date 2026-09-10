#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=qvlseg-convert-policy248k
#SBATCH --output=/mingli01/project/xiyongkai/qwen3vl-seg/logs/convert-policy248k-%j.out
#SBATCH --error=/mingli01/project/xiyongkai/qwen3vl-seg/logs/convert-policy248k-%j.err
set -e
export PYTHONNOUSERSITE=1
export PATH=/persist_data/home/mingli/.conda/envs/segment/bin:$PATH
cd /mingli01/project/xiyongkai/qwen3vl-seg
export HF_HOME=/mingli01/data/xyk/cache/hf
export PYTHONPATH=.
python scripts/model/convert_stage2_policy.py \
  /mingli01/data/xyk/checkpoints/small/stage2-248k/ds \
  /mingli01/data/xyk/model/grpo-policy-248k \
  /mingli01/models/Qwen3-VL-4B-Instruct
echo DONE
