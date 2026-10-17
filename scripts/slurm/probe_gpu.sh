#!/bin/bash
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:05:00
#SBATCH --job-name=qvlseg-gpu-probe
#SBATCH --output=/mingli01/data/xyk/grpo/gpu_probe.out
#SBATCH --error=/mingli01/data/xyk/grpo/gpu_probe.err
nvidia-smi -L
echo "---"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
