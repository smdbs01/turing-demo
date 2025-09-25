#!/bin/bash
#SBATCH --job-name=demo2
#SBATCH --output=logs/demo2-%j.out
#SBATCH --error=logs/demo2-%j.err
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=short

# uv should handle the venv activation
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
BETA="${BETA:-1.0}"

uv run demo2.py --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lr=$LR --beta=$BETA
