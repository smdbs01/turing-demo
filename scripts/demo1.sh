#!/bin/bash
#SBATCH --job-name=demo1
#SBATCH --output=logs/demo1-{%j}.out
#SBATCH --error=logs/demo1-{%j}.err
#SBATCH --time=00:01:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --partition=short

mkdir -p logs
# uv should handle the venv activation
NAME="${NAME:-demo1}"
uv run demo1.py --name=$NAME
