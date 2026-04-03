#!/bin/bash
#SBATCH --job-name=lora_autotune
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=0-04:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load apptainer

apptainer exec --nv ~/apptainer.sandbox python3.12 lora_helion_autotune.py
