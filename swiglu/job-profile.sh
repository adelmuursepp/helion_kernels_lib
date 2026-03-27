#!/bin/bash
#SBATCH --job-name=swiglu_profile
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=0-00:10
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load apptainer

apptainer exec --nv ~/helion.sif python8 benchmark.py

