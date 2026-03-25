#!/bin/bash -l
#SBATCH -A hackelb
#SBATCH --job-name=pmpnn_array
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
#SBATCH --output=slurm_logs/pmpnn_%A_%a.out
#SBATCH --error=slurm_logs/pmpnn_%A_%a.err

source activate pmpnn 
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1

python final_mpnn.py