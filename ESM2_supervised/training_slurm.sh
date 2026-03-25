#!/bin/bash -l

# --- SLURM DIRECTIVES ---
#SBATCH -A hackelb
#SBATCH --job-name=esm2_training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               
#SBATCH --mem=60g
#SBATCH --tmp=50g
#SBATCH -p msigpu
#SBATCH --gres=gpu:1                    
#SBATCH --output=slurm_logs/Base_training_%j.out   # %j = Job ID
#SBATCH --error=slurm_logs/Base_training_%j.err    # Separate error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mulli468@umn.edu

# Ensure log directory exists so Slurm doesn't fail silently
mkdir -p slurm_logs

singularity exec --nv \
    --bind /projects/standard/hackelb:/projects/standard/hackelb \
    --bind /scratch.global/hackelb:/scratch.global/hackelb \
    docker://tensorflow/tensorflow:2.16.1-gpu \
    bash -c "
        pip install --no-cache-dir pandas scikit-learn matplotlib optuna &&
        python training.py
    "