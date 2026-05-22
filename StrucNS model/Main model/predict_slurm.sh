#!/bin/bash -l

# --- SLURM DIRECTIVES ---
#SBATCH -A hackelb
#SBATCH --job-name=Model_Inference
#SBATCH --time=05:00:00            # Predictions are much faster than training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                
#SBATCH --mem=64g                  # Memory requirement is usually lower for inference
#SBATCH -p msigpu
#SBATCH --gres=gpu:1               # Still using GPU for faster inference
#SBATCH --output=slurm_logs/Predict_%j.out
#SBATCH --error=slurm_logs/Predict_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mulli468@umn.edu

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Execute the prediction script inside the container
singularity exec --nv \
    --bind /projects/standard/hackelb:/projects/standard/hackelb \
    --bind /scratch.global/hackelb:/scratch.global/hackelb \
    docker://tensorflow/tensorflow:2.16.1-gpu \
    bash -c "
        pip install --no-cache-dir pandas scikit-learn joblib &&
        python run_predictions.py
    "