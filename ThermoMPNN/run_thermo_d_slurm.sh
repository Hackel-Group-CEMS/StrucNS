#!/bin/bash -l
#SBATCH -A hackelb
#SBATCH --job-name=thermo_d
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --tmp=20g
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/thermo_%j.out
#SBATCH --error=slurm_logs/thermo_%j.err

# 1. Create logs directory (inside your current experiment folder)
mkdir -p slurm_logs

# 2. Define the folder name where your script lives
# (Replace 'my_experiment' with the actual name of your folder)
MY_FOLDER="case1"

# 3. Move UP to the ThermoMPNN-D root directory
cd ..

# 4. Run Singularity from the Root
# We bind "$PWD" which is now the Root directory
singularity exec --nv \
    --bind /projects/standard/hackelb:/projects/standard/hackelb \
    --bind /scratch.global/hackelb:/scratch.global/hackelb \
    --bind "$PWD" \
    docker://pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
    bash -c "pip install --no-cache-dir pandas scikit-learn biopython scipy && \
             export PYTHONPATH=\$PYTHONPATH:. && \
             python $MY_FOLDER/run_thermompnn_d.py"