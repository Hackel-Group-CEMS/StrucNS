#!/bin/bash -l
#SBATCH -A daoutidi
#SBATCH --time=24:00:00             # Set runtime based on your expected duration
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=1           # Number of CPUs per task
#SBATCH --mem=50g
#SBATCH --tmp=50g                    # Memory per task (adjust as needed)
#SBATCH --requeue
#SBATCH -p preempt  
#SBATCH --output=slurm_output_%j.out # Output file with job ID
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mulli468@umn.edu



source ~/esm2_env/bin/activate

python run_one_shot.py
