#!/bin/bash -l
#SBATCH -A hackelb

#SBATCH --time=02:00:00             # Adjust based on runtime per batch

#SBATCH --ntasks=1                  # One task per array index

#SBATCH --cpus-per-task=2           # Number of CPUs per task

#SBATCH --mem=50g                    # Memory per task

#SBATCH --tmp=50g                   # Temporary disk space

#SBATCH -p msismall

#SBATCH --output=output_files/merge.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=mulli468@umn.edu

python merge_deltaG.py

