#!/bin/bash -l
#SBATCH -A daoutidi
#SBATCH --time=24:00:00           # Adjust based on runtime per batch
#SBATCH --ntasks=1                # One task per array index
#SBATCH --cpus-per-task=2         # Number of CPUs per task
#SBATCH --mem=10g                 # Memory per task
#SBATCH --tmp=10g                 # Temporary disk space
#SBATCH -p msismall
#SBATCH --output=output_files/file.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mulli468@umn.edu
 
python graphml_to_features.py