#!/bin/bash -l

#SBATCH -A hackelb

#SBATCH --time=24:00:00             # Adjust based on runtime per batch

#SBATCH --ntasks=1                  # One task per array index

#SBATCH --cpus-per-task=1           # Number of CPUs per task

#SBATCH --mem=20g                    # Memory per task

#SBATCH --tmp=20g                   # Temporary disk space

#SBATCH -p msismall              # Partition name

#SBATCH --output=output_files/unire_Embed.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=mulli468@umn.edu



python unirep_input_prep.py

