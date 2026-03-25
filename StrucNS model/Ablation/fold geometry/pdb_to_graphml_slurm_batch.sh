#!/bin/bash -l

#SBATCH -A hackelb

# --- Array Job Configuration ---
# Job name for easy identification
#SBATCH --job-name=PDB_Graph_Array
# Set the total number of array tasks (chunks)
#SBATCH --array=1-100 
#SBATCH --time=24:00:00        
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50g
#SBATCH --tmp=50g
#SBATCH -p msismall
#SBATCH --output=output_files/array_logs/%x_%A_%a.out 
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mulli468@umn.edu

# --- Execution ---

# 1. Main Processing Phase (Job Array)
echo "Starting Array Job $SLURM_ARRAY_TASK_ID"

# CALLING STATEMENT: Passes the 1-based array ID to the Python script
python pdb_to_graphml_batch.py $SLURM_ARRAY_TASK_ID 

echo "Array Job $SLURM_ARRAY_TASK_ID Finished."