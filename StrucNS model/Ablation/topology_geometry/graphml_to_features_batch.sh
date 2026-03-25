#!/bin/bash -l
#SBATCH -A daoutidi
#SBATCH --time=24:00:00        # Adjust based on expected maximum runtime per chunk
#SBATCH --ntasks=1             # One task per array index
#SBATCH --cpus-per-task=2      # Number of CPUs per task
#SBATCH --mem=10g              # Memory per task
#SBATCH --tmp=10g              # Temporary disk space
#SBATCH -p msismall
#SBATCH --job-name=NS_FeatureExtraction
#SBATCH --output=logs/array_chunk_%a.out  # Unique log file for each array task
#SBATCH --error=logs/array_chunk_%a.err   # Unique error log for each array task
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mulli468@umn.edu

# --- Array Setup ---
# Set the total number of chunks
TOTAL_CHUNKS=100
# Job array runs from 1 to TOTAL_CHUNKS
# We convert the Slurm 1-indexed array ID to a 0-indexed chunk ID for Python
CHUNK_ID=$((SLURM_ARRAY_TASK_ID - 1))
# The array range will be --array=1-100

# Define your directories based on the Python script
OUTPUT_DIR="/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/StructureNS_analysis/random_walk_with_random_edges"

# Create log directory if it doesn't exist
mkdir -p logs

echo "Starting array task ${SLURM_ARRAY_TASK_ID} (Chunk ID: ${CHUNK_ID}) of ${TOTAL_CHUNKS}"
echo "Output will be saved to ${OUTPUT_DIR}/StructureNS_features_chunk_${CHUNK_ID}.csv"

# Load necessary modules (adjust for your environment)
# module load anaconda/xxxx # Example

# Execute the modified Python script
# Pass the 0-indexed CHUNK_ID and TOTAL_CHUNKS as arguments
python graphml_to_features_batch.py ${CHUNK_ID} ${TOTAL_CHUNKS}

echo "Array task ${SLURM_ARRAY_TASK_ID} finished."

# To submit this job:
# sbatch --array=1-${TOTAL_CHUNKS} submit_array_job.sh