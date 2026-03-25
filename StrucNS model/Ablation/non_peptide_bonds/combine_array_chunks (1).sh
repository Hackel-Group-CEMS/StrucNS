#!/bin/bash -l
#SBATCH -A daoutidi
#SBATCH --time=01:00:00        # Should be quick, adjust if needed
#SBATCH --ntasks=1             # Single task
#SBATCH --cpus-per-task=1
#SBATCH --mem=5g               # Memory for reading/writing all CSVs
#SBATCH -p msismall
#SBATCH --job-name=NS_Consolidation
#SBATCH --output=logs/consolidation.out
#SBATCH --error=logs/consolidation.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mulli468@umn.edu

# Define your directories and parameters
OUTPUT_DIR="/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/StructureNS_analysis/no_nonc_edges"
MASTER_CSV="${OUTPUT_DIR}/StructureNS_features_MASTER.csv"
TOTAL_CHUNKS=100 # Must match the array job submission

echo "Starting CSV consolidation..."
mkdir -p logs

# Remove old master CSV if it exists to ensure a clean rebuild
if [ -f "$MASTER_CSV" ]; then
    rm "$MASTER_CSV"
    echo "Removed existing master CSV: $MASTER_CSV"
fi

# Find the first non-empty chunk CSV to use its header
FIRST_CSV=""
for i in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    CHUNK_FILE="${OUTPUT_DIR}/StructureNS_features_chunk_${i}.csv"
    if [ -s "$CHUNK_FILE" ]; then
        FIRST_CSV="$CHUNK_FILE"
        break
    fi
done

if [ -z "$FIRST_CSV" ]; then
    echo "🚨 ERROR: No non-empty chunk CSVs found. Nothing to consolidate."
    exit 1
fi

echo "Using header from $FIRST_CSV"

# 1. Copy the header of the first non-empty file to the master CSV
head -n 1 "$FIRST_CSV" > "$MASTER_CSV"

# 2. Append the content (excluding the header line) of all chunk files
for i in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    CHUNK_FILE="${OUTPUT_DIR}/StructureNS_features_chunk_${i}.csv"
    if [ -s "$CHUNK_FILE" ]; then # -s checks if file exists and is not zero size
        # tail -n +2 skips the header line from the chunk file
        tail -n +2 "$CHUNK_FILE" >> "$MASTER_CSV"
        echo "Appended $CHUNK_FILE"
    else
        echo "Chunk file $CHUNK_FILE is empty or missing, skipping."
    fi
done

echo "✨ Consolidation complete. Final file saved to $MASTER_CSV"
echo "You may now remove the individual chunk files: StructureNS_features_chunk_*.csv"

# To submit this job after the array job finishes:
# sbatch combine_array_chunks.sh