#!/bin/bash -l

# --- SLURM DIRECTIVES ---
#SBATCH -A hackelb
#SBATCH --job-name=OF
#SBATCH --time=24:00:00 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/omegafold_%A_%a.out
#SBATCH --array=0-130
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mulli468@umn.edu

# --- Configuration ---
FILES_PER_TASK=100 
OMEGAFOLD_TIMEOUT="15m" 

# 1. SETUP DIRECTORIES IMMEDIATELY
# Using $(pwd) ensures we know exactly where folders are being made
BASE_DIR=$(pwd)
INPUT_DIR="${BASE_DIR}/variant_fastas"
OUTPUT_DIR="${BASE_DIR}/omegafold_pdbs"
LOG_DIR="${BASE_DIR}/processing_logs"
SLURM_OUT_DIR="${BASE_DIR}/slurm_logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${SLURM_OUT_DIR}"

CENTRAL_LOG_FILE="${LOG_DIR}/completed_fasta_files.txt"

# 2. ENVIRONMENT
module load conda
eval "$(conda shell.bash hook)"
conda activate omegafold_env

# 3. GET FILES
# Check if the directory exists first to avoid silent failure
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory $INPUT_DIR not found!"
    exit 1
fi

ALL_FASTA_FILES=(${INPUT_DIR}/*.fasta)
TOTAL_FILES=${#ALL_FASTA_FILES[@]}

# 4. RANGE LOGIC
START_INDEX=$((SLURM_ARRAY_TASK_ID * FILES_PER_TASK))
END_INDEX=$((START_INDEX + FILES_PER_TASK - 1))

echo "--- Task ${SLURM_ARRAY_TASK_ID} Started ---"
echo "Working in: ${BASE_DIR}"

# 5. LOOP
for (( i=${START_INDEX}; i<=${END_INDEX}; i++ )); do
    if (( i >= TOTAL_FILES )); then break; fi

    INPUT_FASTA="${ALL_FASTA_FILES[i]}"
    FASTA_BASE_NAME=$(basename "${INPUT_FASTA}")

    # OmegaFold names output based on the ID after '>' in the file
    # We grab that here to check if the .pdb already exists
    PDB_ID=$(sed -n '1p' "${INPUT_FASTA}" | sed 's/^>//' | awk '{print $1}')
    OUTPUT_PDB_FILE="${OUTPUT_DIR}/${PDB_ID}.pdb"

    # Robust Check
    if [ -f "${OUTPUT_PDB_FILE}" ] || grep -q "^${FASTA_BASE_NAME}$" "${CENTRAL_LOG_FILE}" 2>/dev/null; then
        echo "✅ Skip: ${FASTA_BASE_NAME}"
        continue
    fi

    echo "Folding Index ${i}: ${FASTA_BASE_NAME}"
    
    # Run OmegaFold
    timeout ${OMEGAFOLD_TIMEOUT} omegafold "${INPUT_FASTA}" "${OUTPUT_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "${FASTA_BASE_NAME}" >> "${CENTRAL_LOG_FILE}"
    fi
done