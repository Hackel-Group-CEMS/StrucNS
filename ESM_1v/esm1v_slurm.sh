#!/bin/bash -l
#SBATCH -A daoutidi
#SBATCH --time=24:00:00             # 24 hours should be sufficient for 3 datasets
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4           # Increased slightly for data loading
#SBATCH --mem=64g                   # 60GB to be safe with model + 3 CSVs processing
#SBATCH --tmp=30g
#SBATCH -p msigpu
#SBATCH --gres=gpu:1 
#SBATCH --array=0-2                 # Create 3 jobs (0, 1, and 2)
#SBATCH --output=slurm_logs/esm1v_zs_%A_%a.out  # %A is JobID, %a is ArrayID
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mulli468@umn.edu




source ~/esm2_env/bin/activate
# Add this line here
export TORCH_HOME=$HOME

echo "Starting ESM-1v Zero-Shot for Array ID: $SLURM_ARRAY_TASK_ID"
python run_esm1v.py
echo "Job Complete."