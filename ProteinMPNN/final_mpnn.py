import os
import pandas as pd
import subprocess
import re
import sys
import shutil

# --- CONFIGURATION ---
BASE_PATH = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/feature_datasets/StrucNS_sets/Training/base_model/case1/predictions"
WT_PDB_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs_WT"
MUT_PDB_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
MPNN_SCRIPT = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/feature_datasets/ProteinMPNN/protein_mpnn_run.py"
PYTHON_EXE = "/users/5/mulli468/.conda/envs/pmpnn/bin/python"

def get_mpnn_score(pdb_path, temp_out):
    if not pdb_path or not os.path.exists(pdb_path):
        print(f"  [ERROR] PDB NOT FOUND: {pdb_path}")
        return None
    
    seqs_dir = os.path.join(temp_out, "seqs")
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(seqs_dir, exist_ok=True)
    
    cmd = [PYTHON_EXE, MPNN_SCRIPT, "--pdb_path", pdb_path, "--pdb_path_chains", "A", 
           "--out_folder", temp_out, "--num_seq_per_target", "1", "--batch_size", "1", "--model_name", "v_48_020"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=os.environ)
        if os.path.exists(seqs_dir):
            fasta_files = [f for f in os.listdir(seqs_dir) if f.endswith(".fa")]
            if fasta_files:
                with open(os.path.join(seqs_dir, fasta_files[0]), "r") as f:
                    content = f.read()
                    match = re.search(r"score=([\d\.]+)", content)
                    if match:
                        score = float(match.group(1))
                        print(f"  [DEBUG] Scored {os.path.basename(pdb_path)}: {score}")
                        return score
        return None
    except Exception as e:
        print(f"  [ERROR] Subprocess crash: {e}")
        return None

def main():
    # Get Slurm Array ID
    set_num = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    
    input_csv = os.path.join(BASE_PATH, f"Test_Set_{set_num}_predictions.csv")
    output_csv = f"Test_Set_{set_num}_results.csv"
    
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    df['WT_LL'] = 0.0
    df['mutant_LL'] = 0.0
    df['WT_LLR'] = 0.0
    df['llr_score'] = 0.0

    temp_out = os.path.join(os.getcwd(), f"temp_mpnn_{os.getpid()}_set{set_num}")
    wt_cache = {}

    print(f"Processing Set {set_num}: {len(df)} rows...")

    for i, row in df.iterrows():
        fname = str(row['file']).strip()
        base_id = fname.split(".graphml")[0]
        suffix = fname.split(".graphml")[1] if ".graphml" in fname else ""

        # 1. WT SCORE - Uses WT_PDB_DIR
        if base_id not in wt_cache:
            wt_path = os.path.join(WT_PDB_DIR, f"{base_id}.pdb")
            # Fallback to MUT_DIR if not in WT_DIR
            if not os.path.exists(wt_path):
                wt_path = os.path.join(MUT_PDB_DIR, f"{base_id}.pdb")
            wt_cache[base_id] = get_mpnn_score(wt_path, temp_out)
        
        ll_wt = wt_cache[base_id]

        # 2. MUTANT SCORE - Uses MUT_PDB_DIR and the .pdb_suffix.pdb pattern
        if not suffix or "_" not in suffix:
            ll_mut = ll_wt
        else:
            mut_path = os.path.join(MUT_PDB_DIR, f"{base_id}.pdb{suffix}.pdb")
            ll_mut = get_mpnn_score(mut_path, temp_out)

        # 3. WRITE
        if ll_wt is not None and ll_mut is not None:
            llr = ll_wt - ll_mut
            df.loc[i, 'WT_LL'] = ll_wt
            df.loc[i, 'mutant_LL'] = ll_mut
            df.loc[i, 'WT_LLR'] = llr
            df.loc[i, 'llr_score'] = llr
        
        if i % 20 == 0:
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    shutil.rmtree(temp_out, ignore_errors=True)
    print(f"Task {set_num} Finished.")

if __name__ == "__main__":
    main()