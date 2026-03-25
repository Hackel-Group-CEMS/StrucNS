import os
import torch
import torch.nn as nn
import esm
import esm.inverse_folding
import pandas as pd
import csv
import warnings
import time

# --- 1. Environment and Paths ---
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TORCH_HOME"] = os.path.expanduser("~")

BASE_INPUT_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/feature_datasets/StrucNS_sets/Training/base_model/case1/predictions"
VARIANT_PDB_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
WT_PDB_DIR_1 = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
WT_PDB_DIR_2 = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs_WT"

INPUT_CSVS = ["Test_Set_1_predictions.csv", "Test_Set_2_predictions.csv", "Test_Set_3_predictions.csv"]

# --- 2. The ESM-IF1 Loader ---
def load_esm_if():
    print(f"Initializing ESM-IF1 on {DEVICE}...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(DEVICE)
    return model, alphabet

# --- 3. Helper Functions ---

def find_pdb_path(identifier: str, is_wt: bool = False) -> str:
    """Enhanced search logic to catch .wte and varied extensions."""
    if is_wt:
        base_name = identifier.replace(".graphml", "")
        # Covers standard, wte, and double-dot variations seen in your logs
        possible_names = [
            f"{base_name}.pdb", 
            f"{base_name}.pdb_wte.pdb", 
            f"{base_name}..pdb_wte.pdb",
            f"{identifier}.pdb", 
            f"{identifier}.pdb.pdb"
        ]
        for directory in [WT_PDB_DIR_1, WT_PDB_DIR_2]:
            for fname in possible_names:
                full_path = os.path.join(directory, fname)
                if os.path.exists(full_path): return full_path
    else:
        # Variant logic: Check .pdb.pdb and single .pdb
        base = identifier.replace(".graphml", ".pdb")
        name_options = [base + ".pdb", base, f"{identifier}.pdb"] 
        for name in name_options:
            full_path = os.path.join(VARIANT_PDB_DIR, name)
            if os.path.exists(full_path): return full_path
            
    raise FileNotFoundError(f"PDB not found for {identifier}")

def score_with_if1(model, alphabet, pdb_path):
    """Calculates log-likelihood. Note: removed 'device' keyword."""
    coords, native_seq = esm.inverse_folding.util.load_coords(pdb_path, chain="A")
    
    with torch.no_grad():
        # ESM-IF util handles device internally based on model.device
        ll, _ = esm.inverse_folding.util.score_sequence(
            model, alphabet, coords, native_seq
        )
    return ll

# --- 4. Main Processing ---

def main():
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if array_id >= len(INPUT_CSVS):
        return

    csv_filename = INPUT_CSVS[array_id]
    input_path = os.path.join(BASE_INPUT_DIR, csv_filename)
    output_path = f"scored_esmIF_{csv_filename}"

    model, alphabet = load_esm_if()
    df = pd.read_csv(input_path)
    
    processed = set()
    wt_score_cache = {} 
    
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            processed = set(existing_df['file'].unique())
            wt_score_cache = dict(zip(existing_df['WT'], existing_df['wt_IF_ll']))
            print(f"Resuming: {len(processed)} variants already done.")
        except: pass

    print(f"Starting {csv_filename}. Total rows: {len(df)}")
    start_time = time.time()

    with open(output_path, 'a', newline='') as f:
        fieldnames = ['file', 'WT', 'wt_IF_ll', 'mut_IF_ll', 'delta_IF_ll']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not os.path.exists(output_path) or os.stat(output_path).st_size == 0:
            writer.writeheader()

        for index, row in df.iterrows():
            var_id = row['file']
            wt_id = row['WT']

            if var_id in processed:
                continue

            try:
                # 1. Score WT
                if wt_id in wt_score_cache:
                    wt_ll = wt_score_cache[wt_id]
                else:
                    wt_path = find_pdb_path(wt_id, is_wt=True)
                    wt_ll = score_with_if1(model, alphabet, wt_path)
                    wt_score_cache[wt_id] = wt_ll

                # 2. Score Mutant
                mut_path = find_pdb_path(var_id, is_wt=False)
                mut_ll = score_with_if1(model, alphabet, mut_path)

                writer.writerow({
                    'file': var_id, 
                    'WT': wt_id, 
                    'wt_IF_ll': round(float(wt_ll), 4), 
                    'mut_IF_ll': round(float(mut_ll), 4), 
                    'delta_IF_ll': round(float(mut_ll - wt_ll), 4)
                })
                
                if index > 0 and index % 50 == 0:
                    f.flush()
                    elapsed = time.time() - start_time
                    per_row = elapsed / (index + 1)
                    remaining = (len(df) - index) * per_row / 3600
                    print(f"Row {index} | {var_id} | Est. Remaining: {remaining:.2f} hours")

            except FileNotFoundError as e:
                # Print once per missing file type to avoid log bloat
                if index % 100 == 0: print(f"  -> File issues at row {index}: {e}")
            except Exception as e:
                print(f"  -> Error at row {index} ({var_id}): {e}")

if __name__ == "__main__":
    main()