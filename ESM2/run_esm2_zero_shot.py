import os
import torch
import esm
import pandas as pd
from Bio import SeqIO
import csv
from typing import Set, Dict, Tuple
import warnings

# Suppress PDB warnings from Biopython
warnings.filterwarnings("ignore")

# --- Configuration ---
# Input directory containing the 3 predictions CSVs
BASE_INPUT_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/feature_datasets/StrucNS_sets/Training/base_model/case1_n/predictions"

# Directories containing PDB files
VARIANT_PDB_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
WT_PDB_DIR_1 = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
WT_PDB_DIR_2 = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs_WT"

INPUT_CSVS = [
    "Test_Set_1_predictions.csv",
    "Test_Set_2_predictions.csv",
    "Test_Set_3_predictions.csv"
]

MODEL_NAME = "esm2_t33_650M_UR50D"

# --- 1. Model Setup ---
print(f"Loading ESM-2 model: {MODEL_NAME}...")
try:
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    print(f"Model running on {DEVICE}.")
except Exception as e:
    print(f"Error loading ESM-2 model: {e}")
    exit()

# --- 2. Helper Functions ---

def get_sequence_from_pdb(pdb_path: str) -> str:
    """Extracts amino acid sequence from PDB ATOM records."""
    try:
        for record in SeqIO.parse(pdb_path, "pdb-atom"):
            return str(record.seq).upper()
    except Exception as e:
        raise ValueError(f"Could not parse PDB {pdb_path}: {e}")
    raise ValueError(f"No sequence found in PDB {pdb_path}")

def find_wt_pdb(wt_identifier: str) -> Tuple[str, str]:
    """Locates the WT PDB file given the identifier."""
    base_name = wt_identifier.replace(".graphml", "")
    possible_names = [
        f"{base_name}.pdb",             
        f"{base_name}.pdb_wte.pdb",     
        f"{base_name}..pdb_wte.pdb"     
    ]
    search_dirs = [WT_PDB_DIR_1, WT_PDB_DIR_2]
    
    for directory in search_dirs:
        for fname in possible_names:
            full_path = os.path.join(directory, fname)
            if os.path.exists(full_path):
                return full_path, fname
                
    raise FileNotFoundError(f"Could not find WT PDB for {wt_identifier} in search directories.")

def calculate_log_likelihood(sequence: str, model: torch.nn.Module, alphabet, batch_converter, device: torch.device) -> float:
    """Calculates sequence pseudo-likelihood (MLM objective)."""
    if len(sequence) > 1022:
        print(f"Warning: Sequence length {len(sequence)} exceeds limit. Truncating.")
        sequence = sequence[:1022]

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    log_likelihood_sum = 0.0
    seq_len = len(sequence)
    
    with torch.no_grad():
        for i in range(seq_len):
            masked_tokens = batch_tokens.clone()
            wt_token_index = masked_tokens[0, i + 1].item()
            masked_tokens[0, i + 1] = alphabet.mask_idx 

            out = model(masked_tokens)
            log_probability = torch.log_softmax(out["logits"][0, i + 1], dim=-1)[wt_token_index]
            log_likelihood_sum += log_probability.item()
            
    return log_likelihood_sum

def append_to_csv(data_dict: Dict, output_csv_path: str):
    """Appends a single row to CSV immediately."""
    fieldnames = ['file', 'WT', 'variant_score_esm2', 'WT_score_Esm2', 'LLR', 'dg_mut', 'dg_WT', 'ddG']
    file_exists_and_not_empty = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0
    
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists_and_not_empty:
            writer.writeheader()
        writer.writerow(data_dict)

def get_processed_files(output_csv_path: str) -> Set[str]:
    """Reads output CSV to find which variants are already done."""
    if not os.path.exists(output_csv_path):
        return set()
    try:
        df = pd.read_csv(output_csv_path, usecols=['file'])
        return set(df['file'].unique())
    except Exception:
        return set()

def get_existing_wt_scores(output_csv_path: str) -> Dict[str, float]:
    """
    Reads the output CSV to retrieve WT scores calculated in previous runs/steps.
    Returns dict: {'WT_filename': score}
    """
    if not os.path.exists(output_csv_path):
        return {}
    try:
        # We read WT name and its score
        df = pd.read_csv(output_csv_path, usecols=['WT', 'WT_score_Esm2'])
        # Drop duplicates to keep unique WTs and convert to dict
        return pd.Series(df.WT_score_Esm2.values, index=df.WT).to_dict()
    except Exception:
        return {}

# --- 3. Main Processing Logic ---

def process_single_dataset(input_csv_name: str):
    input_path = os.path.join(BASE_INPUT_DIR, input_csv_name)
    output_csv_name = input_csv_name.replace(".csv", "_ESM2_scored.csv")
    output_path = output_csv_name  # Save in current directory

    print(f"\n=== Processing Dataset: {input_csv_name} ===")
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {os.path.abspath(output_path)}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to read input CSV {input_path}: {e}")
        return

    # 1. Load checkpoint (Skip variants already done)
    processed_variants = get_processed_files(output_path)
    print(f"Skipping {len(processed_variants)} already processed variants.")

    # 2. Load "Warm Start" WT scores (WTs calculated in previous runs)
    # Dictionary mapping WT_ID -> Score
    csv_wt_cache = get_existing_wt_scores(output_path)
    print(f"Loaded {len(csv_wt_cache)} WT scores from existing output file.")

    # 3. In-memory cache for current run
    # Key: WT_filename (pdb), Value: (sequence, log_likelihood)
    wt_cache: Dict[str, Tuple[str, float]] = {}

    total_rows = len(df)
    
    for index, row in df.iterrows():
        variant_file_id = row['file']
        wt_file_id = row['WT']
        
        if variant_file_id in processed_variants:
            continue

        print(f"Row {index+1}/{total_rows}: {variant_file_id}")

        try:
            # --- Step A: Get WT Sequence and Score (Optimized) ---
            wt_path, wt_fname = find_wt_pdb(wt_file_id)
            
            # Check 1: Is it in current memory? (Fastest)
            if wt_fname in wt_cache:
                wt_seq, wt_ll = wt_cache[wt_fname]
                
            else:
                # We always need the sequence for length checks
                wt_seq = get_sequence_from_pdb(wt_path)

                # Check 2: Was it in the CSV from a previous run?
                if wt_file_id in csv_wt_cache:
                    wt_ll = csv_wt_cache[wt_file_id]
                    # Add to memory cache so next lookup is even faster
                    wt_cache[wt_fname] = (wt_seq, wt_ll) 
                else:
                    # Check 3: Calculate fresh (Slowest)
                    wt_ll = calculate_log_likelihood(wt_seq, model, alphabet, batch_converter, DEVICE)
                    wt_cache[wt_fname] = (wt_seq, wt_ll)

            # --- Step B: Get Variant Sequence and Score ---
            variant_pdb_name = variant_file_id.replace(".graphml", ".pdb") + ".pdb"
            variant_path = os.path.join(VARIANT_PDB_DIR, variant_pdb_name)

            if not os.path.exists(variant_path):
                print(f"  Error: Variant PDB not found at {variant_path}")
                continue

            variant_seq = get_sequence_from_pdb(variant_path)
            
            # Sanity Check
            if len(variant_seq) != len(wt_seq):
                print(f"  Warning: Seq length mismatch (WT: {len(wt_seq)}, Var: {len(variant_seq)}). Proceeding.")

            variant_ll = calculate_log_likelihood(variant_seq, model, alphabet, batch_converter, DEVICE)

            # --- Step C: Calculate LLR and Save ---
            llr = variant_ll - wt_ll

            result_entry = {
                'file': variant_file_id,
                'WT': wt_file_id,
                'variant_score_esm2': variant_ll,
                'WT_score_Esm2': wt_ll,
                'LLR': llr,
                'dg_mut': row.get('dg_mut', ''),
                'dg_WT': row.get('dg_wt', ''),
                'ddG': row.get('ddG', '')
            }
            
            append_to_csv(result_entry, output_path)

        except Exception as e:
            print(f"  Failed to process {variant_file_id}: {e}")
            continue

if __name__ == "__main__":
    # Get the array ID from SLURM (default to 0 for local testing)
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    
    # Map the ID to your specific files
    if array_id < len(INPUT_CSVS):
        target_csv = INPUT_CSVS[array_id]
        process_single_dataset(target_csv)
    else:
        print(f"Array ID {array_id} out of range for INPUT_CSVS.")