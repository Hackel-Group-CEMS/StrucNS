import os
import torch
import esm
import pandas as pd
from Bio import SeqIO
import csv 
from typing import Set, Dict

# --- Configuration ---
# NOTE: The batch logic will process ALL *.fasta files in this directory.
VARIANT_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/fasta_files_dataset1_only"
WT_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/fasta_files_WT"
OUTPUT_CSV = "esm2_one_shot_batch_scores.csv"
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
    print(f"Error loading ESM-2 model. Ensure 'fair-esm' and 'torch' are installed: {e}")
    exit()

# --- 2. Helper Functions ---

def read_fasta_sequence(file_path: str):
    """Reads the header description and sequence from the first record of a FASTA file."""
    try:
        record = next(SeqIO.parse(file_path, "fasta"))
        return record.description.strip(), record.seq.__str__().upper()
    except StopIteration:
        raise ValueError(f"FASTA file {file_path} contains no sequences.")

def calculate_log_likelihood(sequence: str, model: torch.nn.Module, alphabet, batch_converter, device: torch.device) -> float:
    """Calculates the sequence pseudo-likelihood using the MLM objective."""
    
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    log_likelihood_sum = 0.0
    seq_len = len(sequence)
    
    with torch.no_grad():
        for i in range(seq_len):
            masked_tokens = batch_tokens.clone()
            
            # Index i+1 because of the <cls> token at index 0
            wt_token_index = masked_tokens[0, i + 1].item()
            masked_tokens[0, i + 1] = alphabet.mask_idx # Apply mask

            out = model(masked_tokens)
            logits = out["logits"]

            # Log-softmax converts logits to log probabilities
            log_probability = torch.log_softmax(logits[0, i + 1], dim=-1)[wt_token_index]
            log_likelihood_sum += log_probability.item()
            
    return log_likelihood_sum

# --- 3. Checkpointing and Batch Processing Logic ---

def get_processed_variants(output_csv_path: str) -> Set[str]:
    """Reads the output CSV to get a set of already processed variant filenames for checkpointing."""
    if not os.path.exists(output_csv_path):
        return set()
    try:
        # Only read the 'Variant_File' column
        df = pd.read_csv(output_csv_path, usecols=['Variant_File'])
        return set(df['Variant_File'].unique())
    except Exception as e:
        print(f"Warning: Could not read existing CSV for checkpointing ({e}). Assuming no existing data.")
        # If file exists but is empty or corrupt, we start writing from the start.
        return set()

def append_to_csv(data_dict: Dict, output_csv_path: str):
    """Appends a single dictionary entry to the CSV file efficiently."""
    fieldnames = ['Variant_File', 'Variant_Header', 'WT_Name', 'LL_WT', 'LL_Variant', 'ESM2_LLR']
    
    # Check if file exists and is empty to determine if header needs to be written
    file_exists_and_not_empty = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0
    
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file does not exist or is empty
        if not file_exists_and_not_empty:
            writer.writeheader()
        
        writer.writerow(data_dict)


def process_all_variants_with_checkpoint(variant_dir: str, wt_dir: str, output_csv_base: str):
    """
    Processes all FASTA files with checkpointing, LL_WT caching, and appends results to CSV.
    """
    
    # --- Checkpointing Step ---
    processed_variants = get_processed_variants(output_csv_base)
    print(f"Found {len(processed_variants)} variants already processed in {output_csv_base}.")
    
    # CORRECTED SYNTAX: Iterates over files and filters for '.fasta'
    all_variant_files = [f for f in os.listdir(variant_dir) if f.endswith(".fasta")]
    
    # Filter out files that have already been processed and sort for consistent restarts
    variant_files_to_process = sorted([
        f for f in all_variant_files if f not in processed_variants
    ])
    
    if not variant_files_to_process:
        print("All variants are already processed. Job finished.")
        return

    print(f"Starting batch process for {len(variant_files_to_process)} variants.")
    
    wt_ll_cache: Dict[str, float] = {} 
    
    # Start counter to track progress through the files *to be processed*
    for i, variant_filename in enumerate(variant_files_to_process):
        print(f"\n--- Processing {i+1}/{len(variant_files_to_process)}: {variant_filename} ---")
        
        variant_filepath = os.path.join(variant_dir, variant_filename)
        
        try:
            # --- Steps A & B: Read Sequences and Validate ---
            variant_header, variant_seq = read_fasta_sequence(variant_filepath)
            
            # 1. Parse WT name from header (e.g., 'xyz.pdb_abc' -> 'xyz')
            wt_name = variant_header.split('.pdb')[0]
            wt_filename = f"{wt_name}.fasta"
            wt_filepath = os.path.join(wt_dir, wt_filename)
            
            if not os.path.exists(wt_filepath):
                 print(f"Skipping: WT file not found at {wt_filepath}.")
                 continue
                 
            _, wt_seq = read_fasta_sequence(wt_filepath)
                
            if len(wt_seq) != len(variant_seq):
                print("Skipping: WT and Variant sequences are different lengths.")
                continue

            # --- Step C: Calculate Log-Likelihoods ---
            
            # 2. Get LL_WT (Use cache or calculate)
            if wt_name in wt_ll_cache:
                wt_ll = wt_ll_cache[wt_name]
                print(f"  Retrieved LL_WT for {wt_name} from cache.")
            else:
                wt_ll = calculate_log_likelihood(wt_seq, model, alphabet, batch_converter, DEVICE)
                wt_ll_cache[wt_name] = wt_ll # Store in cache
                print(f"  Calculated and cached LL_WT for {wt_name}.")
            
            # 3. Calculate LL_Variant
            variant_ll = calculate_log_likelihood(variant_seq, model, alphabet, batch_converter, DEVICE)

            # 4. Calculate LLR
            llr_score = variant_ll - wt_ll

            # --- Step D: Save Result Immediately ---
            result_data = {
                'Variant_File': variant_filename,
                'Variant_Header': variant_header,
                'WT_Name': wt_name,
                'LL_WT': wt_ll,
                'LL_Variant': variant_ll,
                'ESM2_LLR': llr_score
            }
            append_to_csv(result_data, output_csv_base)
            print(f"  LLR: {llr_score:.4f} | Result saved and checkpoint updated.")

        except Exception as e:
            # Log the error but continue to the next file
            print(f"Critical error processing {variant_filename}: {e}. Continuing to next file.")
            continue


if __name__ == "__main__":
    process_all_variants_with_checkpoint(VARIANT_DIR, WT_DIR, OUTPUT_CSV)