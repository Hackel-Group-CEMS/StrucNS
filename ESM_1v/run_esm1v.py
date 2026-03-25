import os
import torch
import torch.nn as nn
import esm
import pandas as pd
from Bio import SeqIO
import csv
from argparse import Namespace
from esm.model.esm1 import ProteinBertModel
from typing import Tuple
import warnings

# --- 1. Environment and Paths ---
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TORCH_HOME"] = os.path.expanduser("~")

# Root Directories (Matching your ESM-2 setup)
BASE_INPUT_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/feature_datasets/StrucNS_sets/Training/base_model/case1/predictions"
VARIANT_PDB_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
WT_PDB_DIR_1 = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs"
WT_PDB_DIR_2 = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/omegafold_pdbs_WT"

INPUT_CSVS = ["Test_Set_1_predictions.csv", "Test_Set_2_predictions.csv", "Test_Set_3_predictions.csv"]

# --- 2. The Universal Loader ---
def load_esm1v_ensemble():
    print(f"Initializing ESM-1v Ensemble on {DEVICE}...")
    alphabet = esm.Alphabet.from_architecture("roberta_large")
    args = Namespace(
        layers=33, embed_dim=1280, logit_bias=True, ffn_embed_dim=5120,
        attention_heads=20, token_dropout=True, max_positions=1024,
        padding_idx=alphabet.padding_idx, dropout=0.1, attention_dropout=0.1,
        arch='roberta_large'
    )

    ensemble = []
    ckpt_dir = os.path.join(os.environ["TORCH_HOME"], ".cache/torch/hub/checkpoints")
    for i in range(1, 6):
        model_name = f"esm1v_t33_650M_UR90S_{i}"
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
        if not os.path.exists(ckpt_path): continue

        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        sd = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        clean_sd = {k.replace("sentence_encoder.", "").replace("encoder.", "").replace("esm.", "").replace("decoder.", ""): v for k, v in sd.items()}
        model = ProteinBertModel(args, alphabet)
        
        if 'lm_head.weight' in clean_sd:
            target_vocab_size = clean_sd['lm_head.weight'].shape[0]
            if model.lm_head.weight.shape[0] != target_vocab_size:
                model.lm_head.bias = nn.Parameter(torch.zeros(target_vocab_size))
                model.lm_head.weight = nn.Parameter(torch.zeros(target_vocab_size, 1280))
        
        model.load_state_dict(clean_sd, strict=False)
        model.eval().to(DEVICE)
        ensemble.append(model)
        print(f"  -> Loaded {model_name}")
    return ensemble, alphabet

# --- 3. Helper Functions (Ported from your ESM-2 code) ---

def find_wt_pdb(wt_identifier: str) -> Tuple[str, str]:
    """EXACT logic from your ESM-2 script to find WT files."""
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
    raise FileNotFoundError(f"WT PDB not found for {wt_identifier}")

def get_sequence_from_pdb(pdb_path: str) -> str:
    """EXACT logic from your ESM-2 script to extract sequence."""
    for record in SeqIO.parse(pdb_path, "pdb-atom"):
        return str(record.seq).upper()
    raise ValueError(f"No sequence in {pdb_path}")

def score_variant_ensemble(wt_seq, mut_seq, models, alphabet):
    """WT Marginal scoring across the ensemble."""
    diffs = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
    if not diffs: return 0.0

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("protein", wt_seq)])
    tokens = tokens.to(DEVICE)

    total_llr = 0.0
    with torch.no_grad():
        for model in models:
            logits = model(tokens)["logits"][0]
            log_probs = torch.log_softmax(logits, dim=-1)
            for pos in diffs:
                wt_idx = alphabet.get_idx(wt_seq[pos])
                mut_idx = alphabet.get_idx(mut_seq[pos])
                total_llr += (log_probs[pos + 1, mut_idx] - log_probs[pos + 1, wt_idx]).item()
    return total_llr / len(models)

# --- 4. Main Loop ---

def main():
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if array_id >= len(INPUT_CSVS): return

    csv_filename = INPUT_CSVS[array_id]
    input_path = os.path.join(BASE_INPUT_DIR, csv_filename)
    output_path = f"scored_esm1v_{csv_filename}"

    models, alphabet = load_esm1v_ensemble()
    df = pd.read_csv(input_path)
    
    # Processed check
    processed = set()
    if os.path.exists(output_path):
        processed = set(pd.read_csv(output_path)['file'].unique())

    print(f"Starting {csv_filename}. Total rows: {len(df)}")

    with open(output_path, 'a', newline='') as f:
        fieldnames = ['file', 'WT', 'ESM1v_LLR']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if os.stat(output_path).st_size == 0 if os.path.exists(output_path) else True:
            writer.writeheader()

        for index, row in df.iterrows():
            var_id = row['file']
            wt_id = row['WT']

            if var_id in processed: continue

            try:
                # 1. Find WT
                wt_path, _ = find_wt_pdb(wt_id)
                wt_seq = get_sequence_from_pdb(wt_path)

                # 2. Find Variant (Using ESM-2 logic: .graphml -> .pdb.pdb)
                variant_pdb_name = var_id.replace(".graphml", ".pdb") + ".pdb"
                variant_path = os.path.join(VARIANT_PDB_DIR, variant_pdb_name)
                
                if not os.path.exists(variant_path):
                    # Fallback check for standard .pdb extension
                    variant_path = os.path.join(VARIANT_PDB_DIR, var_id.replace(".graphml", ".pdb"))

                mut_seq = get_sequence_from_pdb(variant_path)

                if len(wt_seq) == len(mut_seq):
                    llr = score_variant_ensemble(wt_seq, mut_seq, models, alphabet)
                    writer.writerow({'file': var_id, 'WT': wt_id, 'ESM1v_LLR': llr})
                    f.flush()
                
                if index % 100 == 0:
                    print(f"  -> Row {index} done")

            except Exception as e:
                if index % 100 == 0:
                    print(f"  -> Skipping {var_id}: {e}")

if __name__ == "__main__":
    main()