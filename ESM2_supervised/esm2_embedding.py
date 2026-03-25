import os
import torch
import esm
import pandas as pd
from Bio import SeqIO

# === Configuration ===
input_dir = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/fasta_files_dataset1_only"
output_csv = "esm2_embeddings.csv"

# === Load ESM-2 model ===
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()

# === Set up output CSV ===
columns = ['name'] + [f"emb_{i}" for i in range(1280)]
if not os.path.exists(output_csv):
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

# === Load existing names to skip processed ===
try:
    existing = pd.read_csv(output_csv, usecols=["name"])
    processed_names = set(existing["name"].tolist())
except Exception:
    processed_names = set()

# === Process FASTA files ===
fasta_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".fasta")])

for fasta_file in fasta_files:
    path = os.path.join(input_dir, fasta_file)

    try:
        records = list(SeqIO.parse(path, "fasta"))
        if not records:
            print(f"⚠️ Skipping {fasta_file} (empty).")
            continue

        data = []
        seen_seqs = set()
        for rec in records:
            seq_id = rec.id
            seq = str(rec.seq)

            # Skip if already processed or duplicate sequence
            if seq_id in processed_names or seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            data.append((seq_id, seq))

        if not data:
            print(f"⚠️ Skipping {fasta_file} (no new unique sequences).")
            continue

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_reps = results["representations"][33]
        rows = []

        for i, (_, seq) in enumerate(data):
            seq_len = len(seq)
            rep = token_reps[i, 1:seq_len + 1].mean(0).cpu().numpy()
            row = [data[i][0]] + rep.tolist()  # use sequence ID as name
            rows.append(row)
            processed_names.add(data[i][0])  # update processed set live

        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"✅ Processed {fasta_file} with {len(rows)} new sequence(s).")

    except Exception as e:
        print(f"❌ Error processing {fasta_file}: {e}")
