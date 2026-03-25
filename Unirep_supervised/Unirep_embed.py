import pandas as pd
import os
import numpy as np
from jax_unirep.layers import mLSTM
from jax_unirep.utils import get_embeddings, load_params
from jax import device_put

# Load UniRep model
params = load_params()[1]
_, apply_fun = mLSTM(output_dim=1900)

def get_unirep(seq):
    x = get_embeddings([seq])[0]  # shape (T, 10)
    x = device_put(x)
    h_final, c_final, h = apply_fun(params, x)
    return np.mean(np.array(h), axis=0)

# Setup file paths
pkl_name = 'unirep_input_data'
input_csv = pkl_name + '.csv'
output_csv = pkl_name + '_embedded.csv'

# Create output file with header if it doesn't exist
if not os.path.exists(output_csv):
    with open(output_csv, 'w') as f:
        header = ['name', 'sequence'] + [f'full_unirep_{i}' for i in range(1900)]
        f.write(','.join(header) + '\n')

# Read existing processed names
processed_names = set()
if os.path.exists(output_csv):
    with open(output_csv) as f:
        next(f)  # skip header
        for line in f:
            processed_names.add(line.split(',')[0])

# Process input row by row
with pd.read_csv(input_csv, chunksize=1) as reader:
    for chunk in reader:
        row = chunk.iloc[0]
        name = row['name']
        seq = row['sequence']

        if name in processed_names:
            continue

        try:
            emb = get_unirep(seq)
            emb_str = ','.join(map(str, emb))
            with open(output_csv, 'a') as f:
                f.write(f"{name},{seq},{emb_str}\n")
            print(f"Processed: {name}")
        except Exception as e:
            print(f"Error processing {name}: {e}")

