import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Directories
os.makedirs('predictions', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# --------------------------------------------------------------------------------
# 1. LOAD DATA & RECREATE SPLITS
# --------------------------------------------------------------------------------
print("Loading datasets...")
try:
    df1 = pd.read_csv('set1_features.csv')
    df2 = pd.read_csv('set2_features.csv')
    df3 = pd.read_csv('set3_features.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- Recreate Test Set 1 Split ---
# We need to recreate the exact split to ensure we are testing on the correct "Test Set 1"
# and not mixing training data into the evaluation.
GLOBAL_SEED = 42 
train_list, test1_list = [], []

for _, group in df1.groupby('Family_Name'):
    if len(group) < 2:
        train_list.append(group)
    else:
        grp_shuffled = group.sample(frac=1, random_state=GLOBAL_SEED)
        split_point = int(len(grp_shuffled) * 0.8)
        
        train_list.append(grp_shuffled.iloc[:split_point])
        test1_list.append(grp_shuffled.iloc[split_point:])

test1_df = pd.concat(test1_list).reset_index(drop=True)

# Define the dictionary of Test Sets to iterate over
test_sets = {
    "Test_Set_1": test1_df, 
    "Test_Set_2": df2, 
    "Test_Set_3": df3
}

# --------------------------------------------------------------------------------
# 2. GLOBAL PREDICTIONS (Lookup Table Creation)
# --------------------------------------------------------------------------------
# To calculate ddG and LLR, we need the scores of the Wild Types (WT).
# A WT required for a mutant in Test Set 1 might exist in the Train Set part of Set 1.
# Therefore, we predict on the ENTIRE available data to build a master lookup table.

print("Building Master Lookup Table for WTs...")

# Combine all dataframes
master_df = pd.concat([df1, df2, df3]).drop_duplicates(subset='file')

# Identify Feature Columns (exclude metadata)
metadata_cols = ['file', 'Family_Name', 'deltaG']
feature_cols = [c for c in df1.columns if c not in metadata_cols]

# Load Resources
print("Loading model and scaler...")
try:
    final_scaler = joblib.load('models/final_scaler.gz')
    final_model = tf.keras.models.load_model('models/final_model.h5')
except Exception as e:
    print(f"Error loading resources: {e}\nPlease run the training script first.")
    exit(1)

# Scale Features
X_master = final_scaler.transform(master_df[feature_cols].values)

# Predict Probabilities (P(Stable))
# The model outputs a single sigmoid score: Probability of being Stable (High deltaG)
master_probs = final_model.predict(X_master, verbose=1).flatten()
master_df['prob_stable'] = master_probs

# Create a Dictionary for O(1) Lookup: file -> {prob, deltaG}
lookup_dict = master_df.set_index('file')[['prob_stable', 'deltaG']].to_dict('index')

# --------------------------------------------------------------------------------
# 3. GENERATE PREDICTION CSVs (Inference)
# --------------------------------------------------------------------------------
print("\n--- Generating Prediction CSVs ---")

def get_wt_name(filename):
    """
    Revised Heuristic:
    Identify WT by splitting explicitly at the '.graphml' extension.
    It checks for two potential WT naming conventions in the lookup_dict:
    1. Standard: 'WT.graphml'
    2. With suffix: 'WT.graphml_wte'
    """
    # Define the extension we use as the delimiter
    extension = ".graphml"
    
    if extension in filename:
        # partition returns a tuple: (part_before, separator, part_after)
        # Splits at the first occurrence of the separator
        head, sep, tail = filename.partition(extension)
        
        # 1. Construct the base WT filename (e.g., "1BNZ.graphml")
        base_candidate = head + sep 
        
        # Check if the base version exists
        if base_candidate in lookup_dict:
            return base_candidate

        # 2. Construct the _wte version (e.g., "1BNZ.graphml_wte")
        wte_candidate = base_candidate + "_wte"

        # Check if the _wte version exists
        if wte_candidate in lookup_dict:
            return wte_candidate

    return None

prediction_files = {} # Store paths for next step

for set_name, df in test_sets.items():
    print(f"Processing {set_name}...")
    results = []
    
    for _, row in df.iterrows():
        mutant_file = row['file']
        
        # 1. Identify WT
        wt_file = get_wt_name(mutant_file)
        
        # If we can't find a WT (or the row IS a WT), we skip or handle accordingly.
        # Here we only process rows that are valid mutants with found WTs.
        if wt_file and wt_file in lookup_dict:
            
            # 2. Get Data
            p_mut = lookup_dict[mutant_file]['prob_stable']
            p_wt = lookup_dict[wt_file]['prob_stable']
            
            dg_mut = row['deltaG']
            dg_wt = lookup_dict[wt_file]['deltaG']
            
            # 3. Calculate Metrics
            # Log Likelihood (Natural Log)
            # Add epsilon to avoid log(0)
            epsilon = 1e-10
            ll_mut = np.log(p_mut + epsilon)
            ll_wt = np.log(p_wt + epsilon)
            
            # Log Likelihood Ratio (LLR)
            # Positive LLR implies Mutant is MORE probable to be stable than WT
            llr = ll_mut - ll_wt
            
            # Experimental ddG (delta-delta G)
            # deltaG is stability. ddG = Mut - WT.
            # Positive ddG implies Mutant is MORE stable than WT.
            ddg_exp = dg_mut - dg_wt
            
            results.append({
                'file': mutant_file,
                'WT': wt_file,
                'LL_mutant': ll_mut,
                'LL_WT': ll_wt,
                'LLR': llr,
                'dg_mut': dg_mut,
                'dg_wt': dg_wt,
                'ddG': ddg_exp
            })
            
    # Save to CSV
    results_df = pd.DataFrame(results)
    save_path = f'predictions/{set_name}_predictions.csv'
    results_df.to_csv(save_path, index=False)
    prediction_files[set_name] = save_path
    print(f"Saved {len(results_df)} predictions to {save_path}")

# --------------------------------------------------------------------------------
# 4. EVALUATION & METRICS
# --------------------------------------------------------------------------------
print("\n--- Evaluating Predictions (Binary Classification of Stability Change) ---")

metrics_summary = []

for set_name, csv_path in prediction_files.items():
    if not os.path.exists(csv_path):
        continue
        
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print(f"Warning: {set_name} yielded no valid mutant-WT pairs.")
        continue

    # Define Ground Truth and Predictions
    # Stable (Positive Class) = Value > 0
    # Unstable (Negative Class) = Value <= 0
    
    y_true = (df['ddG'] > 0).astype(int)
    y_pred = (df['LLR'] > 0).astype(int)
    
    # Calculate Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # Recall of Positive (Stable)
    neg_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0) # Recall of Negative (Unstable)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Format Output
    report = (
        f"=== {set_name} ===\n"
        f"Total Pairs Evaluated: {len(df)}\n"
        f"Confusion Matrix (TN, FP, FN, TP):\n{cm}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall (Stable/Pos): {recall:.4f}\n"
        f"Recall (Unstable/Neg): {neg_recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"--------------------------------------------------\n"
    )
    
    print(report)
    metrics_summary.append(report)

# Save Evaluation to Text File
with open('logs/final_stability_metrics.txt', 'w') as f:
    f.write("Stability Prediction Evaluation (Based on LLR vs ddG)\n")
    f.write("Positive Class: Stable (ddG > 0, LLR > 0)\n")
    f.write("Negative Class: Unstable (ddG <= 0, LLR <= 0)\n\n")
    for item in metrics_summary:
        f.write(item + "\n")

print("Evaluation complete. Metrics saved to logs/final_stability_metrics.txt")