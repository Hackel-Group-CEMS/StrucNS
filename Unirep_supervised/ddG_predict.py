import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

os.makedirs('predictions', exist_ok=True)
os.makedirs('logs', exist_ok=True)

GLOBAL_SEED = 42

# --------------------------------------------------------------------------------
# 1. LOAD DATA & RECREATE SPLITS
# --------------------------------------------------------------------------------
print("Loading datasets...")

def force_numeric(df, exclude_cols):
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

try:
    df1 = pd.read_csv('set1_embeddings_with_score.csv', low_memory=False)
    df2 = pd.read_csv('set2_embeddings_with_score.csv', low_memory=False)
    df3 = pd.read_csv('set3_embeddings_with_score.csv', low_memory=False)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

# UPDATED: 'name' is the identifier column, not 'file'
metadata_cols = ['name', 'file', 'Family_Name', 'deltaG', 'sequence', 'target'] 
df1 = force_numeric(df1, metadata_cols)
df2 = force_numeric(df2, metadata_cols)
df3 = force_numeric(df3, metadata_cols)

feature_cols = [c for c in df1.columns if c not in metadata_cols]
print(f"Detected {len(feature_cols)} embedding features.")

# --- Recreate Test Set 1 Split ---
if 'Family_Name' in df1.columns:
    print("Family_Name found. Recreating family-aware split for Test Set 1...")
    test1_list = []
    for _, group in df1.groupby('Family_Name'):
        if len(group) >= 2:
            grp_shuffled = group.sample(frac=1, random_state=GLOBAL_SEED)
            split_point = int(len(grp_shuffled) * 0.8)
            test1_list.append(grp_shuffled.iloc[split_point:])
    
    test1_df = pd.concat(test1_list).reset_index(drop=True)
else:
    print("Family_Name NOT found. Recreating random split...")
    _, test1_df = train_test_split(df1, test_size=0.2, random_state=GLOBAL_SEED, shuffle=True)

test_sets = {
    "Test_Set_1": test1_df, 
    "Test_Set_2": df2, 
    "Test_Set_3": df3
}

# --------------------------------------------------------------------------------
# 2. LOAD RESOURCES
# --------------------------------------------------------------------------------
print("\nLoading Preprocessing Pipeline & Model...")
try:
    final_imputer = joblib.load('models/final_imputer.pkl')
    final_scaler = joblib.load('models/final_scaler.pkl')
    final_pca = joblib.load('models/final_pca.pkl')
    final_model = tf.keras.models.load_model('models/final_model.h5')
    print("Resources loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    exit(1)

# --------------------------------------------------------------------------------
# 3. GLOBAL PREDICTIONS (Lookup Table Creation)
# --------------------------------------------------------------------------------
print("Building Master Lookup Table for WTs...")

# Combine all dataframes
# UPDATED: Drop duplicates based on 'name'
master_df = pd.concat([df1, df2, df3]).drop_duplicates(subset='name')

# Apply Pipeline
X_raw = master_df[feature_cols].values
X_imp = final_imputer.transform(X_raw)
X_sc = final_scaler.transform(X_imp)
X_pca = final_pca.transform(X_sc)

# Predict
master_probs = final_model.predict(X_pca, verbose=1).flatten()
master_df['prob_stable'] = master_probs

# UPDATED: Create Dictionary using 'name' as key
lookup_dict = master_df.set_index('name')[['prob_stable', 'deltaG']].to_dict('index')

# --------------------------------------------------------------------------------
# 4. GENERATE PREDICTION CSVs
# --------------------------------------------------------------------------------
print("\n--- Generating Prediction CSVs ---")

def get_wt_name(filename):
    """
    Robustly finds WT name from mutant filename.
    Logic: Looks for .pdb or .graphml and splits there.
    Example: '1abc.pdb_mutation' -> '1abc.pdb'
    """
    for ext in ['.pdb', '.graphml']:
        if ext in filename:
            # Split at the extension
            base = filename.split(ext)[0] + ext
            
            # Check if this base exists in our data
            if base in lookup_dict:
                return base
            
            # Sometimes there might be a simple suffix check
            if base == filename:
                return None # It IS the wild type, so it has no parent
                
    return None

prediction_files = {}

for set_name, df in test_sets.items():
    print(f"Processing {set_name}...")
    results = []
    
    for _, row in df.iterrows():
        # UPDATED: Use 'name' to identify the row
        mutant_name = row['name']
        wt_name = get_wt_name(mutant_name)
        
        # Only process if we found a valid WT parent in our data
        if wt_name and wt_name in lookup_dict:
            
            p_mut = lookup_dict[mutant_name]['prob_stable']
            p_wt = lookup_dict[wt_name]['prob_stable']
            
            dg_mut = row['deltaG']
            dg_wt = lookup_dict[wt_name]['deltaG']
            
            # Metrics
            epsilon = 1e-10
            ll_mut = np.log(p_mut + epsilon)
            ll_wt = np.log(p_wt + epsilon)
            
            # LLR > 0 => Mutant is more stable (predicted)
            llr = ll_mut - ll_wt
            
            # ddG > 0 => Mutant is more stable (experimental)
            ddg_exp = dg_mut - dg_wt
            
            results.append({
                'file': mutant_name, # Renaming 'name' back to 'file' for output csv consistency
                'WT': wt_name,
                'LL_mutant': ll_mut,
                'LL_WT': ll_wt,
                'LLR': llr,
                'ddG': ddg_exp
            })
            
    results_df = pd.DataFrame(results)
    save_path = f'predictions/{set_name}_predictions.csv'
    results_df.to_csv(save_path, index=False)
    prediction_files[set_name] = save_path
    print(f"Saved {len(results_df)} predictions to {save_path}")

# --------------------------------------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------------------------------------
print("\n--- Evaluating Predictions ---")

metrics_summary = []

for set_name, csv_path in prediction_files.items():
    if not os.path.exists(csv_path):
        continue
        
    df_res = pd.read_csv(csv_path)
    
    if df_res.empty:
        print(f"Warning: {set_name} yielded no valid mutant-WT pairs.")
        continue

    # Ground Truth: Stable if ddG > 0
    y_true = (df_res['ddG'] > 0).astype(int)
    
    # Prediction: Stable if LLR > 0
    y_pred = (df_res['LLR'] > 0).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)      
    neg_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    report = (
        f"=== {set_name} ===\n"
        f"Total Pairs Evaluated: {len(df_res)}\n"
        f"Confusion Matrix (TN, FP, FN, TP):\n{cm}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall (Stable/Pos): {recall:.4f}\n"
        f"Recall (Unstable/Neg): {neg_recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"--------------------------------------------------\n"
    )
    
    print(report)
    metrics_summary.append(report)

with open('logs/final_stability_metrics.txt', 'w') as f:
    f.write("Stability Prediction Evaluation (Embedding Model)\n")
    f.write("Positive Class: Stable (ddG > 0, LLR > 0)\n")
    f.write("Negative Class: Unstable (ddG <= 0, LLR <= 0)\n\n")
    for item in metrics_summary:
        f.write(item + "\n")

print("Evaluation complete. Metrics saved to logs/final_stability_metrics.txt")