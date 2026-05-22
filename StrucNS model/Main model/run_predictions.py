import os
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
# Base path where case folders (case1, case2, etc.) are located
BASE_DIR = "/scratch.global/hackelb/mulli468/Tsuboyama_analysis/processing_data/feature_datasets/StrucNS_sets/Training/base_model"
NEW_TEST_SET_PATH = "StructureNS_features.csv"
OUTPUT_FILENAME = "predictions.csv"

# Load the new test set
print(f"Loading new test set from {NEW_TEST_SET_PATH}...")
df_new = pd.read_csv(NEW_TEST_SET_PATH)

# Identify feature columns (excluding metadata as per your training code)
metadata_cols = ['file', 'Family_Name', 'deltaG']
feature_cols = [c for c in df_new.columns if c not in metadata_cols]

# --- LOOP THROUGH CASES ---
for i in range(1, 7):
    case_name = f"case{i}"
    case_path = os.path.join(BASE_DIR, case_name)
    
    print(f"\n--- Processing {case_name} ---")
    
    # Paths to the saved assets within each case folder
    model_path = os.path.join(case_path, 'models', 'final_model.h5')
    scaler_path = os.path.join(case_path, 'models', 'final_scaler.gz')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"Warning: Model or Scaler not found for {case_name}. Skipping...")
        continue

    # 1. Load Model and Scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Preprocess the data
    X_new = df_new[feature_cols].values
    X_scaled = scaler.transform(X_new)

    # 3. Run Prediction
    # y_prob is the continuous score from the sigmoid activation (0 to 1)
    y_prob = model.predict(X_scaled, verbose=0).flatten()

    # 4. Prepare results dataframe
    results_df = pd.DataFrame({
        'file': df_new['file'],
        'predicted_score': y_prob
    })

    # 5. Create local output directory and save
    os.makedirs(case_name, exist_ok=True)
    save_path = os.path.join(case_name, OUTPUT_FILENAME)
    results_df.to_csv(save_path, index=False)
    
    print(f"Results saved to {save_path}")

print("\nAll predictions complete.")