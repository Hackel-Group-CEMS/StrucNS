import os
import json
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
import csv
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# --------------------------------------------------------------------------------
# CONFIGURATION & SEEDS
# --------------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Ensure output dirs exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

print("--- RECOVERY MODE STARTED ---")

# --------------------------------------------------------------------------------
# 1. LOAD DATA 
# --------------------------------------------------------------------------------
print("Loading datasets...")
df1 = pd.read_csv('set1_embeddings_with_score.csv', low_memory=False)
df2 = pd.read_csv('set2_embeddings_with_score.csv', low_memory=False)
df3 = pd.read_csv('set3_embeddings_with_score.csv', low_memory=False)

def force_numeric(df, exclude_cols):
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

metadata_cols = ['name', 'file', 'Family_Name', 'deltaG', 'sequence', 'target'] 
df1 = force_numeric(df1, metadata_cols)
df2 = force_numeric(df2, metadata_cols)
df3 = force_numeric(df3, metadata_cols)

test_sets = {
    "Test Set 2": df2, 
    "Test Set 3": df3
}

feature_cols = [c for c in df1.columns if c not in metadata_cols]
print(f"Detected {len(feature_cols)} embedding features.")

# --------------------------------------------------------------------------------
# 2. RECREATE TRAIN/TEST SPLIT
# --------------------------------------------------------------------------------
# We must recreate the exact same split to ensure validity
if 'Family_Name' in df1.columns:
    print("Family_Name found. Recreating family-aware split...")
    train_list, test1_list = [], []
    for _, group in df1.groupby('Family_Name'):
        if len(group) < 2:
            train_list.append(group)
        else:
            grp_shuffled = group.sample(frac=1, random_state=GLOBAL_SEED)
            split_point = int(len(grp_shuffled) * 0.8)
            train_list.append(grp_shuffled.iloc[:split_point])
            test1_list.append(grp_shuffled.iloc[split_point:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    test1_df = pd.concat(test1_list).reset_index(drop=True)
else:
    print("Family_Name NOT found. Recreating random 80/20 split...")
    train_df, test1_df = train_test_split(df1, test_size=0.2, random_state=GLOBAL_SEED, shuffle=True)

test_sets["Test Set 1"] = test1_df

# --------------------------------------------------------------------------------
# 3. RECOVER BEST PARAMS FROM OPTUNA DB
# --------------------------------------------------------------------------------
storage_url = "sqlite:///logs/optuna_study.db"
print(f"Connecting to database at {storage_url}...")

try:
    study = optuna.load_study(study_name="esm_optimization_case1", storage=storage_url)
    best_params = study.best_params
    best_value = study.best_value
    print(f"\nSUCCESS! Found best trial in DB.")
    print(f"Best F1 from optimization: {best_value:.4f}")
    print(f"Best Params: {best_params}")
except Exception as e:
    print(f"\nERROR: Could not load study from DB. ({e})")
    print("Checking for 'logs/best_trial.json' as fallback...")
    if os.path.exists('logs/best_trial.json'):
        with open('logs/best_trial.json', 'r') as f:
            best_params = json.load(f)
        print("Loaded params from JSON file.")
    else:
        raise FileNotFoundError("Could not find Optuna DB or JSON backup. Cannot proceed.")

# --------------------------------------------------------------------------------
# 4. PREPARE DATA FOR FINAL TRAINING (Impute -> Scale -> PCA)
# --------------------------------------------------------------------------------
print("\nPreparing final training data...")

# Set seeds from best params for reproducibility
final_seed = best_params.get('random_seed', 42)
np.random.seed(final_seed)
tf.random.set_seed(final_seed)
random.seed(final_seed)

# Internal Train/Val split for the final model training
if 'Family_Name' in train_df.columns:
    groups = [group for _, group in train_df.groupby('Family_Name')]
    random.shuffle(groups)
    split_idx = int(len(groups) * 0.8)
    final_train_grp = pd.concat(groups[:split_idx])
    final_val_grp = pd.concat(groups[split_idx:])
    
    X_tr_raw = final_train_grp[feature_cols].values
    y_tr = (final_train_grp['deltaG'] > 3).astype(int).values
    X_val_raw = final_val_grp[feature_cols].values
    y_val = (final_val_grp['deltaG'] > 3).astype(int).values
else:
    X_full_train = train_df[feature_cols].values
    y_full_train = (train_df['deltaG'] > 3).astype(int).values
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.2, random_state=final_seed, shuffle=True
    )

# --- RE-FIT PIPELINE ---
print("Fitting Imputer...")
final_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_tr_imp = final_imputer.fit_transform(X_tr_raw)
X_val_imp = final_imputer.transform(X_val_raw)

print("Fitting Scaler...")
final_scaler = StandardScaler()
X_tr_sc = final_scaler.fit_transform(X_tr_imp)
X_val_sc = final_scaler.transform(X_val_imp)

print("Fitting PCA (99%)...")
final_pca = PCA(n_components=0.99, random_state=final_seed)
X_tr_pca = final_pca.fit_transform(X_tr_sc)
X_val_pca = final_pca.transform(X_val_sc)

print(f"PCA reduced dimensions: {X_tr_raw.shape[1]} -> {X_tr_pca.shape[1]}")

# Save these artifacts immediately so you don't lose them again
joblib.dump(final_imputer, 'models/final_imputer.pkl')
joblib.dump(final_scaler, 'models/final_scaler.pkl')
joblib.dump(final_pca, 'models/final_pca.pkl')
print("Preprocessing artifacts saved to 'models/'.")

# --------------------------------------------------------------------------------
# 5. RE-TRAIN FINAL MODEL
# --------------------------------------------------------------------------------
def build_model_from_params(input_dim, params):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    num_layers = params['num_layers']
    dropout_rate = params['dropout_rate']
    lr = params['learning_rate']
    batchnorm_prob = params['batchnorm_prob']

    for i in range(num_layers):
        units = params[f'units_l{i}']
        model.add(Dense(units, activation='relu'))
        
        # Check if batchnorm param exists for this layer, else default to 0
        bn_val = params.get(f'batchnorm_{i}', 0.5) 
        if bn_val < batchnorm_prob:
            model.add(BatchNormalization())
            
        do_val = params.get(f'dropout_{i}', 0.5)
        if do_val < 0.5:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    return model

class F1ScoreCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred, zero_division=0)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()
            print(f" - Epoch {epoch+1}: New Best Val F1: {f1:.4f}")

print("\nBuilding and training final model...")
final_model = build_model_from_params(X_tr_pca.shape[1], best_params)
f1_cb = F1ScoreCallback(X_val_pca, y_val)
es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=False, verbose=1)

final_model.fit(
    X_tr_pca, y_tr,
    validation_data=(X_val_pca, y_val),
    epochs=150,
    batch_size=32,
    verbose=0, # Keep it clean
    callbacks=[es, f1_cb]
)

if f1_cb.best_weights is not None:
    print(f"Restoring best weights (F1: {f1_cb.best_f1:.4f})...")
    final_model.set_weights(f1_cb.best_weights)

final_model.save('models/final_model.h5')
print("Final model saved to 'models/final_model.h5'.")

# --------------------------------------------------------------------------------
# 6. EVALUATION ON ALL TEST SETS
# --------------------------------------------------------------------------------
print("\n=== FINAL EVALUATION ===")
results = {"Set": [], "Precision": [], "Recall": [], "NegRecall": [], "F1": []}

with open('logs/final_model_metrics.txt', 'w') as f:
    for name, df in test_sets.items():
        X_test_raw = df[feature_cols].values
        y_test = (df['deltaG'] > 3).astype(int).values
        
        # Apply Saved Pipeline
        X_test_imp = final_imputer.transform(X_test_raw)
        X_test_sc = final_scaler.transform(X_test_imp)
        X_test_pca = final_pca.transform(X_test_sc)
        
        y_prob = final_model.predict(X_test_pca, verbose=0)
        y_pred = (y_prob > 0.5).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        neg_recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_s = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results["Set"].append(name)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["NegRecall"].append(neg_recall)
        results["F1"].append(f1_s)

        log_output = (
            f"=== {name} ===\n"
            f"Confusion Matrix:\n{cm}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall (Pos): {recall:.4f}\n"
            f"Recall (Neg): {neg_recall:.4f}\n"
            f"F1 Score: {f1_s:.4f}\n\n"
        )
        print(log_output)
        f.write(log_output)

pd.DataFrame(results).to_csv('logs/final_model_test_results.csv', index=False)
print("Results saved to 'logs/final_model_test_results.csv'.")
print("Done.")