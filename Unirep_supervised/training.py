import os
import json
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import random
import joblib

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer  # <--- NEW IMPORT

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Output directories
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Static seed
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# Global variable to track best F1 across all trials
BEST_GLOBAL_F1 = -1.0

# --------------------------------------------------------------------------------
# 1. LOAD DATA & CLEANING (Fixes DtypeWarning)
# --------------------------------------------------------------------------------
print("Loading datasets...")
# Fix 1: Use low_memory=False to handle mixed types temporarily
df1 = pd.read_csv('set1_embeddings_with_score.csv', low_memory=False)
df2 = pd.read_csv('set2_embeddings_with_score.csv', low_memory=False)
df3 = pd.read_csv('set3_embeddings_with_score.csv', low_memory=False)

# Helper function to force numeric types
def force_numeric(df, exclude_cols):
    print("Coercing columns to numeric...")
    for col in df.columns:
        if col not in exclude_cols:
            # Coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

metadata_cols = ['name', 'file', 'Family_Name', 'deltaG', 'sequence', 'target'] 

# Clean all dataframes
df1 = force_numeric(df1, metadata_cols)
df2 = force_numeric(df2, metadata_cols)
df3 = force_numeric(df3, metadata_cols)

test_sets = {
    "Test Set 2": df2, 
    "Test Set 3": df3
}

# --------------------------------------------------------------------------------
# 2. DEFINE FEATURES & TARGET
# --------------------------------------------------------------------------------
# Only select columns that are numeric (this filters out metadata)
feature_cols = [c for c in df1.columns if c not in metadata_cols]

print(f"Detected {len(feature_cols)} embedding features.")

# --------------------------------------------------------------------------------
# 3. SPLIT SET 1 (Train / Test 1)
# --------------------------------------------------------------------------------
if 'Family_Name' in df1.columns:
    print("Family_Name found. Using family-aware splitting.")
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
    print("Family_Name NOT found. Using random 80/20 splitting.")
    train_df, test1_df = train_test_split(df1, test_size=0.2, random_state=GLOBAL_SEED, shuffle=True)

test_sets["Test Set 1"] = test1_df

# Prepare X and y
X_full = train_df[feature_cols].values
y_full = (train_df['deltaG'] > 3).astype(int).values

# --------------------------------------------------------------------------------
# 4. UTILITIES
# --------------------------------------------------------------------------------
class F1ScoreCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = -1
        self.best_weights = None
        self.f1s = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred, zero_division=0)
        self.f1s.append(f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()

def build_model(input_dim, trial):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    num_layers = trial.suggest_int('num_layers', 1, 4) 
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batchnorm_prob = trial.suggest_float('batchnorm_prob', 0.0, 1.0)

    for i in range(num_layers):
        units = trial.suggest_categorical(f'units_l{i}', [512, 256, 128, 64, 32])
        model.add(Dense(units, activation='relu'))
        
        if trial.suggest_float(f'batchnorm_{i}', 0, 1) < batchnorm_prob:
            model.add(BatchNormalization())
            
        if trial.suggest_float(f'dropout_{i}', 0, 1) < 0.5:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    return model

def save_best_plot_and_csv(histories, f1_matrix, trial_number):
    loss_matrix = [h['loss'] for h in histories]
    val_loss_matrix = [h['val_loss'] for h in histories]
    max_len = max(len(l) for l in loss_matrix)

    avg_loss, avg_val_loss, avg_f1 = [], [], []
    for i in range(max_len):
        valid_losses = [losses[i] for losses in loss_matrix if i < len(losses)]
        valid_val_losses = [val_losses[i] for val_losses in val_loss_matrix if i < len(val_losses)]
        valid_f1s = [f1s[i] for f1s in f1_matrix if i < len(f1s)]
        
        avg_loss.append(np.mean(valid_losses) if valid_losses else 0)
        avg_val_loss.append(np.mean(valid_val_losses) if valid_val_losses else 0)
        avg_f1.append(np.mean(valid_f1s) if valid_f1s else 0)

    with open('plots/best_avg_curves.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg Train Loss', 'Avg Val Loss', 'Avg F1'])
        for i, (l, v, f1s) in enumerate(zip(avg_loss, avg_val_loss, avg_f1), 1):
            writer.writerow([i, l, v, f1s])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(range(1, max_len + 1), avg_loss, label='Avg Train Loss', linestyle='-', color='blue', alpha=0.6)
    ax1.plot(range(1, max_len + 1), avg_val_loss, label='Avg Val Loss', linestyle='--', color='orange', alpha=0.6)
    ax2.plot(range(1, max_len + 1), avg_f1, label='Avg F1 Score', linestyle='-', color='green', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('F1 Score')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Current Best Trial Metrics (Trial {trial_number})')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/best_avg_plot.png')
    plt.close()

# --------------------------------------------------------------------------------
# 5. OPTUNA OBJECTIVE (With Imputation & PCA)
# --------------------------------------------------------------------------------
def objective(trial):
    global BEST_GLOBAL_F1
    
    seed = trial.suggest_int('random_seed', 1, 10000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    histories, f1_matrix = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        X_train_raw, X_val_raw = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        # --- FIX: Imputation ---
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)

        # --- Scale ---
        cv_scaler = StandardScaler()
        X_train_sc = cv_scaler.fit_transform(X_train_imp)
        X_val_sc = cv_scaler.transform(X_val_imp)

        # --- PCA ---
        cv_pca = PCA(n_components=0.99, random_state=seed)
        X_train_pca = cv_pca.fit_transform(X_train_sc)
        X_val_pca = cv_pca.transform(X_val_sc)
        
        input_dim = X_train_pca.shape[1]

        model = build_model(input_dim, trial)
        f1_cb = F1ScoreCallback(X_val_pca, y_val)
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False, verbose=0)

        history = model.fit(
            X_train_pca, y_train,
            validation_data=(X_val_pca, y_val),
            epochs=150,
            batch_size=32,
            verbose=0,
            callbacks=[es, f1_cb]
        )
        histories.append(history.history)
        f1_matrix.append(f1_cb.f1s)

    fold_max_f1s = [max(f1s) if f1s else 0 for f1s in f1_matrix]
    avg_max_f1 = np.mean(fold_max_f1s)
    
    if avg_max_f1 > BEST_GLOBAL_F1:
        BEST_GLOBAL_F1 = avg_max_f1
        print(f"\n[Trial {trial.number}] New Best F1: {avg_max_f1:.4f}. Saving plots...")
        save_best_plot_and_csv(histories, f1_matrix, trial.number)
        
    return avg_max_f1

# --------------------------------------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    TOTAL_TRIALS = 300
    
    storage_url = "sqlite:///logs/optuna_study.db"
    
    print(f"Connecting to Optuna study storage at {storage_url}...")
    study = optuna.create_study(
        study_name="esm_optimization_case1", 
        storage=storage_url,
        direction='maximize',
        load_if_exists=True 
    )

    try:
        BEST_GLOBAL_F1 = study.best_value
        print(f"Resumed study. Previous Best F1: {BEST_GLOBAL_F1:.4f}")
    except ValueError:
        print("Starting new study (no previous trials found).")
        BEST_GLOBAL_F1 = -1.0

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_trials_remaining = TOTAL_TRIALS - completed_trials
    
    print(f"Trials completed so far: {completed_trials}")
    print(f"Trials remaining: {n_trials_remaining}")

    if n_trials_remaining > 0:
        print("Starting optimization...")
        study.optimize(objective, n_trials=n_trials_remaining, timeout=85000)
    else:
        print("Optimization target reached. Skipping to final training.")

    # --------------------------------------------------------------------------------
    # 7. FINAL RETRAINING
    # --------------------------------------------------------------------------------
    best_params = study.best_trial.params
    print("\nOptimization Complete.")
    print(f"Best F1: {study.best_value}")
    print(f"Best Params: {best_params}")

    with open('logs/best_trial.json', 'w') as f_json:
        json.dump(best_params, f_json, indent=2)
    pd.DataFrame([best_params]).to_csv('logs/best_trial.csv', index=False)
    
    pd.DataFrame(study.trials_dataframe()).to_csv("logs/all_trials.csv", index=False)
    
    print("\nStarting Final Retraining...")
    final_seed = best_params['random_seed']
    np.random.seed(final_seed)
    tf.random.set_seed(final_seed)
    random.seed(final_seed)

    # Final Train/Val Split
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
        X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=final_seed, shuffle=True
        )

    # 1. Fit Final Imputer
    print("Fitting Final Imputer...")
    final_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_tr_imp = final_imputer.fit_transform(X_tr_raw)
    X_val_imp = final_imputer.transform(X_val_raw)

    # 2. Fit Final Scaler
    print("Fitting Final Scaler...")
    final_scaler = StandardScaler()
    X_tr_sc = final_scaler.fit_transform(X_tr_imp)
    X_val_sc = final_scaler.transform(X_val_imp)
    
    # 3. Fit Final PCA
    print("Fitting Final PCA (99% variance)...")
    final_pca = PCA(n_components=0.99, random_state=final_seed)
    X_tr_pca = final_pca.fit_transform(X_tr_sc)
    X_val_pca = final_pca.transform(X_val_sc)

    print(f"Original Dim: {X_tr_raw.shape[1]} -> PCA Dim: {X_tr_pca.shape[1]}")

    # Save Pipeline
    joblib.dump(final_imputer, 'models/final_imputer.pkl') # Save Imputer
    joblib.dump(final_scaler, 'models/final_scaler.pkl')
    joblib.dump(final_pca, 'models/final_pca.pkl')
    
    final_model = build_model(X_tr_pca.shape[1], trial=optuna.trial.FixedTrial(best_params))
    f1_cb_final = F1ScoreCallback(X_val_pca, y_val)
    es_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, verbose=1)

    print("Fitting final model...")
    final_model.fit(
        X_tr_pca, y_tr,
        validation_data=(X_val_pca, y_val),
        epochs=150,
        batch_size=32,
        verbose=1,
        callbacks=[es_final, f1_cb_final]
    )

    if f1_cb_final.best_weights is not None:
        print("Restoring best F1 weights for final model...")
        final_model.set_weights(f1_cb_final.best_weights)
    
    final_model.save('models/final_model.h5')

    # --------------------------------------------------------------------------------
    # 8. EVALUATION
    # --------------------------------------------------------------------------------
    print("\nEvaluating on Test Sets...")
    results = {"Set": [], "Precision": [], "Recall": [], "NegRecall": [], "F1": []}
    
    with open('logs/final_model_metrics.txt', 'w') as f:
        for name, df in test_sets.items():
            X_test_raw = df[feature_cols].values
            y_test = (df['deltaG'] > 3).astype(int).values
            
            # PIPELINE: Impute -> Scale -> PCA
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
    print("Process Complete.")