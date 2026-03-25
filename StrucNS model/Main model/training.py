import os
import json
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import random
import joblib  # <--- Added for saving the Scaler

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Suppress TF warnings for cleaner output
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
# We will initialize this in main() to ensure it syncs with the DB on resume
BEST_GLOBAL_F1 = -1.0

# --------------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------------
# NOTE: Ensure these files are in your current directory
# Using try-except block to make it safer if files are missing during dry runs
try:
    df1 = pd.read_csv('set1_features.csv')
    df2 = pd.read_csv('set2_features.csv')
    df3 = pd.read_csv('set3_features.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    # Create dummy data for structure if files missing (prevents immediate crash during code review)
    # In production, this will just crash, which is intended.
    raise e

# Configure Test Sets Dictionary
test_sets = {
    "Test Set 2": df2, 
    "Test Set 3": df3
}

# --------------------------------------------------------------------------------
# 2. SPLIT SET 1 (Train / Test 1) WITH FAMILY GROUPS
# --------------------------------------------------------------------------------
train_list, test1_list = [], []

# Group by 'Family_Name'
for _, group in df1.groupby('Family_Name'):
    if len(group) < 2:
        train_list.append(group)
    else:
        # Shuffle group safely
        grp_shuffled = group.sample(frac=1, random_state=GLOBAL_SEED)
        split_point = int(len(grp_shuffled) * 0.8) # 80/20 split
        
        train_list.append(grp_shuffled.iloc[:split_point])
        test1_list.append(grp_shuffled.iloc[split_point:])

train_df = pd.concat(train_list).reset_index(drop=True)
test1_df = pd.concat(test1_list).reset_index(drop=True)

# Add Test Set 1 to our evaluation dictionary
test_sets["Test Set 1"] = test1_df

# Define feature columns dynamically
metadata_cols = ['file', 'Family_Name', 'deltaG']
feature_cols = [c for c in df1.columns if c not in metadata_cols]

print(f"Total features selected: {len(feature_cols)}")

# Prepare X_full and y_full (Unscaled)
X_full = train_df[feature_cols].values
y_full = (train_df['deltaG'] > 3).astype(int).values

# --------------------------------------------------------------------------------
# 3. UTILITIES
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
    
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    batchnorm_prob = trial.suggest_float('batchnorm_prob', 0.0, 1.0)

    for i in range(num_layers):
        units = trial.suggest_categorical(f'units_l{i}', [512, 256, 128, 64, 32, 16])
        model.add(Dense(units, activation='relu'))
        
        if trial.suggest_float(f'batchnorm_{i}', 0, 1) < batchnorm_prob:
            model.add(BatchNormalization())
            
        if trial.suggest_float(f'dropout_{i}', 0, 1) < 0.5:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    return model

def save_best_plot_and_csv(histories, f1_matrix, trial_number):
    """Generates plot and CSV only for the best trial so far."""
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

    # Save CSV
    with open('plots/best_avg_curves.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg Train Loss', 'Avg Val Loss', 'Avg F1'])
        for i, (l, v, f1s) in enumerate(zip(avg_loss, avg_val_loss, avg_f1), 1):
            writer.writerow([i, l, v, f1s])

    # Save Plot
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
# 4. OPTUNA OBJECTIVE
# --------------------------------------------------------------------------------
def objective(trial):
    global BEST_GLOBAL_F1
    
    seed = trial.suggest_int('random_seed', 1, 10000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    histories, f1_matrix = [], []

    # Iterate on indices to slice X_full/y_full
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), 1):
        X_train_raw, X_val_raw = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        # Fit scaler ONLY on training fold (prevent leakage)
        cv_scaler = StandardScaler()
        X_train = cv_scaler.fit_transform(X_train_raw)
        X_val = cv_scaler.transform(X_val_raw)

        model = build_model(X_full.shape[1], trial)

        f1_cb = F1ScoreCallback(X_val, y_val)
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False, verbose=0)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            verbose=0,
            callbacks=[es, f1_cb]
        )

        histories.append(history.history)
        f1_matrix.append(f1_cb.f1s)

    # Calculate optimization metric (Average of Max F1s per fold)
    fold_max_f1s = [max(f1s) if f1s else 0 for f1s in f1_matrix]
    avg_max_f1 = np.mean(fold_max_f1s)
    
    # Check if this trial is the best so far
    if avg_max_f1 > BEST_GLOBAL_F1:
        BEST_GLOBAL_F1 = avg_max_f1
        print(f"\n[Trial {trial.number}] New Best F1: {avg_max_f1:.4f}. Saving plots...")
        save_best_plot_and_csv(histories, f1_matrix, trial.number)
        
    return avg_max_f1

# --------------------------------------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # --- RESUME CAPABILITY SETUP ---
    # Define a persistent storage URL (SQLite)
    # This creates a file 'optuna_study.db' in the logs folder.
    storage_url = "sqlite:///logs/optuna_study.db"
    study_name = "deltag_optimization_v1"
    TOTAL_TRIALS = 300

    # Create (or load) the study
    print(f"Loading/Creating study '{study_name}' with storage '{storage_url}'...")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='maximize'
    )

    # Sync Global Variable with DB state (in case we are resuming)
    try:
        if len(study.trials) > 0:
            BEST_GLOBAL_F1 = study.best_value
            print(f"Resuming study. Current Best F1 from DB: {BEST_GLOBAL_F1:.4f}")
        else:
            print("Starting fresh study.")
    except Exception:
        # Failsafe if best_value is not yet set (e.g. only failed trials exist)
        BEST_GLOBAL_F1 = -1.0

    # Calculate remaining trials
    # trials includes RUNNING, WAITING, PRUNED, COMPLETE, FAIL
    # We generally only care about completed trials, but simply checking len(study.trials) 
    # is a safe approximation to ensure we don't run forever.
    completed_trials = len(study.trials)
    remaining_trials = TOTAL_TRIALS - completed_trials

    if remaining_trials > 0:
        print(f"Starting optimization for {remaining_trials} more trials...")
        study.optimize(objective, n_trials=remaining_trials, timeout=72000)
    else:
        print("Study has already reached target number of trials. Skipping optimization.")

    # --- POST OPTIMIZATION ---

    best_params = study.best_trial.params
    print("\nOptimization Complete.")
    print(f"Best F1: {study.best_value}")
    print(f"Best Params: {best_params}")

    # Save Study Results
    with open('logs/best_trial.json', 'w') as f_json:
        json.dump(best_params, f_json, indent=2)
    
    # Export all trials to CSV (good for analysis later)
    pd.DataFrame(study.trials_dataframe()).to_csv("logs/all_trials.csv", index=False)
    
    with open('logs/best_trial.txt', 'w') as f_txt:
        for k, v in best_params.items():
            f_txt.write(f"{k}: {v}\n")

    # --------------------------------------------------------------------------------
    # 6. FINAL RETRAINING (With Correct Splits)
    # --------------------------------------------------------------------------------
    print("\nStarting Final Retraining...")
    
    # Use best seed
    final_seed = best_params['random_seed']
    np.random.seed(final_seed)
    tf.random.set_seed(final_seed)
    random.seed(final_seed)

    # Re-split train_df into final_train and final_val ensuring no family leakage
    groups = [group for _, group in train_df.groupby('Family_Name')]
    random.shuffle(groups)
    
    split_idx = int(len(groups) * 0.8)
    final_train_grp = pd.concat(groups[:split_idx])
    final_val_grp = pd.concat(groups[split_idx:])
    
    # Fit Final Scaler on Final Train split only
    final_scaler = StandardScaler()
    X_train_f = final_scaler.fit_transform(final_train_grp[feature_cols].values)
    y_train_f = (final_train_grp['deltaG'] > 3).astype(int).values
    
    # --- SAVE SCALER HERE ---
    print("Saving StandardScaler to models/final_scaler.gz ...")
    joblib.dump(final_scaler, 'models/final_scaler.gz')
    # ------------------------

    X_val_f = final_scaler.transform(final_val_grp[feature_cols].values)
    y_val_f = (final_val_grp['deltaG'] > 3).astype(int).values

    # Build and Train Final Model
    final_model = build_model(X_full.shape[1], trial=optuna.trial.FixedTrial(best_params))
    
    f1_cb_final = F1ScoreCallback(X_val_f, y_val_f)
    es_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, verbose=1)

    print("Fitting final model...")
    final_model.fit(
        X_train_f, y_train_f,
        validation_data=(X_val_f, y_val_f),
        epochs=150,
        batch_size=32,
        verbose=1,
        callbacks=[es_final, f1_cb_final]
    )

    # Explicitly set the weights to the one that yielded max F1 on validation
    if f1_cb_final.best_weights is not None:
        print("Restoring best F1 weights for final model...")
        final_model.set_weights(f1_cb_final.best_weights)
    
    final_model.save('models/final_model.h5')

    # --------------------------------------------------------------------------------
    # 7. EVALUATION
    # --------------------------------------------------------------------------------
    print("\nEvaluating on Test Sets...")
    results = {"Set": [], "Precision": [], "Recall": [], "NegRecall": [], "F1": []}
    
    with open('logs/final_model_metrics.txt', 'w') as f:
        for name, df in test_sets.items():
            # Apply Final Scaler to Test Sets
            # IMPORTANT: We use the saved scaler instance here
            X_eval = final_scaler.transform(df[feature_cols].values)
            y_eval = (df['deltaG'] > 3).astype(int).values
            
            y_prob = final_model.predict(X_eval, verbose=0)
            y_pred = (y_prob > 0.5).astype(int)

            precision = precision_score(y_eval, y_pred, zero_division=0)
            recall = recall_score(y_eval, y_pred, zero_division=0)
            neg_recall = recall_score(y_eval, y_pred, pos_label=0, zero_division=0)
            f1_s = f1_score(y_eval, y_pred, zero_division=0)
            cm = confusion_matrix(y_eval, y_pred)

            results["Set"].append(name)
            results["Precision"].append(precision)
            results["Recall"].append(recall)
            results["NegRecall"].append(neg_recall)
            results["F1"].append(f1_s)

            log_output = (
                f"=== {name} ===\n"
                f"Confusion Matrix:\n{cm}\n"
                f"Precision: {precision:.1f}\n"
                f"Recall (Pos): {recall:.1f}\n"
                f"Recall (Neg): {neg_recall:.1f}\n"
                f"F1 Score: {f1_s:.1f}\n\n"
            )
            print(log_output)
            f.write(log_output)

    pd.DataFrame(results).to_csv('logs/final_model_test_results.csv', index=False)
    print("Process Complete.")