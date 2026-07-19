import pandas as pd
import numpy as np
import os
import json
import hashlib
import sklearn
import sys
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    PREPROCESSED_TRAIN_DATA_PATH,
    OUTPUTS_DIR,
    CLASSIFIERS_DIR,
    BASE_DIR
)

# Use dummy logger if experiment_logger is not found
try:
    from src.experiment_logger import log_metrics
except ModuleNotFoundError:
    def log_metrics(name, metrics):
        pass

def main():
    predictions_dir = OUTPUTS_DIR / "predictions"
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load original dataset to extract original text for error analysis
    df_orig = pd.read_csv(PREPROCESSED_TRAIN_DATA_PATH)
    df_orig['row_id'] = df_orig.index

    results = []
    error_dfs = []

    for file_path in predictions_dir.glob("*.csv"):
        model_variant = file_path.stem # e.g. "LogReg_baseline"

        # Get model size
        model_path = CLASSIFIERS_DIR / f"{model_variant}.pkl"
        model_size_bytes = os.path.getsize(model_path) if model_path.exists() else 0

        # Read predictions
        df_preds = pd.read_csv(file_path)
        y_true = df_preds['true_label']
        y_pred = df_preds['predicted_label']

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "Model": model_variant,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "train_time": 0.0, # Not recorded by train.py currently
            "predict_time": 0.0,
            "model_size_bytes": model_size_bytes
        })

        log_metrics(model_variant, {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

        # Collect errors
        errors = df_preds[df_preds['true_label'] != df_preds['predicted_label']].copy()
        if not errors.empty:
            errors['model_variant'] = model_variant
            errors = errors.merge(df_orig[['row_id', 'text', 'clean_text']], on='row_id', how='left')
            error_dfs.append(errors)

    if results:
        results_df = pd.DataFrame(results)
        # Sort by F1 descending for a nice output
        results_df = results_df.sort_values(by="F1", ascending=False)
        results_df.to_csv(metrics_dir / "comparison_table.csv", index=False)
        print("\n--- Final Comparison Table ---")
        print(results_df.to_string(index=False))

    if error_dfs:
        all_errors_df = pd.concat(error_dfs, ignore_index=True)
        all_errors_df.to_csv(metrics_dir / "error_analysis.csv", index=False)
        print(f"\nSaved error analysis for {len(all_errors_df)} total misclassifications.")

    # Write training metadata
    try:
        commit_hash = os.popen("git rev-parse HEAD").read().strip()
    except Exception:
        commit_hash = "unknown"

    try:
        with open(PREPROCESSED_TRAIN_DATA_PATH, "rb") as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        dataset_hash = "unknown"

    metadata = {
        "dataset_sha256": dataset_hash,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "random_state": 42,
        "git_commit_hash": commit_hash,
        "timestamp": time.time()
    }

    with open(BASE_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()
