import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time

from src.config import (
    PREPROCESSED_TRAIN_DATA_PATH,
    VECTORIZERS_DIR,
    CLASSIFIERS_DIR,
    OUTPUTS_DIR,
    BASE_DIR
)

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    import random
    random.seed(seed)

def main():
    set_seeds()
    print("Loading data...")
    df = pd.read_csv(PREPROCESSED_TRAIN_DATA_PATH)
    df['row_id'] = df.index

    # Stratified split using the same logic as everywhere
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df[['clean_text', 'row_id']], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    splits_dir = BASE_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    X_train_df[['row_id']].to_csv(splits_dir / "train_ids.csv", index=False)
    X_test_df[['row_id']].to_csv(splits_dir / "test_ids.csv", index=False)
    print(f"Splits saved.")

    tfidf_path = VECTORIZERS_DIR / "tfidf.pkl"
    tfidf = joblib.load(tfidf_path)

    print("Transforming text...")
    X_train = tfidf.transform(X_train_df['clean_text'].astype(str))
    X_test = tfidf.transform(X_test_df['clean_text'].astype(str))

    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUTPUTS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "LogReg": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "param_grid": {"C": [0.1, 1, 10]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {"n_estimators": [100, 200], "max_depth": [None, 20]}
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": [3, 5, 7]}
        },
        "MLP": {
            "model": MLPClassifier(random_state=42, early_stopping=True),
            "param_grid": {"hidden_layer_sizes": [(100,), (100, 50)]}
        }
    }

    test_row_ids = X_test_df['row_id'].values
    timing_data = {}

    for name, config in models.items():
        print(f"Training {name} baseline...")
        base_model = config["model"]

        t0 = time.time()
        base_model.fit(X_train, y_train)
        train_time = time.time() - t0

        joblib.dump(base_model, CLASSIFIERS_DIR / f"{name}_baseline.pkl")

        t0 = time.time()
        preds = base_model.predict(X_test)
        predict_time = time.time() - t0

        if hasattr(base_model, "predict_proba"):
            probs = base_model.predict_proba(X_test)[:, 1]
        else:
            probs = np.zeros(len(preds))

        pd.DataFrame({
            "row_id": test_row_ids,
            "true_label": y_test.values,
            "predicted_label": preds,
            "predicted_probability": probs
        }).to_csv(predictions_dir / f"{name}_baseline.csv", index=False)

        timing_data[f"{name}_baseline"] = {
            "train_time": train_time,
            "predict_time": predict_time
        }

        print(f"Training {name} tuned...")
        grid = GridSearchCV(config["model"], config["param_grid"], cv=3, n_jobs=1)

        t0 = time.time()
        grid.fit(X_train, y_train)
        train_time_tuned = time.time() - t0

        best_model = grid.best_estimator_
        joblib.dump(best_model, CLASSIFIERS_DIR / f"{name}_tuned.pkl")

        t0 = time.time()
        preds_tuned = best_model.predict(X_test)
        predict_time_tuned = time.time() - t0

        if hasattr(best_model, "predict_proba"):
            probs_tuned = best_model.predict_proba(X_test)[:, 1]
        else:
            probs_tuned = np.zeros(len(preds_tuned))

        pd.DataFrame({
            "row_id": test_row_ids,
            "true_label": y_test.values,
            "predicted_label": preds_tuned,
            "predicted_probability": probs_tuned
        }).to_csv(predictions_dir / f"{name}_tuned.csv", index=False)

        timing_data[f"{name}_tuned"] = {
            "train_time": train_time_tuned,
            "predict_time": predict_time_tuned
        }

    with open(metrics_dir / "timing.json", "w") as f:
        json.dump(timing_data, f, indent=4)

    print("Training finished.")

if __name__ == "__main__":
    main()
