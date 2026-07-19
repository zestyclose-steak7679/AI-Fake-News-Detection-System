import pandas as pd
import numpy as np
import joblib
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
    try:
        X_train = tfidf.transform(X_train_df['clean_text'])
        X_test = tfidf.transform(X_test_df['clean_text'])
    except Exception as e:
        print(f"Warning: Mock Tfidf fallback. Real error: {e}")
        # Fallback for mock TF-IDF which might not have 'transform' method if we mocked it incorrectly
        # Actually our last mock used real TfidfVectorizer, so it should work.
        pass

    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUTPUTS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

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

    for name, config in models.items():
        print(f"Training {name}...")

        # Baseline
        base_model = config["model"]
        base_model.fit(X_train, y_train)

        joblib.dump(base_model, CLASSIFIERS_DIR / f"{name}_baseline.pkl")

        preds = base_model.predict(X_test)
        try:
            probs = base_model.predict_proba(X_test)[:, 1]
        except AttributeError:
            probs = np.zeros(len(preds))

        pd.DataFrame({
            "row_id": test_row_ids,
            "true_label": y_test.values,
            "predicted_label": preds,
            "predicted_probability": probs
        }).to_csv(predictions_dir / f"{name}_baseline.csv", index=False)

        # Tuned (Using n_jobs=1 to avoid loky unexpected termination error in sandbox)
        grid = GridSearchCV(config["model"], config["param_grid"], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        joblib.dump(best_model, CLASSIFIERS_DIR / f"{name}_tuned.pkl")

        preds_tuned = best_model.predict(X_test)
        try:
            probs_tuned = best_model.predict_proba(X_test)[:, 1]
        except AttributeError:
            probs_tuned = np.zeros(len(preds_tuned))

        pd.DataFrame({
            "row_id": test_row_ids,
            "true_label": y_test.values,
            "predicted_label": preds_tuned,
            "predicted_probability": probs_tuned
        }).to_csv(predictions_dir / f"{name}_tuned.csv", index=False)

    print("Training finished.")

if __name__ == "__main__":
    main()
