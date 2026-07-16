"""
train.py
--------
Trains and tunes the four required models (KNN, Logistic Regression,
Random Forest, Neural Net) on TF-IDF features, plus a majority-class
baseline for comparison. Saves fitted models and the train/test split
to outputs/models/ for evaluate.py to consume.

Usage:
    python src/train.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from features import build_tfidf_features

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("outputs", "models")


def load_clean_data():
    path = os.path.join(PROCESSED_DIR, "clean_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run preprocess.py first.")
    return pd.read_csv(path)


def get_train_test_split(df, test_size=0.2, random_state=42):
    X_vec, vectorizer = build_tfidf_features(df["clean_text"].tolist())
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, vectorizer


def train_baseline(X_train, y_train):
    """Majority-class baseline so reported accuracy numbers mean something."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    return baseline


def train_models(X_train, y_train, cv_folds=5):
    """
    Trains all four required models. Logistic Regression and Random Forest
    get GridSearchCV tuning; KNN and MLP use reasonable fixed defaults to
    keep runtime manageable on a laptop without a dedicated GPU.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fitted = {}

    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    fitted["KNN"] = knn

    print("Tuning Logistic Regression (GridSearchCV over C)...")
    logreg_grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid={"C": [0.01, 0.1, 1, 10]},
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    logreg_grid.fit(X_train, y_train)
    print(f"  Best C: {logreg_grid.best_params_}")
    fitted["LogReg"] = logreg_grid.best_estimator_

    print("Tuning Random Forest (GridSearchCV over n_estimators, max_depth)...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 30],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"  Best params: {rf_grid.best_params_}")
    fitted["RandomForest"] = rf_grid.best_estimator_

    print("Training Neural Net (MLPClassifier)...")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    fitted["NeuralNet"] = mlp

    return fitted


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading cleaned data...")
    df = load_clean_data()

    print("Building TF-IDF features and train/test split...")
    X_train, X_test, y_train, y_test, vectorizer = get_train_test_split(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    baseline = train_baseline(X_train, y_train)
    models = train_models(X_train, y_train)
    models["Baseline"] = baseline

    print("\nSaving models and split to outputs/models/...")
    with open(os.path.join(MODELS_DIR, "trained_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    with open(os.path.join(MODELS_DIR, "test_split.pkl"), "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)

    print("Done. Run src/evaluate.py next.")


if __name__ == "__main__":
    main()
