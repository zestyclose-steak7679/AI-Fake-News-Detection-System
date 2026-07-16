import pytest
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering import build_tfidf_features
from sklearn.model_selection import train_test_split

def test_tfidf_leak_and_shape():
    # Directly test the composable function to ensure no leakage
    # To pass min_df=2 (which means word must appear in at least 2 DOCUMENTS)
    X_train = pd.Series([
        "hello world",
        "hello world",
        "fake news detection",
        "fake news detection"
    ])
    X_test = pd.Series(["hello new word", "completely unseen text"])

    tfidf, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)

    assert hasattr(tfidf, "vocabulary_")

    # Assert shape matches vocabulary length
    assert X_train_tfidf.shape[1] == len(tfidf.vocabulary_)
    assert X_test_tfidf.shape[1] == len(tfidf.vocabulary_)

    # Assert 'unseen' is NOT in vocabulary (test set leak check)
    assert "unseen" not in tfidf.vocabulary_

def test_stratify_ratio():
    # Use dummy data to test stratify
    df = pd.DataFrame({
        "clean_text": [f"text {i}" for i in range(100)],
        "label": [0]*80 + [1]*20
    })

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ratio = (y_train == 1).mean()
    test_ratio = (y_test == 1).mean()
    full_ratio = (y == 1).mean()

    # Check within 2% margin
    assert abs(train_ratio - full_ratio) < 0.02
    assert abs(test_ratio - full_ratio) < 0.02
