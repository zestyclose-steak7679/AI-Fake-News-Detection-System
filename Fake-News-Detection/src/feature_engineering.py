import random
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.logger import get_logger
from src.config import PREPROCESSED_TRAIN_DATA_PATH, VECTORIZERS_DIR, FEATURES_DIR

logger = get_logger(__name__)

def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def build_tfidf_features(X_train: pd.Series, X_test: pd.Series) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """Fits TF-IDF on train and transforms both train and test. Composable for pipeline."""
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return tfidf, X_train_tfidf, X_test_tfidf

def main():
    set_seeds()
    logger.info("Starting feature engineering phase...")
    try:
        df = pd.read_csv(PREPROCESSED_TRAIN_DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"File not found: {PREPROCESSED_TRAIN_DATA_PATH}")
        raise e

    # Drop any remaining NaN values to avoid invalid documents
    df = df.dropna(subset=['clean_text'])
    df = df[df['clean_text'].astype(str).str.strip() != '']

    X = df["clean_text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Train class balance:\n{y_train.value_counts(normalize=True).to_dict()}")
    logger.info(f"Test class balance:\n{y_test.value_counts(normalize=True).to_dict()}")

    tfidf, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)

    # Save model
    joblib.dump(tfidf, VECTORIZERS_DIR / "tfidf.pkl")

    # Save feature names
    feature_names = tfidf.get_feature_names_out().tolist()
    with open(FEATURES_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)

    # Save training config
    training_config = {
        "vectorizer": "tfidf",
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": [1, 2],
        "random_state": 42
    }
    with open(VECTORIZERS_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=4)

    # Save metadata
    params = tfidf.get_params()
    for k, v in params.items():
        if type(v) == type:
            params[k] = str(v)

    metadata = {
        "params": params,
        "n_features": len(tfidf.vocabulary_),
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now().isoformat()
    }
    with open(VECTORIZERS_DIR / "tfidf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Save vocabulary
    vocab = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
    with open(FEATURES_DIR / "vocabulary.txt", "w", encoding="utf-8") as f:
        for word, idx in vocab:
            f.write(f"{word}\n")

    # GenSim Word2Vec constraint: only build Word2Vec + BoW for comparison
    cv = CountVectorizer(max_features=5000)
    cv.fit(X_train)

    from gensim.models import Word2Vec
    tokenized_train = [str(text).split() for text in X_train]
    w2v = Word2Vec(sentences=tokenized_train, vector_size=100, window=5, min_count=2, workers=4, seed=42)

    import hashlib
    with open(PREPROCESSED_TRAIN_DATA_PATH, "rb") as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()
    with open(VECTORIZERS_DIR / "tfidf.pkl", "rb") as f:
        vectorizer_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "processed_dataset_sha256": dataset_hash,
        "vectorizer_sha256": vectorizer_hash,
        "row_count": len(df),
        "vocabulary_size": len(tfidf.vocabulary_)
    }
    with open(VECTORIZERS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    logger.info("Feature engineering phase completed successfully.")

if __name__ == "__main__":
    main()
