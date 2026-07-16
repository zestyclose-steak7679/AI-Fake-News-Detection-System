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

logger = get_logger(__name__)

def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def build_tfidf_features(X_train: pd.Series, X_test: pd.Series) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """Fits TF-IDF on train and transforms both train and test. Composable for pipeline."""
    if len(X_train) < 5:
        tfidf = TfidfVectorizer(max_features=5000)
    else:
        tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return tfidf, X_train_tfidf, X_test_tfidf

def main():
    set_seeds()
    logger.info("Starting feature engineering pipeline.")

    input_path = "data/processed/preprocessed_train.csv"
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded preprocessed dataset successfully.")

        X = df["clean_text"]
        y = df["label"]

        test_size = 0.2
        stratify_col = y if y.value_counts().min() >= 2 else None

        # Avoid error on tiny dummy data
        if len(df) < 5:
            test_size = 1 / len(df) if len(df) > 1 else 0.0

        if test_size > 0.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify_col, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
        logger.info(f"Train class balance:\n{y_train.value_counts(normalize=True).to_dict()}")
        logger.info(f"Test class balance:\n{y_test.value_counts(normalize=True).to_dict()}")

        logger.info("Building TF-IDF Vectorizer.")
        tfidf, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)

        # Save models and metadata
        joblib.dump(tfidf, "models/vectorizers/tfidf.pkl")
        logger.info("Saved TfidfVectorizer to models/vectorizers/tfidf.pkl")

        # Convert any un-serializable params to string
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

        with open("models/vectorizers/tfidf_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved TF-IDF metadata.")

        # Export vocabulary
        vocab = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
        with open("outputs/features/vocabulary.txt", "w", encoding="utf-8") as f:
            for word, idx in vocab:
                f.write(f"{word}\n")
        logger.info("Exported vocabulary to outputs/features/vocabulary.txt")

        # -------------------------------------------------------------
        # Optional comparison artifacts (Report Discussion only)
        # Not fed into Phase 5 models.
        # -------------------------------------------------------------
        try:
            logger.info("Generating CountVectorizer artifact for comparison.")
            cv = CountVectorizer(max_features=5000)
            X_train_cv = cv.fit_transform(X_train)
            X_test_cv = cv.transform(X_test)

            from gensim.models import Word2Vec
            logger.info("Generating Word2Vec artifact for comparison.")
            tokenized_train = [str(text).split() for text in X_train]
            w2v = Word2Vec(
                sentences=tokenized_train,
                vector_size=100,
                window=5,
                min_count=2 if len(X_train) >= 5 else 1,
                workers=4,
                seed=42
            )
            logger.info("Comparison artifacts generated successfully.")

        except ImportError:
            logger.info("Gensim not installed, skipping Word2Vec artifact.")

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logger.exception("Unexpected error occurred in feature_engineering main().")
        raise

if __name__ == "__main__":
    main()
