"""
features.py
------------
Feature extraction methods for the fake news classifier:
  1. manual_bag_of_words   - hand-rolled BoW (satisfies "from scratch" requirement)
  2. build_tfidf_features  - production TF-IDF via sklearn (used for actual modeling)
  3. build_word2vec_features - optional embedding-based representation (gensim)

Usage (as a library):
    from features import manual_bag_of_words, build_tfidf_features

Usage (standalone demo):
    python src/features.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("outputs", "models")


# ---------------------------------------------------------------------------
# 1. Manual Bag-of-Words (from scratch, no CountVectorizer)
# ---------------------------------------------------------------------------
def build_vocabulary(tokenized_docs, max_features=2000):
    """Builds a vocabulary of the top-N most frequent words across all docs."""
    counter = Counter()
    for doc in tokenized_docs:
        counter.update(doc)
    most_common = counter.most_common(max_features)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    return vocab


def manual_bag_of_words(tokenized_docs, vocab=None, max_features=2000):
    """
    Builds a Bag-of-Words matrix by hand: for each document, count occurrences
    of each vocabulary word. Returns (matrix, vocab).

    tokenized_docs: list of lists of tokens, e.g. [["fake", "news", ...], ...]
    Note: intentionally O(n_docs * n_vocab) and unoptimized -- this is meant to
    demonstrate the mechanics for the report/appendix, not to scale to the
    full dataset. Use build_tfidf_features() for actual model training.
    """
    if vocab is None:
        vocab = build_vocabulary(tokenized_docs, max_features=max_features)

    n_docs = len(tokenized_docs)
    n_vocab = len(vocab)
    matrix = np.zeros((n_docs, n_vocab), dtype=np.int32)

    for i, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        for word, count in counts.items():
            if word in vocab:
                matrix[i, vocab[word]] = count

    return matrix, vocab


# ---------------------------------------------------------------------------
# 2. TF-IDF (production feature set used for actual training)
# ---------------------------------------------------------------------------
def build_tfidf_features(texts, max_features=5000, ngram_range=(1, 2), save=True):
    """Fits a TfidfVectorizer on the given texts and returns (X, vectorizer)."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

    return X, vectorizer


# ---------------------------------------------------------------------------
# 3. Word2Vec embeddings (optional, for the "with proper expansion" comparison)
# ---------------------------------------------------------------------------
def build_word2vec_features(tokenized_docs, vector_size=100, window=5, min_count=2):
    """
    Trains a Word2Vec model on the corpus and represents each document as the
    mean of its word vectors. Requires gensim (pip install gensim).
    """
    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError(
            "gensim is required for Word2Vec features. Install with: "
            "pip install gensim"
        ) from e

    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42,
    )

    def doc_vector(tokens):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            return np.zeros(vector_size)
        return np.mean(vecs, axis=0)

    doc_vectors = np.array([doc_vector(doc) for doc in tokenized_docs])
    return doc_vectors, model


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
def main():
    clean_path = os.path.join(PROCESSED_DIR, "clean_data.csv")
    if not os.path.exists(clean_path):
        print(f"Run preprocess.py first -- {clean_path} not found.")
        return

    df = pd.read_csv(clean_path)
    sample_df = df.sample(n=min(500, len(df)), random_state=42)  # keep demo fast
    tokenized_docs = sample_df["clean_text"].str.split().tolist()

    print("Building manual Bag-of-Words on a 500-doc sample (demo only)...")
    bow_matrix, vocab = manual_bag_of_words(tokenized_docs, max_features=200)
    print(f"Manual BoW matrix shape: {bow_matrix.shape}")
    print("Top 10 vocab words:", list(vocab.keys())[:10])

    print("\nBuilding TF-IDF on full cleaned dataset...")
    X_tfidf, vectorizer = build_tfidf_features(df["clean_text"].tolist())
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")


if __name__ == "__main__":
    main()
