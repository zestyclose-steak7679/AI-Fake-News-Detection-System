import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from collections import Counter

def build_vocabulary(texts, max_features=None):
    """
    Builds a vocabulary from scratch for manual Bag-of-Words.
    """
    word_counts = Counter()
    for text in texts:
        if isinstance(text, str):
            words = text.split()
            word_counts.update(words)

    # Sort by frequency
    if max_features:
        most_common = word_counts.most_common(max_features)
        vocab = {word: i for i, (word, _) in enumerate(most_common)}
    else:
        vocab = {word: i for i, word in enumerate(word_counts.keys())}

    return vocab

def manual_bag_of_words(texts, vocab):
    """
    Manual Bag-of-Words implementation (word count dictionary to vector).
    """
    vectors = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):
        if isinstance(text, str):
            words = text.split()
            for word in words:
                if word in vocab:
                    vectors[i, vocab[word]] += 1

    return vectors

def extract_features(processed_dir='data/processed', output_dir='outputs/models'):
    print("Starting feature extraction...")

    input_path = os.path.join(processed_dir, 'processed_data.csv')
    df = pd.read_csv(input_path)

    # Fill NAs
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    texts = df['cleaned_text'].tolist()
    labels = df['label'].values

    print("1. Testing manual Bag-of-Words (from scratch)...")
    # Use a small subset for demonstration if data is large
    sample_texts = texts[:min(len(texts), 100)]
    vocab = build_vocabulary(sample_texts, max_features=100)
    bow_matrix = manual_bag_of_words(sample_texts, vocab)
    print(f"Manual BoW matrix shape for subset: {bow_matrix.shape}")

    print("2. Building production TF-IDF pipeline...")
    # TfidfVectorizer for production (unigrams + bigrams, max 5000 features)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tfidf = tfidf_vectorizer.fit_transform(texts)

    print(f"TF-IDF matrix shape: {X_tfidf.shape}")

    # Save the vectorizer
    os.makedirs(output_dir, exist_ok=True)
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # Save extracted features
    features_path = os.path.join(processed_dir, 'features_tfidf.npz')
    from scipy.sparse import save_npz
    save_npz(features_path, X_tfidf)

    # Save labels array
    labels_path = os.path.join(processed_dir, 'labels.npy')
    np.save(labels_path, labels)

    print("Saved vectorizer and extracted features.")

    # Optional: Word2Vec (Bonus)
    try:
        from gensim.models import Word2Vec
        print("3. Building Word2Vec embeddings (Optional bonus)...")
        tokenized_texts = [text.split() for text in texts]
        w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

        # Create document vectors by averaging word vectors
        w2v_features = np.zeros((len(texts), 100))
        for i, text in enumerate(tokenized_texts):
            vectors = [w2v_model.wv[word] for word in text if word in w2v_model.wv]
            if vectors:
                w2v_features[i] = np.mean(vectors, axis=0)

        print(f"Word2Vec features shape: {w2v_features.shape}")

        # Save w2v features
        w2v_features_path = os.path.join(processed_dir, 'features_w2v.npy')
        np.save(w2v_features_path, w2v_features)
        print("Saved Word2Vec features.")
    except ImportError:
        print("Gensim not installed, skipping Word2Vec option.")

    return X_tfidf, labels

if __name__ == "__main__":
    extract_features()
