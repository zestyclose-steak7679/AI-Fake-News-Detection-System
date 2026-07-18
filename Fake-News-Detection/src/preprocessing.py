import re
import string
import random
import unicodedata
import pandas as pd
import numpy as np
import nltk
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji

from src.logger import get_logger
from src.config import CLEANED_TRAIN_DATA_PATH, PREPROCESSED_TRAIN_DATA_PATH

# Initialize logger
logger = get_logger(__name__)

# Preload resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
exceptions = {"not", "no", "nor", "never"}
stop_words = stop_words - exceptions

def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Maps treebank POS tags to WordNet POS tags.

    Args:
        treebank_tag (str): NLTK POS tag.
    Returns:
        str: WordNet POS tag.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def normalize_text(text: str) -> str:
    """
    Normalizes unicode, emojis, and specific characters.
    """
    text = emoji.replace_emoji(text, replace='')
    text = text.replace('€', 'currency')
    text = text.replace('—', ' ')
    return text

def preprocess(text: str) -> str:
    """
    Original per-row preprocessing for test compatibility.
    """
    if pd.isna(text):
        return ""
    text = str(text)

    text = normalize_text(text)
    text = contractions.fix(text)
    text = text.lower()

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""

    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

def _normalize_fast(text: str) -> str:
    """Helper for fast batched normalize up to tokenization."""
    if pd.isna(text):
        return ""
    text = str(text)

    text = normalize_text(text)
    text = contractions.fix(text)
    text = text.lower()

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_batch(texts: list) -> list:
    """
    Batched processing using pos_tag_sents.
    """
    cleaned_texts = [_normalize_fast(t) for t in texts]

    # Tokenize
    tokenized_docs = [word_tokenize(t) if t else [] for t in cleaned_texts]

    # Batch pos_tag
    pos_tagged_docs = nltk.pos_tag_sents(tokenized_docs)

    # Lemmatize and remove stop words
    results = []
    for doc in pos_tagged_docs:
        if not doc:
            results.append("")
            continue
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in doc]
        tokens = [word for word in tokens if word not in stop_words]
        results.append(" ".join(tokens))
    return results

import os
import time

def main():
    set_seeds()
    logger.info("Starting Preprocessing phase...")
    try:
        df = pd.read_csv(CLEANED_TRAIN_DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"File not found: {CLEANED_TRAIN_DATA_PATH}")
        raise e

    initial_count = len(df)

    checkpoint_file = str(PREPROCESSED_TRAIN_DATA_PATH) + ".tmp"

    start_idx = 0
    if os.path.exists(checkpoint_file):
        processed_df = pd.read_csv(checkpoint_file)
        start_idx = len(processed_df)
        logger.info(f"Found checkpoint. Resuming from row {start_idx}...")

    chunk_size = 2000
    total_rows = len(df)

    if start_idx < total_rows:
        t0 = time.time()
        for i in range(start_idx, total_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            logger.info(f"Processing chunk {i} to {min(i+chunk_size, total_rows)} of {total_rows}...")

            chunk["clean_text"] = process_batch(chunk["text"].tolist())

            mode = 'a' if i > 0 else 'w'
            header = True if i == 0 else False

            chunk.to_csv(checkpoint_file, mode=mode, header=header, index=False)

            t_elapsed = time.time() - t0
            rows_processed = (i + len(chunk)) - start_idx
            rows_per_sec = rows_processed / t_elapsed
            logger.info(f"Processed {rows_processed} rows in {t_elapsed:.2f}s ({rows_per_sec:.2f} rows/s)")

    logger.info("All chunks processed. Finalizing data...")
    final_df = pd.read_csv(checkpoint_file)

    # Drop rows where clean_text is empty or whitespace or nan
    final_df = final_df.dropna(subset=['clean_text'])
    final_df = final_df[final_df["clean_text"].astype(str).str.strip() != ""]
    final_count = len(final_df)

    logger.info(f"Dropped {initial_count - final_count} rows with empty clean_text.")

    final_df.to_csv(PREPROCESSED_TRAIN_DATA_PATH, index=False)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    logger.info("Preprocessing phase completed successfully.")

if __name__ == "__main__":
    main()
