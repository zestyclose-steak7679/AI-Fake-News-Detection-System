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
    Orchestrates the full preprocessing pipeline.

    Order: normalize -> expand_contractions -> lowercase ->
    remove_urls/html/emails/numbers/punctuation -> collapse whitespace ->
    tokenize -> POS-tag on full sequence -> lemmatize using POS tags ->
    remove_stopwords -> rejoin.

    Args:
        text (str): Input raw text.
    Returns:
        str: Cleaned and preprocessed text.
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

def main():
    set_seeds()
    logger.info("Starting Preprocessing phase...")
    try:
        df = pd.read_csv(CLEANED_TRAIN_DATA_PATH)
    except FileNotFoundError as e:
        logger.error(f"File not found: {CLEANED_TRAIN_DATA_PATH}")
        raise e

    initial_count = len(df)

    df["clean_text"] = df["text"].apply(preprocess)

    # Drop rows where clean_text is empty or whitespace
    df = df[df["clean_text"].astype(str).str.strip() != ""]
    final_count = len(df)

    logger.info(f"Dropped {initial_count - final_count} rows with empty clean_text.")

    df.to_csv(PREPROCESSED_TRAIN_DATA_PATH, index=False)
    logger.info("Preprocessing phase completed successfully.")

if __name__ == "__main__":
    main()
