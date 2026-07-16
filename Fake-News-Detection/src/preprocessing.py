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
from src.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


# Seed setting helper
def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def normalize_unicode(text: str) -> str:
    """
    Normalizes unicode characters, handling emojis, currency, and em-dashes.

    Args:
        text (str): Input text.
    Returns:
        str: Normalized text.
    """
    # Replace em-dashes with space to separate words
    text = text.replace('—', ' ').replace('–', ' ')
    # Normalize unicode to NFKD
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text: str) -> str:
    """
    Expands contractions in the text using the contractions library.

    Args:
        text (str): Input text.
    Returns:
        str: Text with expanded contractions.
    """
    return contractions.fix(text)

def to_lowercase(text: str) -> str:
    """
    Converts text to lowercase.

    Args:
        text (str): Input text.
    Returns:
        str: Lowercased text.
    """
    return text.lower()

def remove_urls(text: str) -> str:
    """
    Removes URLs from text.

    Args:
        text (str): Input text.
    Returns:
        str: Text without URLs.
    """
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_html(text: str) -> str:
    """
    Removes HTML tags from text.

    Args:
        text (str): Input text.
    Returns:
        str: Text without HTML tags.
    """
    return re.sub(r'<.*?>', '', text)

def remove_emails(text: str) -> str:
    """
    Removes email addresses from text.

    Args:
        text (str): Input text.
    Returns:
        str: Text without emails.
    """
    return re.sub(r'\S+@\S+', '', text)

def remove_numbers(text: str) -> str:
    """
    Removes numbers from text.

    Args:
        text (str): Input text.
    Returns:
        str: Text without numbers.
    """
    return re.sub(r'\d+', '', text)

def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from text.

    Args:
        text (str): Input text.
    Returns:
        str: Text without punctuation.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_spaces(text: str) -> str:
    """
    Collapses multiple whitespaces into a single space.

    Args:
        text (str): Input text.
    Returns:
        str: Text with single spacing and stripped ends.
    """
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text: str) -> list[str]:
    """
    Tokenizes text using NLTK word_tokenize.

    Args:
        text (str): Input text.
    Returns:
        list[str]: List of word tokens.
    """
    return word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes stopwords except for specific negation words.

    Args:
        tokens (list[str]): List of word tokens.
    Returns:
        list[str]: Filtered list of word tokens.
    """
    stop_words = set(stopwords.words('english'))
    exceptions = {"not", "no", "nor", "never"}
    stop_words = stop_words - exceptions
    return [word for word in tokens if word not in stop_words]

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
        return wordnet.NOUN # Default

def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatizes tokens using POS tags and WordNetLemmatizer.

    Args:
        tokens (list[str]): List of word tokens.
    Returns:
        list[str]: Lemmatized word tokens.
    """
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

def preprocess(text: str) -> str:
    """
    Orchestrates the full preprocessing pipeline.

    Order:
    cast str -> normalize_unicode -> expand_contractions -> lowercase ->
    remove_urls -> remove_html -> remove_emails -> remove_numbers ->
    remove_punctuation -> remove_spaces -> tokenize -> remove_stopwords ->
    lemmatize -> rejoin

    Args:
        text (str): Input raw text.
    Returns:
        str: Cleaned and preprocessed text.
    """
    if pd.isna(text):
        return ""
    text = str(text)

    text = normalize_unicode(text)
    text = expand_contractions(text)
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_emails(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_spaces(text)

    if not text:
        return ""

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)

def main():
    set_seeds()
    logger.info("Starting preprocessing pipeline.")

    input_path = "data/processed/cleaned_train.csv"
    output_path = "data/processed/preprocessed_train.csv"

    try:
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
            logger.info("Loaded dataset successfully with utf-8 encoding.")
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin-1')
            logger.info("Loaded dataset successfully with latin-1 encoding after utf-8 failed.")

        initial_count = len(df)
        logger.info(f"Initial row count: {initial_count}")

        if "text" not in df.columns:
            raise ValueError("Dataset missing 'text' column.")

        df["clean_text"] = df["text"].apply(preprocess)

        # Drop rows where clean_text is empty or just whitespace
        df = df[df["clean_text"].astype(str).str.strip() != ""]
        final_count = len(df)

        logger.info(f"Dropped {initial_count - final_count} rows with empty clean_text.")
        logger.info(f"Final row count: {final_count}")

        df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed dataset to {output_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logger.exception("Unexpected error occurred in preprocessing main().")
        raise

if __name__ == "__main__":
    main()
