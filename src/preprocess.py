"""
preprocess.py
-------------
Loads raw Fake/True news CSVs, cleans text, and builds a manual (from-scratch)
tokenizer alongside NLTK-based lemmatization for the production pipeline.

Expected input files (Kaggle "Fake and Real News Dataset"):
    data/raw/Fake.csv
    data/raw/True.csv
Each should have at least a 'text' (or 'title'+'text') column.

Usage:
    python src/preprocess.py
"""

import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

# ---------------------------------------------------------------------------
# NLTK setup (safe to re-run; downloads are cached after first call)
# ---------------------------------------------------------------------------
def ensure_nltk_resources():
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_data(raw_dir=RAW_DIR):
    """Loads Fake.csv and True.csv, adds labels, concatenates, and shuffles."""
    fake_path = os.path.join(raw_dir, "Fake.csv")
    true_path = os.path.join(raw_dir, "True.csv")

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise FileNotFoundError(
            f"Expected Fake.csv and True.csv in {raw_dir}. "
            "Download the Kaggle 'Fake and Real News Dataset' and place them there."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # 0 = fake
    true_df["label"] = 1  # 1 = real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text if both exist, for a richer signal
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("Could not find a 'text' column in the dataset.")

    df = df[["content", "label"]].dropna().reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


# ---------------------------------------------------------------------------
# Manual tokenizer (satisfies the "from scratch" requirement in the brief)
# ---------------------------------------------------------------------------
def manual_tokenize(text: str):
    """
    A hand-rolled tokenizer: no nltk.word_tokenize call.
    Splits on whitespace after normalizing punctuation spacing.
    """
    # Add spaces around punctuation so it splits cleanly, then split on whitespace
    text = re.sub(r"([\"'.,!?;:()\[\]{}])", r" \1 ", text)
    tokens = text.split()
    return tokens


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str, use_manual_tokenizer: bool = False) -> str:
    """Lowercases, strips HTML/URLs/emojis/punctuation/numbers, removes stopwords,
    and lemmatizes. Returns a cleaned string ready for vectorization."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)  # drops punctuation AND numbers
    text = re.sub(r"\s+", " ", text).strip()

    tokens = manual_tokenize(text) if use_manual_tokenizer else text.split()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(tok) for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    return " ".join(cleaned_tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
    ensure_nltk_resources()
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw data...")
    df = load_raw_data()
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    print("Cleaning and lemmatizing text (this can take a minute)...")
    df = preprocess_dataframe(df)

    out_path = os.path.join(PROCESSED_DIR, "clean_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path} ({len(df)} rows)")

    # Quick demo of the manual tokenizer vs cleaned output, for the report appendix
    sample = df["content"].iloc[0]
    print("\n--- Manual tokenizer demo (first row) ---")
    print("Raw snippet   :", sample[:150])
    print("Manual tokens :", manual_tokenize(sample.lower())[:20])
    print("Final cleaned :", df["clean_text"].iloc[0][:150])


if __name__ == "__main__":
    main()
