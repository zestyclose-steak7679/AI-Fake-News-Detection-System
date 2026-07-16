import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

def manual_tokenize(text):
    """
    Manual tokenization without pre-built solutions (e.g. nltk.word_tokenize)
    as per requirements.
    Uses regex to split by whitespace and non-word characters.
    """
    # Simple split by word boundaries
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def clean_text(text):
    """
    Cleans a text string by removing HTML, URLs, punctuation, numbers, and stopwords.
    Then lemmatizes the remaining tokens.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize manually
    tokens = manual_tokenize(text)

    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(cleaned_tokens)

def load_and_preprocess(raw_dir='data/raw', processed_dir='data/processed'):
    print("Starting preprocessing...")

    fake_path = os.path.join(raw_dir, 'Fake.csv')
    true_path = os.path.join(raw_dir, 'True.csv')

    # Check if files exist, else use dummy data for testing
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("Warning: Real dataset not found in data/raw. Using dummy data for end-to-end testing.")

        # Create dummy data
        dummy_fake = pd.DataFrame({
            'title': ['Fake news title 1', 'Shocking fake story'],
            'text': ['This is a completely made up story with a url http://fake.com.', 'Wow! Unbelievable fake event!'],
            'subject': ['News', 'Politics'],
            'date': ['2023-01-01', '2023-01-02']
        })
        dummy_true = pd.DataFrame({
            'title': ['Real news title 1', 'Boring real story'],
            'text': ['The government passed a new bill today. 100% verified.', 'Stocks closed higher this afternoon.'],
            'subject': ['Politics', 'News'],
            'date': ['2023-01-01', '2023-01-02']
        })

        df_fake = dummy_fake
        df_true = dummy_true
    else:
        print("Loading real dataset...")
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)

    # Add labels
    df_fake['label'] = 0 # 0 for Fake
    df_true['label'] = 1 # 1 for True

    # Concatenate and shuffle
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total articles: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # Fill NA values just in case
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')

    # Combine title and text for features
    df['full_text'] = df['title'] + " " + df['text']

    print("Cleaning text...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)

    # Save processed data
    output_path = os.path.join(processed_dir, 'processed_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

    return df

if __name__ == "__main__":
    # Ensure nltk resources are downloaded
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')

    load_and_preprocess()
