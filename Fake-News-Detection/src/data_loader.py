import pandas as pd
import random
import numpy as np
from src.logger import get_logger
from src.config import TRAIN_DATA_PATH, CLEANED_TRAIN_DATA_PATH

logger = get_logger(__name__)

def set_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def load_data(filepath: str | pd.DataFrame) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    """
    try:
        if isinstance(filepath, pd.DataFrame):
            df = filepath
        else:
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
        logger.info("Successfully loaded data.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise e

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans the loaded data according to business rules.
    Drops rows where 'text' is null or empty.
    Never fills missing text.
    """
    logger.info("Validate data...")
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'text' and 'label' columns.")

    initial_shape = df.shape
    df = df.dropna(subset=['text'])

    # Strip whitespace to check for purely empty strings
    df = df[df['text'].str.strip() != '']

    logger.info(f"Dropped {initial_shape[0] - df.shape[0]} invalid rows during validation.")
    return df

def save_data(df: pd.DataFrame, output_path: str):
    """
    Saves the cleaned DataFrame to a CSV.
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved data to {output_path}")

def main():
    set_seeds()
    logger.info("Starting Data Loading phase...")
    df = load_data(TRAIN_DATA_PATH)
    df = validate_data(df)
    save_data(df, CLEANED_TRAIN_DATA_PATH)
    logger.info("Data Loading phase completed successfully.")

if __name__ == "__main__":
    main()
