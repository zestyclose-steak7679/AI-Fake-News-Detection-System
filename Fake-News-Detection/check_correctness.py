import pandas as pd
from src.preprocessing import preprocess, process_batch
from src.config import CLEANED_TRAIN_DATA_PATH

def main():
    df = pd.read_csv(CLEANED_TRAIN_DATA_PATH).head(20)
    texts = df["text"].tolist()

    old_res = [preprocess(t) for t in texts]
    new_res = process_batch(texts)

    for i, (old, new) in enumerate(zip(old_res, new_res)):
        if old != new:
            print(f"Mismatch at index {i}:")
            print(f"OLD: {old}")
            print(f"NEW: {new}")
            return
    print("Correctness check passed: All 20 rows match exactly.")

if __name__ == "__main__":
    main()
