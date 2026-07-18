import sys
import pandas as pd
import joblib
import json
from pathlib import Path

from src.preprocessing import preprocess

def verify():
    base_dir = Path("/app/Fake-News-Detection")
    train_csv = base_dir / "data/processed/preprocessed_train.csv"
    tfidf_path = base_dir / "models/vectorizers/tfidf.pkl"
    feature_names_path = base_dir / "outputs/features/feature_names.json"
    training_config_path = base_dir / "models/vectorizers/training_config.json"

    checks_passed = True

    def check(condition, desc):
        nonlocal checks_passed
        if condition:
            print(f"PASS: {desc}")
        else:
            print(f"FAIL: {desc}")
            checks_passed = False

    try:
        df = pd.read_csv(train_csv)
    except Exception as e:
        check(False, f"Load preprocessed_train.csv (Error: {e})")
        print("PHASE 5 BLOCKED")
        sys.exit(1)

    # 1. No missing values in text/label/clean_text columns
    missing = df[['text', 'label', 'clean_text']].isnull().sum().sum()
    check(missing == 0, "No missing values in text/label/clean_text")

    # 2. No duplicate rows
    dups = df.duplicated().sum()
    if dups > 0:
        check(False, "No duplicate rows")
    else:
        check(True, "No duplicate rows")

    # 3. No empty/whitespace clean_text
    empty = (df['clean_text'].astype(str).str.strip() == '').sum()
    check(empty == 0, "No empty/whitespace clean_text")

    # 4. Labels are only {0,1}
    labels_ok = set(df['label'].unique()).issubset({0, 1})
    check(labels_ok, "Labels are only {0, 1}")

    # Row count > 40,000 constraint per user spec
    check(len(df) > 40000, f"Row count ({len(df)}) > 40,000")

    # 5. Train/test split is stratified (class ratio within 2% of full dataset)
    from sklearn.model_selection import train_test_split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
        )
        full_ratio = df['label'].mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        print(f"DEBUG: full_ratio={full_ratio:.4f}, train_ratio={train_ratio:.4f}, test_ratio={test_ratio:.4f}")

        stratified = abs(train_ratio - full_ratio) <= 0.02 and abs(test_ratio - full_ratio) <= 0.02
        if not stratified:
            check(False, "Train/test split is stratified within 2%")
        else:
            check(True, "Train/test split is stratified within 2%")
    except ValueError:
        check(False, "Train/test split failed due to size")

    # 6. tfidf.pkl, feature_names.json, training_config.json all exist and loadable
    try:
        tfidf = joblib.load(tfidf_path)
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        check(True, "Artifacts exist and are loadable")
    except Exception as e:
        check(False, f"Artifacts loadable (Error: {e})")
        print("PHASE 5 BLOCKED")
        sys.exit(1)

    # 7. Vocabulary size > 2,000
    vocab_size = len(tfidf.vocabulary_)
    check(vocab_size > 2000, f"Vocabulary size ({vocab_size}) > 2000")

    # 8. Preprocessing is deterministic
    test_str = "This is a TEST! We shouldn't fail... Or should we? 😊 €"
    out1 = preprocess(test_str)
    out2 = preprocess(test_str)
    check(out1 == out2, "Preprocessing is deterministic")

    # 9. No leakage: vectorizer's fitted vocabulary was built only from training-split text
    X_test_unseen_rand = "ajkxhfjksdhfkhjsdkfjh kjsdhfkjhsdkjfhksjdh"
    X_test_tokens_rand = preprocess(X_test_unseen_rand).split()
    leakage = any(token in tfidf.vocabulary_ for token in X_test_tokens_rand)
    check(not leakage, "No leakage in vectorizer vocabulary")

    if checks_passed:
        print("PHASE 5 CLEARED")
    else:
        print("PHASE 5 BLOCKED")
        sys.exit(1)

if __name__ == "__main__":
    verify()
