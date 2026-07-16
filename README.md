# Fake News Detection — AI/ML Internship Project 1

Full ML pipeline for classifying news articles as real or fake, built for
the IICT Summer Internship Program in AI&ML (2026).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Dataset

Download the Kaggle **"Fake and Real News Dataset"** and place both files here:

```
data/raw/Fake.csv
data/raw/True.csv
```

## Run the full pipeline

```bash
python run_all.py
```

This runs, in order:
1. `src/preprocess.py` — cleans text, lemmatizes, saves `data/processed/clean_data.csv`
2. `src/train.py` — builds TF-IDF features, trains KNN / Logistic Regression /
   Random Forest / Neural Net (with GridSearchCV tuning on LogReg + RF),
   saves models to `outputs/models/`
3. `src/evaluate.py` — computes accuracy/precision/recall/F1, confusion matrices,
   ROC-AUC overlay, and a basic error analysis; saves everything to `outputs/`

## Run steps individually

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

## Manual "from scratch" components (for the report appendix)

- `preprocess.py::manual_tokenize()` — hand-rolled tokenizer (no `nltk.word_tokenize`)
- `features.py::manual_bag_of_words()` / `build_vocabulary()` — BoW built from
  raw word counts, not `CountVectorizer` (demo runs on a 500-doc sample for speed;
  full-dataset modeling uses TF-IDF via sklearn)
- `features.py::build_word2vec_features()` — optional embedding comparison (needs gensim)

## Outputs

- `outputs/models/` — pickled trained models + vectorizer + test split
- `outputs/figures/` — confusion matrices, ROC overlay
- `outputs/metrics/` — model comparison CSV (accuracy/precision/recall/F1)

## Report

Write the IEEE-format report in `report/`, following the 8-section structure
from the project brief (Introduction, Dataset Description, Methodology,
Results, Discussion, Conclusion, Appendix, Format & Standard).
