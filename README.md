# AI-Powered Fake News Detection System

A machine learning pipeline that classifies news articles as **real** or **fake**, built from scratch (no pre-trained classifiers) as part of the IICT Summer Internship Program in AI & ML (2026).

## Project Status

- [x] Environment setup (Phase 0)
- [x] Data loading & validation (Phase 1)
- [x] Exploratory Data Analysis (Phase 2)
- [x] Text preprocessing — contraction expansion, POS-aware lemmatization, negation-preserving stopword removal (Phase 3)
- [x] Feature engineering — TF-IDF (primary), BoW/Word2Vec (comparison only), leak-free train/test split (Phase 4)
- [x] CI pipeline verification (GitHub Actions)
- [ ] Model training — KNN, Logistic Regression, Random Forest, Neural Network (Phase 5)
- [ ] Error analysis & model comparison
- [ ] IEEE report
- [ ] Presentation

## Dataset

Kaggle's ["Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — `Fake.csv` (~23,490 rows) and `True.csv` (~21,418 rows), combined and cleaned to ~44,689 rows.

Place both files at:
```
data/raw/Fake.csv
data/raw/True.csv
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python setup_nltk.py
```

`setup_nltk.py` downloads all required NLTK resources (punkt, punkt_tab, stopwords, wordnet, omw-1.4, averaged_perceptron_tagger) and won't fail the whole setup if one optional resource is unavailable.

## Verify your setup

```bash
python test_install.py
```
Expected: `All libraries imported successfully!`

## Running the pipeline

Run each stage in order:

```bash
python src/data_loader.py        # loads Fake.csv + True.csv, cleans, saves data/processed/cleaned_train.csv
python src/eda.py                # generates 6 plots in outputs/graphs/
python src/preprocessing.py      # cleans text, saves data/processed/preprocessed_train.csv
python src/feature_engineering.py # builds TF-IDF features, saves models/vectorizers/
python src/verify_pipeline.py    # hard verification gate — must print "PHASE 5 CLEARED"
pytest tests/ -v                 # unit tests for preprocessing + feature engineering
```

`preprocessing.py` uses batched NLTK POS-tagging (`pos_tag_sents`) with checkpointing — safe to interrupt and re-run; it resumes from the last completed chunk rather than restarting.

## Verification gate

Before any model training begins, `src/verify_pipeline.py` must print `PHASE 5 CLEARED`. It checks, independently of any script's self-reported output:

- No missing values / duplicate rows / empty documents
- Labels are only `{0, 1}`
- Dataset has more than 40,000 rows (guards against ever accidentally training on stub/sample data)
- Train/test split is stratified within 2% of the full dataset's class balance
- TF-IDF vectorizer, feature names, and training config all exist and load correctly
- Vocabulary size exceeds 2,000 terms
- Preprocessing is deterministic (same input always produces the same output)
- No data leakage — the vectorizer's vocabulary was built only from training-split text

This script's logic is treated as frozen/protected — it defines what "correct" means for this project and should not be edited casually. If you ever need to change a threshold here, do it deliberately and document why.

## Continuous Integration

`.github/workflows/verify-pipeline.yml` runs the entire pipeline (data loading → EDA → preprocessing → feature engineering → verification → tests) on every push, independent of any local machine or agent. This exists specifically so that "it works" is confirmed by GitHub's infrastructure, not just a self-report.

## Project structure

```
Fake-News-Detection/
├── data/
│   ├── raw/              # Fake.csv, True.csv (not committed — see .gitignore)
│   └── processed/        # cleaned_train.csv, preprocessed_train.csv
├── src/
│   ├── config.py          # centralized paths, seeds, model defaults
│   ├── logger.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── verify_pipeline.py  # hard verification gate — do not edit casually
├── tests/                 # pytest unit tests
├── models/
│   ├── vectorizers/       # tfidf.pkl, training_config.json
│   └── classifiers/       # (Phase 5) trained models
├── outputs/
│   ├── graphs/            # EDA visualizations
│   ├── features/          # vocabulary.txt, feature_names.json
│   └── metrics/           # experiment logs (Phase 5)
├── report/                # IEEE report
├── logs/project.log
├── setup_nltk.py
├── test_install.py
├── requirements.txt
└── .github/workflows/verify-pipeline.yml
```

## Known limitations

- No non-English language detection — the pipeline assumes English-language input.
- BoW and Word2Vec feature representations are built for report/comparison purposes only; TF-IDF is the sole representation carried forward into model training.

## Reproducibility

Random seeds (`random`, `numpy`, `scikit-learn`) are fixed to `42` throughout. Dataset integrity is tracked via SHA256 fingerprint stored in `models/vectorizers/training_config.json`.
