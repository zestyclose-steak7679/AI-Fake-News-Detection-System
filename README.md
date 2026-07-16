# Fake News Detection Project

## Overview
This repository contains a full machine learning pipeline for detecting fake news. The project implements a classical NLP pipeline (Bag-of-Words / TF-IDF) and trains several models (Baseline, KNN, Logistic Regression, Random Forest, MLP) from scratch, evaluating them on various metrics (Accuracy, Precision, Recall, F1).

## Setup
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the Kaggle "Fake and Real News Dataset" files (`Fake.csv` and `True.csv`) into the `data/raw/` directory.

## Execution
To run the full pipeline end-to-end, simply execute:
```bash
python run_all.py
```

This script will sequentially run:
1. `src/preprocess.py`: Cleans the text and builds the processed dataset.
2. `src/features.py`: Extracts TF-IDF and manual Bag-of-Words features.
3. `src/train.py`: Trains the classification models.
4. `src/evaluate.py`: Generates the evaluation metrics, confusion matrices, and ROC-AUC curve.

## Outputs
- Processed data and extracted features are stored in `data/processed/`.
- Trained models are serialized as pickle files in `outputs/models/`.
- Evaluation visualizations (Confusion Matrices and ROC Curve) are saved to `outputs/figures/`.
- Extracted metrics and misclassified samples are stored in `outputs/metrics/`.