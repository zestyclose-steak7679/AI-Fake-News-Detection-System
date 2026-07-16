"""
evaluate.py
-----------
Loads trained models + test split, computes accuracy/precision/recall/F1,
plots confusion matrices and an ROC-AUC overlay for all models, and pulls
misclassified examples for error analysis (feeds the Discussion section
of the IEEE report).

Usage:
    python src/evaluate.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

MODELS_DIR = os.path.join("outputs", "models")
FIGURES_DIR = os.path.join("outputs", "figures")
METRICS_DIR = os.path.join("outputs", "metrics")
PROCESSED_DIR = os.path.join("data", "processed")


def load_artifacts():
    with open(os.path.join(MODELS_DIR, "trained_models.pkl"), "rb") as f:
        models = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "test_split.pkl"), "rb") as f:
        split = pickle.load(f)
    return models, split["X_test"], split["y_test"]


def compute_metrics_table(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(models, X_test, y_test, save_path):
    non_baseline = {k: v for k, v in models.items() if k != "Baseline"}
    n = len(non_baseline)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, non_baseline.items()):
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrices to {save_path}")


def plot_roc_overlay(models, X_test, y_test, save_path):
    plt.figure(figsize=(7, 6))

    for name, model in models.items():
        if name == "Baseline":
            continue
        if not hasattr(model, "predict_proba"):
            continue
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC overlay to {save_path}")


def error_analysis(models, X_test, y_test, df, n_examples=10):
    """
    Pulls misclassified examples from the best model (by F1) for qualitative
    review. Requires the original cleaned dataframe to map back to text --
    note this uses index alignment from the train/test split, so it's most
    reliable when called right after train.py in the same session.
    """
    best_name = max(
        (n for n in models if n != "Baseline"),
        key=lambda n: f1_score(y_test, models[n].predict(X_test), zero_division=0),
    )
    best_model = models[best_name]
    preds = best_model.predict(X_test)

    misclassified_idx = np.where(preds != y_test)[0]
    print(f"\nBest model: {best_name} | Misclassified: {len(misclassified_idx)} / {len(y_test)}")

    sample_idx = misclassified_idx[:n_examples]
    print(f"\n--- Sample misclassified examples (first {len(sample_idx)}) ---")
    print("Note: these are positional indices into the test set (TF-IDF sparse "
          "matrix rows). Cross-reference with your test-split text export if "
          "you need the original article content for the report.")
    for i in sample_idx:
        print(f"  Row {i}: true={y_test[i]}, predicted={preds[i]}")

    return sample_idx, best_name


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading trained models and test split...")
    models, X_test, y_test = load_artifacts()

    print("\nComputing metrics table...")
    metrics_df = compute_metrics_table(models, X_test, y_test)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)

    print("\nFull classification reports:")
    for name, model in models.items():
        if name == "Baseline":
            continue
        preds = model.predict(X_test)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    plot_confusion_matrices(models, X_test, y_test,
                             os.path.join(FIGURES_DIR, "confusion_matrices.png"))
    plot_roc_overlay(models, X_test, y_test,
                      os.path.join(FIGURES_DIR, "roc_overlay.png"))

    # Error analysis (best-effort; skipped if clean_data.csv unavailable)
    clean_path = os.path.join(PROCESSED_DIR, "clean_data.csv")
    if os.path.exists(clean_path):
        df = pd.read_csv(clean_path)
        error_analysis(models, X_test, y_test, df)

    print("\nDone. Check outputs/figures/ and outputs/metrics/ for report assets.")


if __name__ == "__main__":
    main()
