import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def load_data_and_models(processed_dir='data/processed', models_dir='outputs/models'):
    print("Loading test data and models...")

    # Load test set
    X_test = load_npz(os.path.join(processed_dir, 'X_test.npz'))
    y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))

    # Load original texts for error analysis
    test_indices = np.load(os.path.join(processed_dir, 'test_indices.npy'))
    df_all = pd.read_csv(os.path.join(processed_dir, 'processed_data.csv'))
    df_test = df_all.iloc[test_indices].copy()

    models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl') and filename != 'tfidf_vectorizer.pkl':
            model_name = filename.replace('.pkl', '')
            with open(os.path.join(models_dir, filename), 'rb') as f:
                models[model_name] = pickle.load(f)

    return X_test, y_test, df_test, models

def evaluate_models():
    print("Starting evaluation...")

    X_test, y_test, df_test, models = load_data_and_models()

    metrics_records = []

    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Check if we have both classes in y_test for ROC curve
    unique_classes = np.unique(y_test)
    can_plot_roc = len(unique_classes) > 1

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)

        # In small dummy sets, classes might be missing. Use zero_division=0.
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics_records.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'outputs/figures/cm_{name}.png')
        plt.close()

        # ROC Curve (if model has predict_proba and both classes are in test set)
        if can_plot_roc and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Switch back to the ROC plot
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

        # Error Analysis (save sample misclassifications for best model)
        if name == 'LogReg': # Or any representative model
            df_test['predicted_label'] = y_pred
            errors = df_test[df_test['label'] != df_test['predicted_label']]
            if not errors.empty:
                errors[['title', 'text', 'label', 'predicted_label']].head(10).to_csv('outputs/metrics/sample_errors.csv', index=False)
                print(f"Saved {len(errors)} misclassifications for error analysis.")
            else:
                print("No misclassifications for this model on test set.")

    # Finish ROC Plot
    if can_plot_roc:
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('outputs/figures/roc_curve_overlay.png')
    plt.close()

    # Save Metrics Table
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df = metrics_df.sort_values(by='F1-Score', ascending=False)
    metrics_df.to_csv('outputs/metrics/model_comparison.csv', index=False)

    print("\n--- Model Comparison ---")
    print(metrics_df.to_string(index=False))
    print("\nEvaluation complete. Metrics and figures saved.")

if __name__ == "__main__":
    evaluate_models()
