import numpy as np
import os
import pickle
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_models(processed_dir='data/processed', output_dir='outputs/models'):
    print("Starting model training...")

    # Load features and labels
    features_path = os.path.join(processed_dir, 'features_tfidf.npz')
    labels_path = os.path.join(processed_dir, 'labels.npy')

    X = load_npz(features_path)
    y = np.load(labels_path)

    print(f"Loaded TF-IDF features of shape {X.shape}")
    print(f"Loaded labels of shape {y.shape}")

    # Train/Test Split
    # Disable stratification if dataset is too small (e.g. dummy data)
    stratify = y if len(y) > 5 else None
    test_size = 0.2 if len(y) > 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Better: save train/test sets to ensure exact match in evaluate.py
    os.makedirs(processed_dir, exist_ok=True)
    from scipy.sparse import save_npz
    save_npz(os.path.join(processed_dir, 'X_test.npz'), X_test)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)

    # For evaluate.py error analysis, we also need raw text. We'll load the full DF and grab by split index.
    # To do this robustly, we use index splitting.
    indices = np.arange(len(y))
    idx_train, idx_test, _, _ = train_test_split(indices, y, test_size=test_size, random_state=42, stratify=stratify)
    np.save(os.path.join(processed_dir, 'test_indices.npy'), idx_test)

    models = {}

    # 0. Baseline (Majority Class)
    print("\nTraining DummyClassifier (Baseline)...")
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    models['Baseline'] = baseline
    print(f"Baseline Train Acc: {accuracy_score(y_train, baseline.predict(X_train)):.4f}")

    # 1. KNN
    print("\nTraining KNeighborsClassifier...")
    knn = KNeighborsClassifier(n_neighbors=min(5, len(y_train))) # Handling small dummy dataset size
    knn.fit(X_train, y_train)
    models['KNN'] = knn

    # Setup CV
    # Use standard KFold if classes per fold is too small for small dataset (like n_splits=2 for dummy data)
    n_splits = min(5, np.min(np.bincount(y_train)))
    if n_splits < 2:
         cv = 2
    else:
         cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 2. Logistic Regression with GridSearchCV
    print("\nTraining Logistic Regression with GridSearchCV...")
    lr = LogisticRegression(solver='liblinear', random_state=42)
    # Simplified grid for fast execution, can be expanded
    lr_params = {'C': [0.01, 0.1, 1, 10]}

    # Fallback to simple fit if dummy dataset is too small for CV
    if len(y_train) < 4 or len(np.unique(y_train)) < 2:
        lr_best = lr.set_params(C=1.0)
        # If there's only 1 class in small y_train due to split randomness
        if len(np.unique(y_train)) < 2:
            print("Warning: Only 1 class in y_train. Forcing dummy train labels for testing.")
            y_train[0] = 1 - y_train[0]
        lr_best.fit(X_train, y_train)
    else:
        lr_grid = GridSearchCV(lr, lr_params, cv=cv, scoring='f1', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        lr_best = lr_grid.best_estimator_
        print(f"Best LR params: {lr_grid.best_params_}")
    models['LogReg'] = lr_best

    # 3. Random Forest with GridSearchCV
    print("\nTraining Random Forest with GridSearchCV...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }

    if len(y_train) < 4:
        rf_best = rf.set_params(n_estimators=50)
        rf_best.fit(X_train, y_train)
    else:
        rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        rf_best = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}")
    models['RandomForest'] = rf_best

    # 4. MLP Classifier
    print("\nTraining MLPClassifier...")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    models['MLP'] = mlp

    # Save all models
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model.")

    print("\nModel training complete.")

if __name__ == "__main__":
    train_models()
