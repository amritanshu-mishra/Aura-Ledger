from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_training_data(features: pd.DataFrame, labels: pd.Series):
    """Align features and labels, discard incomplete rows, and return arrays."""
    combined = features.join(labels.rename("crash_label"), how="inner").dropna()
    if combined.empty:
        raise ValueError("No complete feature/label rows are available for training.")
    return combined[features.columns].to_numpy(), combined["crash_label"].astype(int).to_numpy()


def train_classifier(X_train, y_train):
    """Fit a scaled, imbalance-aware logistic classifier."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data needs both crash and non-crash labels.")
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ]).fit(X_train, y_train)


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Return rare-event metrics rather than misleading accuracy alone."""
    probabilities = predict_crash_probability(model, X_test)
    predictions = (probabilities >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else None
    return {"precision": float(precision_score(y_test, predictions, zero_division=0)), "recall": float(recall_score(y_test, predictions, zero_division=0)), "auc": auc, "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()}


def predict_crash_probability(model, X) -> np.ndarray:
    return model.predict_proba(np.asarray(X))[:, 1]


def save_model(model, path: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_model(path: str):
    return joblib.load(path)


if __name__ == "__main__":
    from pathlib import Path
    from src.data_loader import label_crash_windows, load_price_data
    from src.detection import compute_features
    prices = load_price_data(str(Path(__file__).resolve().parents[1] / "data" / "prices.csv"))
    X, y = prepare_training_data(compute_features(prices), label_crash_windows(prices))
    print(f"Rows: {len(X)}; crash-labelled: {y.sum()}")
