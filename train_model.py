from __future__ import annotations

import json
from pathlib import Path

from src.classifier import evaluate_classifier, prepare_training_data, save_model, train_classifier
from src.data_loader import label_crash_windows, load_price_data
from src.detection import compute_features


def main() -> None:
    root = Path(__file__).resolve().parent
    prices = load_price_data(str(root / "data" / "prices.csv"))
    features = compute_features(prices)
    labels = label_crash_windows(prices)
    aligned = features.join(labels.rename("crash_label"), how="inner").dropna()
    X, y = prepare_training_data(features, labels)
    # A chronological holdout avoids training on future market regimes.
    split = int(len(aligned) * 0.75)
    X_train = aligned.iloc[:split][features.columns].to_numpy()
    y_train = aligned.iloc[:split]["crash_label"].astype(int).to_numpy()
    X_test = aligned.iloc[split:][features.columns].to_numpy()
    y_test = aligned.iloc[split:]["crash_label"].astype(int).to_numpy()
    evaluation_model = train_classifier(X_train, y_train)
    metrics = evaluate_classifier(evaluation_model, X_test, y_test)
    # Persist a full-history model for local dashboard inference only after reporting the holdout result.
    model = train_classifier(X, y)
    save_model(model, str(root / "models" / "crash_classifier.pkl"))
    report = {"evaluation_protocol": "chronological 75/25 holdout", "train_end_date": str(aligned.index[split - 1].date()), "test_start_date": str(aligned.index[split].date()), "metrics": metrics}
    (root / "models" / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Aura Ledger classifier trained and saved (chronological holdout evaluation).")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"AUC:       {metrics['auc']:.3f}" if metrics["auc"] is not None else "AUC: unavailable (single-class test set)")
    print(f"Confusion matrix [TN, FP; FN, TP]: {metrics['confusion_matrix']}")
    print(f"Holdout: train through {report['train_end_date']}; test from {report['test_start_date']}")


if __name__ == "__main__":
    main()
