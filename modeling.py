"""Train, evaluate, save, and use the direction classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_engineering import FEATURE_COLUMNS, latest_feature_row


def train_direction_model(
    feature_frame: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Train with a chronological split; do not shuffle time-series data."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    split = int(len(feature_frame) * (1 - test_fraction))
    if split <= 0 or split >= len(feature_frame):
        raise ValueError("Feature frame is too small for the requested split.")

    train = feature_frame.iloc[:split]
    test = feature_frame.iloc[split:]
    model = _make_model(random_state)
    model.fit(train[FEATURE_COLUMNS], train["target"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    metrics = _metrics(test["target"], predictions, len(train), len(test))
    return model, metrics


def _make_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=random_state,
    )


def _metrics(actual, predictions, train_rows: int, test_rows: int) -> dict:
    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": float(accuracy_score(actual, predictions)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[-1, 0, 1]).tolist(),
        "classification_report": classification_report(
            actual, predictions, labels=[-1, 0, 1], output_dict=True, zero_division=0
        ),
    }


def walk_forward_validate(
    feature_frame: pd.DataFrame,
    folds: int = 5,
    min_train_fraction: float = 0.5,
    test_fraction: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Evaluate expanding-window forecasts without training on future rows."""
    if folds < 2 or not 0 < min_train_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Invalid walk-forward split settings.")
    min_train = int(len(feature_frame) * min_train_fraction)
    test_size = max(1, int(len(feature_frame) * test_fraction))
    results = []
    for fold in range(folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size
        if test_end > len(feature_frame):
            break
        train = feature_frame.iloc[:train_end]
        test = feature_frame.iloc[train_end:test_end]
        model = _make_model(random_state)
        model.fit(train[FEATURE_COLUMNS], train["target"])
        predictions = model.predict(test[FEATURE_COLUMNS])
        metrics = _metrics(test["target"], predictions, len(train), len(test))
        metrics["fold"] = fold + 1
        results.append(metrics)
    if len(results) < 2:
        raise ValueError("Not enough rows for the requested walk-forward folds.")
    return {
        "folds": results,
        "mean_accuracy": float(sum(item["accuracy"] for item in results) / len(results)),
    }


def save_model(model, metrics: dict, path: str | Path) -> None:
    """Persist the estimator and its feature contract for the demo."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, output)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def predict_latest(model, candles: pd.DataFrame, timeframe: str = "15min") -> dict:
    """Generate a live-style prediction from the latest completed candles."""
    row = latest_feature_row(candles, timeframe=timeframe)
    probabilities = model.predict_proba(row[FEATURE_COLUMNS])[0]
    probability_by_class = dict(zip(model.classes_, probabilities))
    predicted_class = int(model.predict(row[FEATURE_COLUMNS])[0])
    return {
        "prediction": {-1: "DOWN", 0: "FLAT", 1: "UP"}[predicted_class],
        "probabilities": {
            {-1: "DOWN", 0: "FLAT", 1: "UP"}[label]: float(probability_by_class.get(label, 0.0))
            for label in (-1, 0, 1)
        },
        "bar_time_utc": row["open_time_utc"].iloc[0].isoformat(),
        "close": float(row["close"].iloc[0]),
    }
