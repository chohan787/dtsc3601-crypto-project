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
    """Train and evaluate a classifier using a chronological holdout.

    The earlier rows are used for training and the later rows are reserved for
    testing. Keeping the time order prevents information from the future from
    entering the training set.

    Args:
        feature_frame: Model-ready rows containing feature columns and target.
        test_fraction: Fraction of the most recent rows reserved for testing.
        random_state: Seed used by the classifier for reproducibility.

    Returns:
        The fitted classifier and a dictionary of evaluation metrics.
    """
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
    """Create the project's configured gradient-boosting classifier."""
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=random_state,
    )


def _metrics(actual, predictions, train_rows: int, test_rows: int) -> dict:
    """Calculate accuracy and class-by-class evaluation results."""
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
    """Evaluate forecasts with expanding chronological training windows.

    Each fold trains on all available earlier rows and tests on the next block.
    This better represents repeated future predictions than a single split and
    ensures that no fold trains on data from its own future.

    Args:
        feature_frame: Time-ordered model-ready rows.
        folds: Maximum number of sequential evaluation windows.
        min_train_fraction: Initial fraction used for the first training set.
        test_fraction: Size of each test window as a fraction of all rows.
        random_state: Seed used for each fold's classifier.

    Returns:
        A dictionary containing per-fold metrics and their mean accuracy.
    """
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
    """Save the trained model, feature order, and metrics to disk.

    The feature contract is saved beside the estimator so a future prediction
    can use the same columns and ordering as training.

    Args:
        model: Fitted scikit-learn classifier.
        metrics: JSON-serializable evaluation results.
        path: Output path for the joblib model file.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, output)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def predict_latest(model, candles: pd.DataFrame, timeframe: str = "15min") -> dict:
    """Predict DOWN, FLAT, or UP from the newest completed candle window.

    Historical candles are transformed with the shared feature pipeline before
    the newest row is sent to the classifier. The returned probabilities are
    labeled by class so the output is easy to display in the demo.

    Args:
        model: Fitted classifier with ``predict`` and ``predict_proba`` methods.
        candles: Recent completed Binance candle records.
        timeframe: Interval used when the model was trained.

    Returns:
        A dictionary containing the prediction, class probabilities, timestamp,
        and latest close price.
    """
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
