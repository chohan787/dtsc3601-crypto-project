"""Run one live-style BTCUSDT direction prediction.

This command fetches recent completed Binance.US candles, loads the trained
model saved by the modeling notebook, and prints the latest prediction.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_fetcher import get_binance_data
from modeling import predict_latest
from s3_to_dataframe import prepare_dataframe


MODEL_PATH = PROJECT_ROOT / "models" / "btcusdt_direction_15m_60m.joblib"


def load_trained_model(model_path: Path = MODEL_PATH):
    """Load the trained classifier package from a joblib file.

    Args:
        model_path: Path to the model produced by ``02_modeling.ipynb``.

    Returns:
        The fitted classifier stored in the model package.

    Raises:
        FileNotFoundError: If the modeling notebook has not saved the model.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 02_modeling.ipynb first."
        )
    return joblib.load(model_path)["model"]


def main():
    """Fetch current data and print a DOWN, FLAT, or UP prediction."""
    model = load_trained_model()
    candles = prepare_dataframe(pd.DataFrame(get_binance_data()))
    prediction = predict_latest(model, candles, timeframe="15min")

    print(f"Completed bar: {prediction['bar_time_utc']}")
    print(f"Current close: ${prediction['close']:,.2f}")
    print(f"Predicted direction in 60 minutes: {prediction['prediction']}")
    print("Class probabilities:")
    for label, probability in prediction["probabilities"].items():
        print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()
