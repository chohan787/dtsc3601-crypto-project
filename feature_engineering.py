"""Shared feature engineering for historical training and live inference."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ema_ratio",
    "rsi_14",
    "volatility_12",
    "volume_ratio",
    "range_pct",
    "close_position",
    "buy_volume_ratio",
    "hour_sin",
    "hour_cos",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
]


def aggregate_candles(dataframe: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    """Aggregate completed OHLCV candles into model bars."""
    required = {"open_time_utc", "open", "high", "low", "close", "volume"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    frame = dataframe.copy()
    frame["open_time_utc"] = pd.to_datetime(frame["open_time_utc"], utc=True)
    frame = frame.sort_values("open_time_utc").drop_duplicates("open_time_utc")
    frame = frame.set_index("open_time_utc")
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for column in (
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ):
        if column in frame:
            aggregations[column] = "sum"

    bars = frame.resample(timeframe, label="right", closed="right").agg(aggregations)
    bars = bars.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return bars


def create_features(
    dataframe: pd.DataFrame,
    timeframe: str = "15min",
    horizon_bars: int = 1,
    include_target: bool = True,
    flat_threshold: float = 0.001,
) -> pd.DataFrame:
    """Create leakage-safe technical and market-activity features.

    The target is -1 (down), 0 (flat), or 1 (up) based on the future return.
    The flat threshold avoids treating insignificant price noise as a signal.
    """
    bars = aggregate_candles(dataframe, timeframe=timeframe)
    close = bars["close"]
    log_return = np.log(close).diff()

    features = bars.copy()
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["return_12"] = close.pct_change(12)

    ema_fast = close.ewm(span=6, adjust=False).mean()
    ema_slow = close.ewm(span=18, adjust=False).mean()
    features["ema_ratio"] = ema_fast / ema_slow - 1

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + relative_strength))

    features["volatility_12"] = log_return.rolling(12).std()
    features["volume_ratio"] = features["volume"] / features["volume"].rolling(12).mean()
    features["range_pct"] = (features["high"] - features["low"]) / features["close"]
    features["close_position"] = (
        (features["close"] - features["low"])
        / (features["high"] - features["low"]).replace(0, np.nan)
    )
    candle_range = (features["high"] - features["low"]).replace(0, np.nan)
    features["body_pct"] = (features["close"] - features["open"]) / features["close"]
    features["upper_wick_pct"] = (
        features["high"] - features[["open", "close"]].max(axis=1)
    ) / features["close"]
    features["lower_wick_pct"] = (
        features[["open", "close"]].min(axis=1) - features["low"]
    ) / features["close"]
    if "taker_buy_base_asset_volume" in features:
        features["buy_volume_ratio"] = (
            features["taker_buy_base_asset_volume"] / features["volume"].replace(0, np.nan)
        )
    else:
        features["buy_volume_ratio"] = np.nan

    hour = features["open_time_utc"].dt.hour + features["open_time_utc"].dt.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    if include_target:
        future_return = close.shift(-horizon_bars) / close - 1
        features["target"] = np.select(
            [
                future_return > flat_threshold,
                future_return < -flat_threshold,
            ],
            [1, -1],
            default=0,
        ).astype("int8")
        features["future_return"] = future_return

    features = features.replace([np.inf, -np.inf], np.nan)
    required_features = list(FEATURE_COLUMNS)
    if include_target:
        required_features.extend(["target", "future_return"])
    return features.dropna(subset=required_features).reset_index(drop=True)


def latest_feature_row(dataframe: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    """Return one inference row with the exact training feature columns."""
    features = create_features(dataframe, timeframe=timeframe, include_target=False)
    if features.empty:
        raise ValueError("Not enough completed candles to create model features.")
    return features.tail(1)


def feature_names() -> Iterable[str]:
    return tuple(FEATURE_COLUMNS)
