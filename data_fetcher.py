import json
from datetime import datetime, timezone

import boto3
import requests


BUCKET_NAME = "dtsc3601-cryptobros-project-879022683465-us-east-2-an"
AWS_PROFILE = "dtsc3601-crypto"

BINANCE_URL = "https://api.binance.us/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
# The feature pipeline uses 15-minute bars and needs trailing history.
LIMIT = 120


def milliseconds_to_utc(timestamp_ms):
    """Convert a Binance millisecond timestamp into an ISO UTC string."""
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc
    ).isoformat()


def get_binance_data():
    """Fetch the latest completed BTCUSDT candles from Binance.US.

    One extra candle is requested so an in-progress candle can be removed.
    The returned rolling window gives the feature pipeline enough history for
    indicators such as RSI, moving averages, and volatility.

    Returns:
        A list of normalized candle dictionaries.

    Raises:
        requests.HTTPError: If Binance.US returns an unsuccessful response.
    """
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT + 1
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    raw_data = response.json()

    current_time_ms = int(
        datetime.now(timezone.utc).timestamp() * 1000
    )

    completed_candles = [
        candle
        for candle in raw_data
        if candle[6] < current_time_ms
    ]

    completed_candles = completed_candles[-LIMIT:]

    formatted_data = []

    for candle in completed_candles:
        formatted_candle = {
            "open_time": candle[0],
            "open_time_utc": milliseconds_to_utc(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
            "close_time": candle[6],
            "close_time_utc": milliseconds_to_utc(candle[6]),
            "quote_asset_volume": float(candle[7]),
            "number_of_trades": candle[8],
            "taker_buy_base_asset_volume": float(candle[9]),
            "taker_buy_quote_asset_volume": float(candle[10])
        }

        formatted_data.append(formatted_candle)

    return formatted_data


def upload_to_s3(data):
    """Store a live Binance response as a timestamped JSON object in S3."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client("s3")

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    file_key = f"raw/binance/{SYMBOL}/{timestamp}.json"

    file_contents = {
        "source": "Binance.US",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "ingested_at_utc": now.isoformat(),
        "candle_count": len(data),
        "candles": data
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=file_key,
        Body=json.dumps(file_contents, indent=4),
        ContentType="application/json"
    )

    print(f"Uploaded to: {file_key}")


def main():
    """Fetch recent market data and upload it to the raw S3 location."""
    data = get_binance_data()

    print(f"Retrieved {len(data)} completed candles from Binance.US")

    upload_to_s3(data)

    print("Binance data uploaded successfully")


if __name__ == "__main__":
    main()