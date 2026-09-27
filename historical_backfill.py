import csv
import io
import time
from datetime import datetime, timezone

import boto3
import requests


# -----------------------------
# Project configuration
# -----------------------------

BUCKET_NAME = "dtsc3601-cryptobros-project-879022683465-us-east-2-an"
AWS_PROFILE = "dtsc3601-crypto"

BINANCE_URL = "https://api.binance.us/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"

# One year of 15-minute candles
DAYS_OF_HISTORY = 365
CANDLES_PER_DAY = 24 * 4
TOTAL_CANDLES = DAYS_OF_HISTORY * CANDLES_PER_DAY

# Binance.US allows at most 1000k lines per request
REQUEST_LIMIT = 1000

INTERVAL_MS = 15 * 60_000


def milliseconds_to_utc(timestamp_ms):
    """Convert a Unix timestamp in milliseconds to readable UTC time."""

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc
    ).isoformat()


def get_time_range():
    """
    Determine the exact seven-day period to retrieve.

    The ending candle is the most recent fully completed
    one-minute candle.
    """

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    current_interval_open = (now_ms // INTERVAL_MS) * INTERVAL_MS

    latest_completed_open = current_interval_open - INTERVAL_MS

    start_time = latest_completed_open - (
        (TOTAL_CANDLES - 1) * INTERVAL_MS
    )

    end_time = latest_completed_open + INTERVAL_MS - 1

    return start_time, end_time


def fetch_historical_data():
    """Download one year of completed BTCUSDT 15-minute candles."""

    start_time, end_time = get_time_range()

    current_start = start_time

    all_candles = []

    print(
        f"Collecting {DAYS_OF_HISTORY} days of "
        f"{SYMBOL} {INTERVAL} data..."
    )

    while current_start <= end_time:

        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": current_start,
            "endTime": end_time,
            "limit": REQUEST_LIMIT
        }

        response = requests.get(
            BINANCE_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        for candle in batch:

            # Ignore anything outside our requested period
            if candle[0] < start_time:
                continue

            if candle[6] > end_time:
                continue

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

            all_candles.append(formatted_candle)

        last_open_time = batch[-1][0]

        current_start = last_open_time + INTERVAL_MS

        print(
            f"Downloaded {len(all_candles)} "
            f"of approximately {TOTAL_CANDLES} candles"
        )

        # Small delay between requests
        time.sleep(0.1)

    return all_candles


def convert_to_csv(data):
    """Convert the historical candle records to CSV text."""

    if not data:
        raise ValueError("No historical data was downloaded.")

    output = io.StringIO()

    fieldnames = list(data[0].keys())

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(data)

    return output.getvalue()


def upload_to_s3(csv_data, candle_count):
    """Upload the historical CSV dataset to the project S3 bucket."""

    session = boto3.Session(
        profile_name=AWS_PROFILE
    )

    s3 = session.client("s3")

    file_key = (
        f"historical/binance/{SYMBOL}/{INTERVAL}/"
        f"btcusdt_15m_1y.csv"
    )

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=file_key,
        Body=csv_data,
        ContentType="text/csv"
    )

    print()
    print("Upload successful.")
    print(f"Candles stored: {candle_count}")
    print(f"S3 location: s3://{BUCKET_NAME}/{file_key}")


def main():
    data = fetch_historical_data()

    print()
    print(f"Total completed candles retrieved: {len(data)}")

    csv_data = convert_to_csv(data)

    upload_to_s3(
        csv_data,
        len(data)
    )


if __name__ == "__main__":
    main()