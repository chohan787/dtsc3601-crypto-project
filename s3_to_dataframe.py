import io

import boto3
import pandas as pd


# -----------------------------
# Project configuration
# -----------------------------

BUCKET_NAME = "dtsc3601-cryptobros-project-879022683465-us-east-2-an"
AWS_PROFILE = "dtsc3601-crypto"

S3_KEY = (
    "historical/binance/BTCUSDT/1m/"
    "btcusdt_1m_7d.csv"
)


def load_data_from_s3():
    """
    Load the historical BTCUSDT dataset from S3
    into a pandas DataFrame.
    """

    session = boto3.Session(
        profile_name=AWS_PROFILE
    )

    s3 = session.client("s3")

    print("Downloading historical data from S3...")

    response = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=S3_KEY
    )

    csv_data = response["Body"].read()

    dataframe = pd.read_csv(
        io.BytesIO(csv_data)
    )

    return dataframe


def prepare_dataframe(dataframe):
    """
    Perform basic preparation so the dataset is ready
    for feature engineering and model development.
    """

    dataframe["open_time_utc"] = pd.to_datetime(
        dataframe["open_time_utc"],
        utc=True
    )

    dataframe["close_time_utc"] = pd.to_datetime(
        dataframe["close_time_utc"],
        utc=True
    )

    dataframe = dataframe.sort_values(
        by="open_time_utc"
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe


def main():
    dataframe = load_data_from_s3()

    dataframe = prepare_dataframe(
        dataframe
    )

    print()
    print("Dataset loaded successfully.")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    print()
    print("Columns:")
    print(dataframe.columns.tolist())

    print()
    print("First five rows:")
    print(dataframe.head())

    print()
    print("Data types:")
    print(dataframe.dtypes)


if __name__ == "__main__":
    main()