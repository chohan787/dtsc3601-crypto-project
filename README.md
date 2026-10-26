# BTCUSDT Direction Prediction

This project collects BTCUSDT market data from Binance.US, stores historical
data in Amazon S3, creates prediction features, and trains a model to predict
whether BTCUSDT will move **up**, **down**, or remain **flat**.

The project is an extension of the initial data-ingestion work. It now includes
the feature pipeline, model training, validation, and a live-style prediction
example.

## What the model predicts

The model uses completed 15-minute candles and predicts the direction of the
price 60 minutes later:

| Class | Meaning |
| --- | --- |
| `UP` | Price increases by more than 0.20% |
| `DOWN` | Price decreases by more than 0.20% |
| `FLAT` | Price changes by no more than 0.20% |

The 0.20% value is a return threshold based on the current price. For example,
0.20% of a BTC price of $84,000 is about $168.

## Repository guide

| Path | Description |
| --- | --- |
| [`historical_backfill.py`](./historical_backfill.py) | Downloads one year of completed 15-minute candles and uploads a CSV to S3 |
| [`s3_to_dataframe.py`](./s3_to_dataframe.py) | Loads the historical S3 CSV into pandas and prepares timestamps |
| [`data_fetcher.py`](./data_fetcher.py) | Gets a rolling window of recent completed Binance.US candles |
| [`feature_engineering.py`](./feature_engineering.py) | Creates the same features for training and prediction |
| [`modeling.py`](./modeling.py) | Trains, validates, saves, and uses the classifier |
| [`01_feature_engineering.ipynb`](./01_feature_engineering.ipynb) | Shows S3 loading and feature creation |
| [`02_modeling.ipynb`](./02_modeling.ipynb) | Shows validation, training, and live-style prediction |
| [`bin/live_prediction.py`](./bin/live_prediction.py) | Command-line example for a live-style prediction |
| [`data/sample_btcusdt_15m.csv`](./data/sample_btcusdt_15m.csv) | Small sample file for the quick-start example |
| [`requirements.txt`](./requirements.txt) | Python packages required by the project |

## Install

### Requirements

- Python 3
- An AWS profile named `dtsc3601-crypto` with access to the project S3 bucket
- Internet access for Binance.US and AWS

### Install the required libraries

From the repository folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main libraries used are:

- `pandas` for tabular market data
- `numpy` for numerical calculations
- `boto3` for Amazon S3
- `requests` for Binance.US API requests
- `scikit-learn` for the classifier and evaluation
- `joblib` for saving the trained model
- `jupyter` for running the notebooks

## Quick start: train on the small sample

The repository includes a small CSV so the project can be tested without
downloading one year of data. Run this from the repository folder:

```powershell
python -c "import pandas as pd; from feature_engineering import create_features; from modeling import train_direction_model; candles = pd.read_csv('data/sample_btcusdt_15m.csv'); features = create_features(candles, timeframe='15min', horizon_bars=4, flat_threshold=0.002); model, metrics = train_direction_model(features); print(f'Rows: {len(features)}'); print(f'Accuracy: {metrics[\"accuracy\"]:.3f}')"
```

This example:

1. Reads the checked-in sample CSV.
2. Creates the same 15-minute features used by the full project.
3. Creates a 60-minute, three-class target.
4. Trains the classifier on a chronological split.
5. Prints the test accuracy.

The sample is for confirming that the code runs. It is too small to represent
the full model's expected performance.

## Store the full historical dataset in S3

The historical backfill downloads one year of completed BTCUSDT 15-minute
candles in batches and stores them at:

```text
s3://dtsc3601-cryptobros-project-879022683465-us-east-2-an/
historical/binance/BTCUSDT/15m/btcusdt_15m_1y.csv
```

Run:

```powershell
python historical_backfill.py
```

The S3 profile and bucket are configured in
[`historical_backfill.py`](./historical_backfill.py) and
[`s3_to_dataframe.py`](./s3_to_dataframe.py). Do not commit AWS credentials,
access keys, or secret files.

## Run the notebooks

### Feature engineering

Open [`01_feature_engineering.ipynb`](./01_feature_engineering.ipynb) and run
all cells. It downloads the S3 CSV, loads it into pandas, saves a local copy,
and displays the raw candles and engineered features.

### Modeling

Open [`02_modeling.ipynb`](./02_modeling.ipynb), restart the kernel, and run all
cells. It:

1. Loads the historical data from S3.
2. Creates the target and feature columns.
3. Uses chronological training and testing.
4. Runs walk-forward validation.
5. Displays accuracy and the classification report.
6. Saves the trained model under `models/`.
7. Fetches recent Binance.US data and prints a live-style prediction.

## Main functions

### Data ingestion

- `historical_backfill.get_time_range()` calculates the one-year time range
  ending at the latest completed 15-minute candle.
- `historical_backfill.fetch_historical_data()` requests the data in batches,
  removes candles outside the requested period, and returns normalized records.
- `historical_backfill.convert_to_csv()` converts those records into CSV text.
- `historical_backfill.upload_to_s3()` writes the CSV to the configured S3 key.
- `data_fetcher.get_binance_data()` retrieves a recent rolling window for
  prediction and excludes an incomplete candle.
- `data_fetcher.upload_to_s3()` stores a live API response as timestamped JSON.

### Data preparation and features

- `s3_to_dataframe.load_data_from_s3()` downloads the historical CSV into a
  pandas DataFrame.
- `s3_to_dataframe.prepare_dataframe()` parses UTC timestamps, sorts the
  records, and returns a clean copy.
- `feature_engineering.aggregate_candles()` creates the model's 15-minute bars
  from candle data.
- `feature_engineering.create_features()` calculates momentum, trend, RSI,
  volatility, volume, candle-shape, and time-of-day features. It also creates
  the optional `DOWN`, `FLAT`, and `UP` target.
- `feature_engineering.latest_feature_row()` calculates features from the
  rolling live window and returns only the newest row for prediction.
- `feature_engineering.feature_names()` returns the feature names in the
  order expected by the model.

### Training and prediction

- `modeling.train_direction_model()` trains with an earlier-period training
  set and a later-period test set.
- `modeling.walk_forward_validate()` repeats training and testing across
  expanding time windows without using future rows.
- `modeling.save_model()` saves the classifier, feature-column order, and
  evaluation metrics.
- `modeling.predict_latest()` returns the predicted class, probabilities,
  timestamp, and current close price.

All functions include docstrings in their Python files with more detail about
their arguments, return values, and errors.

## Run the live prediction example

Run the model notebook first so the trained model exists under `models/`.
Then run:

```powershell
python bin/live_prediction.py
```

The script calls Binance.US, fetches recent completed candles, calculates the
latest feature row, and prints:

```text
Completed bar: ...
Current close: $...
Predicted direction in 60 minutes: UP/DOWN/FLAT
Class probabilities:
  DOWN: ...
  FLAT: ...
  UP: ...
```

The API returns a rolling window rather than one candle because moving
averages, RSI, volatility, and volume features need historical values.

## Questions

Contact **Aashutosh Panchal**:

- GitHub: [Aashu-P](https://github.com/Aashu-P)
- Email: [aashupanchal2006@gmail.com](mailto:aashupanchal2006@gmail.com)

## License

This project is available under the [GNU General Public License v3.0](./LICENSE).
