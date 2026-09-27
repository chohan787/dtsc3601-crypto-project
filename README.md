# BTCUSDT direction demo

The initial demo uses one year of BTCUSDT 15-minute Binance.US candles stored in
S3. The notebooks predict whether the close 60 minutes later will be
meaningfully down, flat, or up.

## Run the notebooks

1. Ensure the AWS profile `dtsc3601-crypto` can read the configured bucket.
2. Run `historical_backfill.py` once to download the one-year 15-minute
   dataset and upload it to S3.
3. Open `01_feature_engineering.ipynb` and run all cells. It downloads the
   historical CSV from S3 and writes a viewable copy to `data/`.
4. Open `02_modeling.ipynb` and run all cells. It trains a chronological
   `HistGradientBoostingClassifier`, performs walk-forward validation, reports
   holdout metrics, saves the model, and runs a live-style prediction using a
   rolling window of completed Binance.US candles.

The same functions in `feature_engineering.py` are used for training and
inference, preventing the live request from receiving a different feature
layout than the model saw during training.

The flat threshold is a return threshold, not a fixed dollar amount: `0.002`
means 0.20% of the current BTC price. At a price of $84,000, that is about
$168. A 60-minute return above +0.20% is UP, below -0.20% is DOWN, and the
range between them is FLAT.

Direction classification is recommended over exact-price regression for this
demo: the label aligns with the assignment's evaluation and avoids pretending
that a precise short-term price is predictable. The three-class label includes
a neutral range so insignificant movements are not forced into UP or DOWN.

Live inference intentionally fetches a rolling history rather than one candle.
Trailing indicators require prior observations; the pipeline computes those
values from the returned window and sends only the latest feature row to the
model. The training and inference paths use the same feature function and
column list.
