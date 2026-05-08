# Results

These results were generated after training on the real Rossmann Store Sales files in `data/train.csv` and `data/store.csv`.

## Run Summary

| Item | Value |
|---|---:|
| Model | XGBoost |
| Training rows | 784,097 |
| Holdout rows | 26,845 |
| Stores | 1,115 |
| Date range | 2013-01-31 to 2015-07-31 |

## Holdout Metrics

| Metric | Value |
|---|---:|
| RMSE | 861.246 |
| MAE | 620.781 |
| MAPE | 9.225% |
| R2 | 0.914 |

## Baseline Comparison

| Method | RMSE | MAE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Model | 861.246 | 620.781 | 9.225% | 0.914 |
| Naive yesterday | 1,934.502 | 1,295.557 | 19.944% | 0.568 |
| Seasonal naive last week | 2,915.972 | 2,304.144 | 36.533% | 0.018 |

## Interpretation

The machine learning model performs much better than both simple forecasting baselines. This matters because naive forecasts are often surprisingly hard to beat in time series projects. Here, the model cuts RMSE by more than half compared with the yesterday baseline and also improves MAPE substantially.

The result suggests that the added features are useful: promotions, calendar features, store metadata, lags, and rolling averages collectively explain demand better than simply repeating previous sales.

Adding 30-day lag and rolling features improved the real-data holdout compared with the earlier shorter-memory setup. Switching the active estimator from scikit-learn's histogram gradient boosting fallback to XGBoost improved the result further:

| Model | RMSE | MAE | MAPE | R2 |
|---|---:|---:|---:|---:|
| HistGradientBoostingRegressor | 889.786 | 637.584 | 9.523% | 0.909 |
| XGBoost | 861.246 | 620.781 | 9.225% | 0.914 |

This is a useful modeling insight: the pipeline was already strong with scikit-learn, but XGBoost handled the tabular feature interactions better on this holdout.

## Forecast Artifacts

The latest run produced:

```text
models/holdout_predictions.csv
models/store_metrics.csv
models/baseline_comparison.csv
models/future_forecast.csv
models/kaggle_submission.csv
```

`future_forecast.csv` is useful for dashboard exploration. `kaggle_submission.csv` is shaped for Kaggle-style submission with `Id` and `Sales` columns.

## Caveat

These are local holdout results, not official Kaggle leaderboard results. The holdout split is time-based and realistic for evaluation, but the final Kaggle score may differ because Kaggle evaluates against its hidden test labels.
