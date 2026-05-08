# Technical Notes

## Data

The project expects the Rossmann-style columns from Kaggle:

- `train.csv`: `Store`, `Date`, `Sales`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`
- `store.csv`: `Store`, `StoreType`, `Assortment`, competition columns, and promo interval columns

For convenience, `src/sample_data.py` creates a small matching dataset under `data/sample/`.

## Preprocessing

Missing competition and promo fields are filled with simple defaults. `StateHoliday` value `0` is normalized to `None`. Closed-store rows are removed because zero demand from closure is a different business problem from open-store sales demand.

## Features

Calendar features capture normal seasonality:

- year, month, day
- quarter, day of year, days since dataset start
- ISO week number
- weekend, month-start, month-end flags

Business features capture domain context:

- whether a competitor is present
- how many months a competitor has been open
- how many weeks Promo2 has been running
- whether the current month is part of the Promo2 interval

Sales history features are grouped by store:

- lagged sales for previous 1, 7, and 14 days
- rolling means for previous 7 and 14 days

Rolling features shift by one day before averaging, so the current day's sales are never used to predict itself.

## Validation

The split is time-based. The latest `test_days` from `config/config.yaml` are held out for evaluation. This is closer to real forecasting than random train/test splitting.

The project also saves a baseline comparison:

- naive forecast: use yesterday's sales
- seasonal naive forecast: use the same store's sales from seven days ago
- machine learning forecast: use the trained regression model

This comparison is important because a forecasting model should beat simple rules before it is considered useful.

## Future Forecasting

`src/forecast.py` creates predictions for rows where future sales are unknown. With the Rossmann dataset, it uses `data/test.csv`.

The forecast loop is recursive:

1. Start with known historical sales.
2. Forecast the first future date.
3. Add that prediction into the store's sales history.
4. Use it to calculate lag and rolling features for later dates.
5. Repeat until all future rows are predicted.

If a future row has `Open = 0`, the forecast is set to zero.

## Model

The active model is XGBoost, wrapped in a log transform for the sales target. If XGBoost is unavailable, the project falls back to `HistGradientBoostingRegressor`.

The latest local run uses XGBoost. See `docs/MODEL_EXPLAINER.md` for a plain-English comparison with the scikit-learn fallback.

Metrics saved to `models/metrics.json`:

- RMSE
- MAE
- MAPE
- R2

## Improvements

Good next steps would be richer holiday features, experiment tracking, SHAP explanations, Docker, and a small API endpoint for serving predictions.
