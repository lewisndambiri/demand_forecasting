from __future__ import annotations
import argparse
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_future_data, load_sales_data, resolve_data_paths
from src.features import add_calendar_features, make_feature_matrix
from src.preprocessing import clean_sales_data
from src.utils import ensure_dir, load_config, project_path

def _store_history(clean_history: pd.DataFrame, target: str) -> dict[int, deque[float]]:
    histories: dict[int, deque[float]] = defaultdict(deque)
    for store, group in clean_history.sort_values(["Store", "Date"]).groupby("Store"):
        histories[int(store)] = deque(group[target].astype(float).tolist(), maxlen=120)
    return histories

def _add_history_features(
    row: pd.Series,
    sales_history: deque[float],
    target: str,
    lags: list[int],
    rolling_windows: list[int],
) -> dict:
    values = row.to_dict()
    for lag in lags:
        values[f"{target}_lag_{lag}"] = sales_history[-lag] if len(sales_history) >= lag else 0.0
    for window in rolling_windows:
        recent = list(sales_history)[-window:]
        values[f"{target}_rolling_mean_{window}"] = sum(recent) / len(recent) if recent else 0.0
    return values

def forecast_future(
    model_artifact: dict,
    history: pd.DataFrame,
    future: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    target = model_artifact["target"]
    lags = config["features"]["lags"]
    rolling_windows = config["features"]["rolling_windows"]
    model = model_artifact["model"]
    columns = model_artifact["columns"]

    future_clean = future.copy()
    future_clean["Date"] = pd.to_datetime(future_clean["Date"])
    if "Open" in future_clean.columns:
        future_clean["Open"] = future_clean["Open"].fillna(1).astype(int)
    if "PromoInterval" in future_clean.columns:
        future_clean["PromoInterval"] = future_clean["PromoInterval"].fillna("None")
    for column in [
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]:
        if column in future_clean.columns:
            future_clean[column] = future_clean[column].fillna(0)
    if "StateHoliday" in future_clean.columns:
        future_clean["StateHoliday"] = future_clean["StateHoliday"].replace({"0": "None", 0: "None"})

    future_featured = add_calendar_features(future_clean).sort_values(["Date", "Store"]).reset_index(drop=True)
    histories = _store_history(history, target)
    forecasts = []

    for _, date_rows in future_featured.groupby("Date", sort=True):
        feature_rows = []
        metadata = []
        for _, row in date_rows.iterrows():
            store = int(row["Store"])
            open_store = int(row.get("Open", 1))
            feature_rows.append(_add_history_features(row, histories[store], target, lags, rolling_windows))
            metadata.append((row, store, open_store))

        feature_frame = pd.DataFrame(feature_rows)
        X = make_feature_matrix(feature_frame, target).reindex(columns=columns, fill_value=0)
        raw_predictions = model.predict(X)

        for prediction_value, (row, store, open_store) in zip(raw_predictions, metadata):
            prediction = 0.0 if open_store == 0 else max(0.0, float(prediction_value))
            histories[store].append(prediction)
            output = {
                "Date": row["Date"],
                "Store": store,
                "PredictedSales": prediction,
                "Open": open_store,
                "Promo": int(row.get("Promo", 0)),
                "SchoolHoliday": int(row.get("SchoolHoliday", 0)),
            }
            if "Id" in row and pd.notna(row["Id"]):
                output["Id"] = int(row["Id"])
            forecasts.append(output)

    forecast = pd.DataFrame(forecasts)
    ordered = [column for column in ["Id", "Date", "Store", "PredictedSales", "Open", "Promo", "SchoolHoliday"] if column in forecast.columns]
    return forecast[ordered]

def forecast_pipeline(use_sample: bool = False) -> Path:
    config = load_config()
    model_path = project_path(config["paths"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError("Model artifact is missing. Run `make train` first.")

    artifact = joblib.load(model_path)
    train_path, store_path = resolve_data_paths(config, use_sample=use_sample)
    raw_history = load_sales_data(train_path, store_path)
    clean_history = clean_sales_data(raw_history)

    test_path = project_path(config["data"]["test_path"])
    if not test_path.exists() or use_sample:
        future = _sample_future_frame(clean_history)
    else:
        future = load_future_data(test_path, store_path)

    forecast = forecast_future(artifact, clean_history, future, config)
    output_path = project_path(config["paths"]["future_forecast_path"])
    ensure_dir(output_path.parent)
    forecast.to_csv(output_path, index=False)
    if "Id" in forecast.columns:
        submission_path = project_path(config["paths"]["kaggle_submission_path"])
        submission = (
            forecast[["Id", "PredictedSales"]]
            .rename(columns={"PredictedSales": "Sales"})
            .sort_values("Id")
        )
        submission.to_csv(submission_path, index=False)
        print(f"Saved Kaggle-style submission to {submission_path.relative_to(project_path())}")
    print(f"Saved future forecast to {output_path.relative_to(project_path())}")
    return output_path

def _sample_future_frame(history: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    stores = sorted(history["Store"].unique())
    last_date = history["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="D")
    rows = []
    store_meta = history.sort_values("Date").groupby("Store").tail(1).set_index("Store")
    row_id = 1
    for date in future_dates:
        for store in stores:
            base = store_meta.loc[store].drop(labels=["Date", "Sales", "Customers"], errors="ignore").to_dict()
            base.update(
                {
                    "Id": row_id,
                    "Store": store,
                    "DayOfWeek": date.weekday() + 1,
                    "Date": date,
                    "Open": int(date.weekday() != 6),
                    "Promo": int(date.day % 10 in {1, 2, 3}),
                    "StateHoliday": "0",
                    "SchoolHoliday": int(date.month in {4, 8, 12}),
                }
            )
            rows.append(base)
            row_id += 1
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Create future sales forecasts.")
    parser.add_argument("--sample", action="store_true", help="Use generated sample future dates.")
    args = parser.parse_args()
    forecast_pipeline(use_sample=args.sample)


if __name__ == "__main__":
    main()
