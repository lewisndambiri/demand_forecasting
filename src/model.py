from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def make_regressor(random_state: int = 42):
    try:
        from xgboost import XGBRegressor

        base_model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=2,
        )
        model_name = "XGBoost"
    except ImportError:
        base_model = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=random_state,
        )
        model_name = "HistGradientBoostingRegressor"

    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model_name, model

def make_baseline(random_state: int = 42):
    return RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=2,
    )

def time_based_split(
    frame: pd.DataFrame,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame["Date"].max() - pd.Timedelta(days=test_days)
    train = frame[frame["Date"] <= cutoff]
    test = frame[frame["Date"] > cutoff]
    return train, test

def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    safe_true = y_true.replace(0, np.nan)
    mape = np.nanmean(np.abs((y_true - y_pred) / safe_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mape),
        "r2": float(r2_score(y_true, y_pred)),
    }

def store_level_metrics(frame: pd.DataFrame, target: str = "Sales") -> pd.DataFrame:
    rows = []
    for store, group in frame.groupby("Store"):
        metrics = regression_metrics(group[target], group["PredictedSales"].to_numpy())
        rows.append({"Store": store, **metrics, "rows": len(group)})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def feature_importance_frame(model, columns: list[str]) -> pd.DataFrame:
    regressor = getattr(model, "regressor_", None)
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

def baseline_comparison(frame: pd.DataFrame, target: str = "Sales") -> pd.DataFrame:
    candidates = {
        "Model": "PredictedSales",
        "Naive yesterday": f"{target}_lag_1",
        "Seasonal naive last week": f"{target}_lag_7",
    }
    rows = []
    for name, column in candidates.items():
        if column not in frame.columns:
            continue
        valid = frame[[target, column]].dropna()
        if valid.empty:
            continue
        rows.append({"method": name, **regression_metrics(valid[target], valid[column].to_numpy())})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
