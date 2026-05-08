from __future__ import annotations
import pandas as pd

MONTH_TO_ABBR = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    featured = data.copy()
    dates = pd.to_datetime(featured["Date"])
    featured["year"] = dates.dt.year
    featured["quarter"] = dates.dt.quarter
    featured["month"] = dates.dt.month
    featured["day"] = dates.dt.day
    featured["day_of_year"] = dates.dt.dayofyear
    featured["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    featured["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    featured["is_month_start"] = dates.dt.is_month_start.astype(int)
    featured["is_month_end"] = dates.dt.is_month_end.astype(int)
    featured["days_since_start"] = (dates - dates.min()).dt.days
    return featured


def add_business_features(data: pd.DataFrame) -> pd.DataFrame:
    featured = data.copy()
    dates = pd.to_datetime(featured["Date"])

    if {"CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"}.issubset(featured.columns):
        competition_year = featured["CompetitionOpenSinceYear"].fillna(0).astype(int)
        competition_month = featured["CompetitionOpenSinceMonth"].fillna(0).astype(int)
        opened = (competition_year > 0) & competition_month.between(1, 12)
        opened_months = (dates.dt.year - competition_year) * 12 + (dates.dt.month - competition_month)
        featured["competition_open_months"] = opened_months.where(opened & (opened_months > 0), 0)

    if {"Promo2", "Promo2SinceYear", "Promo2SinceWeek"}.issubset(featured.columns):
        promo_year = featured["Promo2SinceYear"].fillna(0).astype(int)
        promo_week = featured["Promo2SinceWeek"].fillna(0).astype(int)
        promo_started = (featured["Promo2"].fillna(0).astype(int) == 1) & (promo_year > 0) & (promo_week > 0)
        current_week_index = dates.dt.isocalendar().year.astype(int) * 52 + dates.dt.isocalendar().week.astype(int)
        promo_week_index = promo_year * 52 + promo_week
        featured["promo2_running_weeks"] = (current_week_index - promo_week_index).where(promo_started, 0).clip(lower=0)

    if "PromoInterval" in featured.columns:
        month_abbr = dates.dt.month.map(MONTH_TO_ABBR)
        intervals = featured["PromoInterval"].fillna("None").astype(str)
        featured["is_promo2_month"] = [
            int(month != "None" and month in interval.split(","))
            for month, interval in zip(month_abbr, intervals)
        ]

    if "CompetitionDistance" in featured.columns:
        featured["has_competitor"] = (featured["CompetitionDistance"].fillna(0) > 0).astype(int)

    return featured

def add_sales_history_features(
    data: pd.DataFrame,
    target: str = "Sales",
    group_col: str = "Store",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    lags = lags or [1, 7, 14]
    rolling_windows = rolling_windows or [7, 14]

    featured = data.sort_values([group_col, "Date"]).copy()
    grouped_sales = featured.groupby(group_col)[target]

    for lag in lags:
        featured[f"{target}_lag_{lag}"] = grouped_sales.shift(lag)

    for window in rolling_windows:
        featured[f"{target}_rolling_mean_{window}"] = grouped_sales.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    return featured

def build_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    target = config["features"]["target"]
    featured = add_calendar_features(data)
    featured = add_business_features(featured)
    featured = add_sales_history_features(
        featured,
        target=target,
        lags=config["features"]["lags"],
        rolling_windows=config["features"]["rolling_windows"],
    )
    return featured.dropna(subset=[f"{target}_lag_{lag}" for lag in config["features"]["lags"]])

def make_model_frame(data: pd.DataFrame, target: str = "Sales") -> tuple[pd.DataFrame, pd.Series]:
    X = make_feature_matrix(data, target)
    y = data[target]
    return X, y

def make_feature_matrix(data: pd.DataFrame, target: str = "Sales") -> pd.DataFrame:
    ignored = {"Date", target, "Customers"}
    X = data.drop(columns=[column for column in ignored if column in data.columns])
    return pd.get_dummies(X, drop_first=False)
