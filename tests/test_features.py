import pandas as pd

from src.features import (
    add_business_features,
    add_calendar_features,
    add_sales_history_features,
    make_feature_matrix,
    make_model_frame,
)

def test_add_calendar_features_marks_weekend():
    data = pd.DataFrame({"Date": pd.to_datetime(["2024-01-06", "2024-01-08"])})

    featured = add_calendar_features(data)

    assert featured["is_weekend"].tolist() == [1, 0]
    assert "week_of_year" in featured.columns
    assert "quarter" in featured.columns
    assert "day_of_year" in featured.columns

def test_add_business_features_adds_competition_and_promo2_context():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-15"]),
            "CompetitionDistance": [250.0],
            "CompetitionOpenSinceMonth": [1],
            "CompetitionOpenSinceYear": [2023],
            "Promo2": [1],
            "Promo2SinceWeek": [1],
            "Promo2SinceYear": [2023],
            "PromoInterval": ["Jan,Apr,Jul,Oct"],
        }
    )

    featured = add_business_features(data)

    assert featured["has_competitor"].iloc[0] == 1
    assert featured["competition_open_months"].iloc[0] >= 12
    assert featured["promo2_running_weeks"].iloc[0] > 0
    assert featured["is_promo2_month"].iloc[0] == 1

def test_sales_lag_uses_previous_store_sales_only():
    data = pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "Sales": [100, 150, 300, 330],
        }
    )

    featured = add_sales_history_features(data, lags=[1], rolling_windows=[2])

    assert pd.isna(featured.loc[featured["Store"] == 1, "Sales_lag_1"].iloc[0])
    assert featured.loc[featured["Store"] == 1, "Sales_lag_1"].iloc[1] == 100
    assert featured.loc[featured["Store"] == 2, "Sales_lag_1"].iloc[1] == 300

def test_make_model_frame_removes_future_unknown_customers():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "Sales": [100],
            "Customers": [10],
            "StoreType": ["a"],
            "Promo": [1],
        }
    )

    X, y = make_model_frame(data)

    assert "Customers" not in X.columns
    assert "Sales" not in X.columns
    assert y.iloc[0] == 100

def test_make_feature_matrix_supports_rows_without_target():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "Store": [1],
            "Promo": [1],
            "StoreType": ["a"],
        }
    )

    X = make_feature_matrix(data)

    assert "Date" not in X.columns
    assert "Promo" in X.columns
    assert "StoreType_a" in X.columns
