import pandas as pd
from src.forecast import forecast_future

class ConstantModel:
    def predict(self, X):
        return [100.0] * len(X)

def test_forecast_future_sets_closed_days_to_zero():
    config = {"features": {"target": "Sales", "lags": [1], "rolling_windows": [2]}}
    artifact = {
        "target": "Sales",
        "model": ConstantModel(),
        "columns": ["Store", "Open", "Promo", "Sales_lag_1", "Sales_rolling_mean_2"],
    }
    history = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Sales": [70, 80],
        }
    )
    future = pd.DataFrame(
        {
            "Id": [1, 2],
            "Store": [1, 1],
            "Date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "Open": [1, 0],
            "Promo": [0, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0],
        }
    )

    forecast = forecast_future(artifact, history, future, config)

    assert forecast["PredictedSales"].tolist() == [100.0, 0.0]
