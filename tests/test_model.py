import pandas as pd
from src.model import baseline_comparison

def test_baseline_comparison_includes_naive_methods():
    frame = pd.DataFrame(
        {
            "Sales": [100, 120, 140],
            "PredictedSales": [102, 118, 139],
            "Sales_lag_1": [90, 100, 120],
            "Sales_lag_7": [80, 115, 130],
        }
    )

    comparison = baseline_comparison(frame)

    assert set(comparison["method"]) == {"Model", "Naive yesterday", "Seasonal naive last week"}
    assert comparison.iloc[0]["method"] == "Model"
