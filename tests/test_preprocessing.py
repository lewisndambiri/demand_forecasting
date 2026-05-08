import numpy as np
import pandas as pd
from src.preprocessing import clean_sales_data

def test_clean_sales_data_fills_missing_store_values():
    data = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2024-01-01", "2024-01-02"],
            "Sales": [100, 200],
            "Open": [1, 1],
            "StateHoliday": ["0", 0],
            "CompetitionDistance": [np.nan, 250.0],
            "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct"],
        }
    )

    cleaned = clean_sales_data(data)

    assert cleaned["CompetitionDistance"].isna().sum() == 0
    assert cleaned["PromoInterval"].iloc[0] == "None"
    assert cleaned["StateHoliday"].tolist() == ["None", "None"]

def test_clean_sales_data_removes_closed_days():
    data = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2024-01-01", "2024-01-02"],
            "Sales": [0, 200],
            "Open": [0, 1],
        }
    )

    cleaned = clean_sales_data(data)

    assert len(cleaned) == 1
    assert cleaned["Sales"].iloc[0] == 200
