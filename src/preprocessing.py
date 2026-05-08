from __future__ import annotations
import pandas as pd

NUMERIC_DEFAULTS = {
    "CompetitionDistance": 0,
    "CompetitionOpenSinceMonth": 0,
    "CompetitionOpenSinceYear": 0,
    "Promo2SinceWeek": 0,
    "Promo2SinceYear": 0,
}

def clean_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])

    for column, default in NUMERIC_DEFAULTS.items():
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(default)

    if "PromoInterval" in cleaned.columns:
        cleaned["PromoInterval"] = cleaned["PromoInterval"].fillna("None")

    if "StateHoliday" in cleaned.columns:
        cleaned["StateHoliday"] = cleaned["StateHoliday"].replace({"0": "None", 0: "None"})

    if "Open" in cleaned.columns:
        cleaned = cleaned[cleaned["Open"].fillna(1).astype(int) == 1]

    if "Sales" in cleaned.columns:
        cleaned = cleaned[cleaned["Sales"] >= 0]

    return cleaned.sort_values(["Store", "Date"]).reset_index(drop=True)

