from pathlib import Path
import pandas as pd
import pytest

from src.validation import validate_input_files

def test_validate_input_files_accepts_expected_schema(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    store_path = tmp_path / "store.csv"

    pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2024-01-01"],
            "Sales": [100],
            "Customers": [10],
            "Open": [1],
            "Promo": [0],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
        }
    ).to_csv(train_path, index=False)

    pd.DataFrame(
        {
            "Store": [1],
            "StoreType": ["a"],
            "Assortment": ["a"],
            "CompetitionDistance": [100],
            "Promo2": [0],
        }
    ).to_csv(store_path, index=False)

    validate_input_files(train_path, store_path)

def test_validate_input_files_reports_missing_columns(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    store_path = tmp_path / "store.csv"

    pd.DataFrame({"Store": [1], "Date": ["2024-01-01"]}).to_csv(train_path, index=False)
    pd.DataFrame({"Store": [1]}).to_csv(store_path, index=False)

    with pytest.raises(ValueError, match="missing"):
        validate_input_files(train_path, store_path)

