from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_TRAIN_COLUMNS = {
    "Store",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}

REQUIRED_STORE_COLUMNS = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
}

def validate_input_files(train_path: Path, store_path: Path) -> None:
    train_columns = set(pd.read_csv(train_path, nrows=5).columns)
    store_columns = set(pd.read_csv(store_path, nrows=5).columns)

    missing_train = REQUIRED_TRAIN_COLUMNS - train_columns
    missing_store = REQUIRED_STORE_COLUMNS - store_columns

    messages = []
    if missing_train:
        messages.append(f"{train_path} is missing: {sorted(missing_train)}")
    if missing_store:
        messages.append(f"{store_path} is missing: {sorted(missing_store)}")
    if messages:
        raise ValueError("Invalid data files. " + " ".join(messages))

