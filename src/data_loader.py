from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.sample_data import make_sample_data
from src.utils import project_path
from src.validation import validate_input_files

def resolve_data_paths(config: dict, use_sample: bool = False) -> tuple[Path, Path]:
    data_config = config["data"]
    if use_sample:
        train_path = project_path(data_config["sample_train_path"])
        store_path = project_path(data_config["sample_store_path"])
        if not train_path.exists() or not store_path.exists():
            make_sample_data(train_path.parent)
        return train_path, store_path

    train_path = project_path(data_config["train_path"])
    store_path = project_path(data_config["store_path"])
    if train_path.exists() and store_path.exists():
        return train_path, store_path

    print("Kaggle CSVs not found. Using generated sample data instead.")
    train_path, store_path = make_sample_data(project_path("data/sample"))
    return train_path, store_path

def load_sales_data(train_path: str | Path, store_path: str | Path) -> pd.DataFrame:
    validate_input_files(Path(train_path), Path(store_path))
    train = pd.read_csv(train_path, parse_dates=["Date"], dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(store_path)
    data = train.merge(store, on="Store", how="left")
    return data.sort_values(["Store", "Date"]).reset_index(drop=True)

def load_future_data(test_path: str | Path, store_path: str | Path) -> pd.DataFrame:
    test = pd.read_csv(test_path, parse_dates=["Date"], dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(store_path)
    data = test.merge(store, on="Store", how="left")
    return data.sort_values(["Date", "Store"]).reset_index(drop=True)
