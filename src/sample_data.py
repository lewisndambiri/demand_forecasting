from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir, project_path

def make_sample_data(output_dir: str | Path = "data/sample") -> tuple[Path, Path]:
    """Create a tiny Rossmann-like dataset for demos and tests."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = project_path(str(output_path))
    ensure_dir(output_path)

    dates = pd.date_range("2024-01-01", periods=140, freq="D")
    stores = pd.DataFrame(
        {
            "Store": [1, 2, 3],
            "StoreType": ["a", "b", "c"],
            "Assortment": ["a", "b", "a"],
            "CompetitionDistance": [450.0, 1200.0, np.nan],
            "CompetitionOpenSinceMonth": [5.0, np.nan, 9.0],
            "CompetitionOpenSinceYear": [2018.0, np.nan, 2020.0],
            "Promo2": [0, 1, 1],
            "Promo2SinceWeek": [np.nan, 10.0, 20.0],
            "Promo2SinceYear": [np.nan, 2021.0, 2022.0],
            "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov"],
        }
    )

    rows = []
    rng = np.random.default_rng(42)
    for store_id in stores["Store"]:
        base = 4300 + store_id * 550
        for date in dates:
            promo = int(date.day % 10 in {1, 2, 3})
            open_store = int(date.weekday() != 6)
            season = 320 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekday = 650 if date.weekday() in {4, 5} else 0
            noise = rng.normal(0, 120)
            sales = max(0, base + 520 * promo + season + weekday + noise) * open_store
            rows.append(
                {
                    "Store": store_id,
                    "DayOfWeek": date.weekday() + 1,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Sales": round(sales, 2),
                    "Customers": int(max(0, sales / 10 + rng.normal(0, 12))),
                    "Open": open_store,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": int(date.month in {4, 8, 12}),
                }
            )

    train = pd.DataFrame(rows)
    train_path = output_path / "train.csv"
    store_path = output_path / "store.csv"
    train.to_csv(train_path, index=False)
    stores.to_csv(store_path, index=False)
    return train_path, store_path

if __name__ == "__main__":
    train_path, store_path = make_sample_data()
    print(f"Wrote {train_path}")
    print(f"Wrote {store_path}")

