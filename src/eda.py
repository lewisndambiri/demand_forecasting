from __future__ import annotations
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib")))
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_sales_data, resolve_data_paths
from src.preprocessing import clean_sales_data
from src.utils import ensure_dir, load_config, project_path

def save_plot(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def run_eda(use_sample: bool = False) -> Path:
    config = load_config()
    train_path, store_path = resolve_data_paths(config, use_sample=use_sample)
    raw = load_sales_data(train_path, store_path)
    clean = clean_sales_data(raw)

    plots_dir = ensure_dir(project_path(config["paths"]["plots_dir"]))
    report_path = project_path("docs/EDA_REPORT.md")

    daily_sales = clean.groupby("Date")["Sales"].sum()
    plt.figure(figsize=(12, 5))
    daily_sales.plot(color="#2563eb", linewidth=1.5)
    plt.title("Total Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    save_plot(plots_dir / "daily_sales.png")

    weekday_sales = clean.groupby("DayOfWeek")["Sales"].mean()
    plt.figure(figsize=(8, 4))
    weekday_sales.plot(kind="bar", color="#16a34a")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of week")
    plt.ylabel("Average sales")
    save_plot(plots_dir / "sales_by_weekday.png")

    promo_sales = clean.groupby("Promo")["Sales"].mean()
    plt.figure(figsize=(7, 4))
    promo_sales.plot(kind="bar", color=["#64748b", "#ea580c"])
    plt.title("Average Sales: Promo vs Non-Promo")
    plt.xlabel("Promo")
    plt.ylabel("Average sales")
    save_plot(plots_dir / "promo_lift.png")

    top_stores = clean.groupby("Store")["Sales"].sum().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    top_stores.sort_values().plot(kind="barh", color="#7c3aed")
    plt.title("Top 15 Stores by Total Sales")
    plt.xlabel("Sales")
    save_plot(plots_dir / "top_stores.png")

    promo_lift = 0.0
    if {0, 1}.issubset(set(promo_sales.index)):
        promo_lift = (promo_sales.loc[1] / promo_sales.loc[0] - 1) * 100

    report = f"""# EDA Report

Generated from `{'sample data' if use_sample else 'real Rossmann data'}`.

## Dataset Summary
| Item | Value |
|---|---:|
| Raw rows | {len(raw):,} |
| Clean open-store rows | {len(clean):,} |
| Stores | {clean['Store'].nunique():,} |
| Date min | {clean['Date'].min().date()} |
| Date max | {clean['Date'].max().date()} |
| Total sales | {clean['Sales'].sum():,.0f} |
| Average daily sales per row | {clean['Sales'].mean():,.2f} |

## Promotion Effect
Average non-promo sales: {promo_sales.get(0, 0):,.2f}
Average promo sales: {promo_sales.get(1, 0):,.2f}
Estimated promo lift: {promo_lift:,.2f}%

## Generated Plots

### Total Daily Sales
![Total daily sales](assets/plots/daily_sales.png)

### Average Sales by Day of Week
![Average sales by day of week](assets/plots/sales_by_weekday.png)

### Promotion Lift
![Promotion lift](assets/plots/promo_lift.png)

### Top Stores
![Top stores](assets/plots/top_stores.png)

## What To Look For

- Demand has strong calendar behavior.
- Promotion days usually change average sales.
- Store-level volume varies significantly, so store-specific features matter.
- Total sales over time can reveal holidays, closure periods, and seasonality.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved EDA report to {report_path.relative_to(project_path())}")
    print(f"Saved plots to {plots_dir.relative_to(project_path())}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA plots and report.")
    parser.add_argument("--sample", action="store_true", help="Use generated sample data.")
    args = parser.parse_args()
    run_eda(use_sample=args.sample)


if __name__ == "__main__":
    main()
