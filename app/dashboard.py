from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_sales_data, resolve_data_paths
from src.features import build_features, make_model_frame
from src.forecast import forecast_pipeline
from src.preprocessing import clean_sales_data
from src.train import train_pipeline
from src.utils import load_config, project_path

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def kpi_card(column, label: str, value: str, note: str = "") -> None:
    note_html = f"<div class='kpi-note'>{escape(note)}</div>" if note else ""
    column.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value)}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: 0;}
    section[data-testid="stSidebar"] {background: #f7f8fa;}
    .kpi-card {
        min-height: 116px;
        background: #ffffff;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        color: #475467;
        font-size: 0.82rem;
        font-weight: 650;
        line-height: 1.2;
    }
    .kpi-value {
        color: #101828;
        font-size: 1.35rem;
        font-weight: 760;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }
    .kpi-note {
        color: #667085;
        font-size: 0.76rem;
        line-height: 1.2;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetricLabel"] {color: #475467;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config()
target = config["features"]["target"]
model_path = project_path(config["paths"]["model_path"])
metrics_path = project_path(config["paths"]["metrics_path"])
predictions_path = project_path(config["paths"]["predictions_path"])
store_metrics_path = project_path(config["paths"]["store_metrics_path"])
importance_path = project_path(config["paths"]["feature_importance_path"])
future_forecast_path = project_path(config["paths"]["future_forecast_path"])
baseline_path = project_path(config["paths"]["baseline_comparison_path"])

st.title("Store Demand Forecasting")
st.caption("A time-aware machine learning workflow for daily sales prediction.")

with st.sidebar:
    st.header("Controls")
    use_sample = st.toggle("Use sample data", value=not model_path.exists())
    if st.button("Train model", width="stretch", type="primary"):
        with st.spinner("Training model and refreshing artifacts..."):
            train_pipeline(use_sample=use_sample)
        st.success("Training complete.")
    if st.button("Create future forecast", width="stretch"):
        with st.spinner("Forecasting future/test rows..."):
            forecast_pipeline(use_sample=use_sample)
        st.success("Forecast created.")
    st.divider()
    st.caption("Expected files for real data")
    st.code("data/train.csv\ndata/store.csv", language="text")

if not model_path.exists():
    st.info("Train the model from the sidebar to create the first model artifact.")
    st.stop()

artifact = joblib.load(model_path)
metrics = load_json(metrics_path)
predictions = pd.read_csv(predictions_path, parse_dates=["Date"]) if predictions_path.exists() else pd.DataFrame()
store_metrics = pd.read_csv(store_metrics_path) if store_metrics_path.exists() else pd.DataFrame()
importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
future_forecast = pd.read_csv(future_forecast_path, parse_dates=["Date"]) if future_forecast_path.exists() else pd.DataFrame()
baselines = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()

top = st.columns(5)
kpi_card(top[0], "Model", str(metrics.get("model", artifact["model_name"])), "Active estimator")
kpi_card(top[1], "RMSE", f"{metrics.get('rmse', 0):,.1f}", "Lower is better")
kpi_card(top[2], "MAE", f"{metrics.get('mae', 0):,.1f}", "Avg sales error")
kpi_card(top[3], "MAPE", f"{metrics.get('mape', 0):.2f}%", "Avg percentage error")
kpi_card(top[4], "R2", f"{metrics.get('r2', 0):.3f}", "Variance explained")

overview, store_view, future_view, model_view, data_view = st.tabs(
    ["Overview", "Store Explorer", "Future Forecast", "Model Insights", "Data Health"]
)

with overview:
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Holdout Forecast Performance")
        if predictions.empty:
            st.warning("No holdout prediction file found. Train the model to create it.")
        else:
            daily = (
                predictions.groupby("Date")[[target, "PredictedSales"]]
                .sum()
                .rename(columns={target: "Actual sales", "PredictedSales": "Predicted sales"})
            )
            st.line_chart(daily, height=360)

    with right:
        st.subheader("Run Summary")
        summary = {
            "Training rows": metrics.get("train_rows", "n/a"),
            "Holdout rows": metrics.get("holdout_rows", "n/a"),
            "Stores": metrics.get("stores", "n/a"),
            "Date range": f"{metrics.get('date_min', 'n/a')} to {metrics.get('date_max', 'n/a')}",
            "Features": len(artifact["columns"]),
        }
        summary_frame = pd.DataFrame(summary.items(), columns=["Item", "Value"])
        summary_frame["Value"] = summary_frame["Value"].astype(str)
        st.dataframe(summary_frame, hide_index=True, width="stretch")

        if not store_metrics.empty:
            st.subheader("Best Stores by RMSE")
            st.dataframe(store_metrics.head(8), hide_index=True, width="stretch")

        if not baselines.empty:
            st.subheader("Baseline Comparison")
            st.dataframe(baselines, hide_index=True, width="stretch")

with store_view:
    train_path, store_path = resolve_data_paths(config, use_sample=use_sample)
    raw = load_sales_data(train_path, store_path)
    clean = clean_sales_data(raw)
    featured = build_features(clean, config)

    selected_store = st.selectbox("Store", sorted(featured["Store"].unique()))
    days = st.slider("Recent days", 14, 90, 45)
    store_history = featured[featured["Store"] == selected_store].copy()
    X, _ = make_model_frame(store_history, target)
    X = X.reindex(columns=artifact["columns"], fill_value=0)
    store_history["PredictedSales"] = artifact["model"].predict(X)
    latest = store_history.tail(days)

    store_cols = st.columns(4)
    store_cols[0].metric("Latest actual", f"{latest[target].iloc[-1]:,.0f}")
    store_cols[1].metric("Latest prediction", f"{latest['PredictedSales'].iloc[-1]:,.0f}")
    store_cols[2].metric("Avg actual", f"{latest[target].mean():,.0f}")
    store_cols[3].metric("Promo days", int(latest["Promo"].sum()))

    chart_data = latest.set_index("Date")[[target, "PredictedSales"]].rename(
        columns={target: "Actual sales", "PredictedSales": "Predicted sales"}
    )
    st.line_chart(chart_data, height=360)
    st.dataframe(
        latest[["Date", "Store", target, "PredictedSales", "Promo", "SchoolHoliday"]]
        .sort_values("Date", ascending=False)
        .reset_index(drop=True),
        hide_index=True,
        width="stretch",
    )

with future_view:
    st.subheader("Future/Test-Set Forecast")
    if future_forecast.empty:
        st.info("Create a future forecast from the sidebar. With Rossmann data, this uses `data/test.csv`.")
    else:
        future_store = st.selectbox(
            "Forecast store",
            sorted(future_forecast["Store"].unique()),
            key="future_store",
        )
        store_future = future_forecast[future_forecast["Store"] == future_store].copy()
        future_cols = st.columns(4)
        future_cols[0].metric("Forecast rows", f"{len(future_forecast):,}")
        future_cols[1].metric("Forecast stores", future_forecast["Store"].nunique())
        future_cols[2].metric("Start date", str(future_forecast["Date"].min().date()))
        future_cols[3].metric("End date", str(future_forecast["Date"].max().date()))

        daily_future = (
            future_forecast.groupby("Date")["PredictedSales"]
            .sum()
            .to_frame("Predicted sales")
        )
        st.line_chart(daily_future, height=320)
        st.line_chart(
            store_future.set_index("Date")["PredictedSales"].to_frame("Predicted sales"),
            height=280,
        )
        st.dataframe(store_future.head(80), hide_index=True, width="stretch")

with model_view:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Feature Importance")
        if importance.empty:
            st.info("Feature importance is available when the active model exposes it, for example XGBoost.")
        else:
            st.bar_chart(importance.head(20).set_index("feature")["importance"], height=420)
    with right:
        st.subheader("Per-Store Error Distribution")
        if store_metrics.empty:
            st.info("Train the model to generate per-store metrics.")
        else:
            st.bar_chart(store_metrics.set_index("Store")["rmse"], height=420)

with data_view:
    train_path, store_path = resolve_data_paths(config, use_sample=use_sample)
    raw = load_sales_data(train_path, store_path)
    clean = clean_sales_data(raw)
    data_cols = st.columns(4)
    data_cols[0].metric("Raw rows", f"{len(raw):,}")
    data_cols[1].metric("Clean rows", f"{len(clean):,}")
    data_cols[2].metric("Stores", clean["Store"].nunique())
    data_cols[3].metric("Missing values", int(clean.isna().sum().sum()))

    st.subheader("Data Preview")
    st.dataframe(clean.head(50), hide_index=True, width="stretch")
