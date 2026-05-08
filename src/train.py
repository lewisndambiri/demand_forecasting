from __future__ import annotations
import argparse
from pathlib import Path

import joblib
from src.data_loader import load_sales_data, resolve_data_paths
from src.features import build_features, make_model_frame
from src.model import (
    baseline_comparison,
    feature_importance_frame,
    make_regressor,
    regression_metrics,
    store_level_metrics,
    time_based_split,
)
from src.preprocessing import clean_sales_data
from src.utils import ensure_dir, load_config, project_path, save_json

def train_pipeline(use_sample: bool = False) -> dict:
    config = load_config()
    train_path, store_path = resolve_data_paths(config, use_sample=use_sample)

    raw = load_sales_data(train_path, store_path)
    clean = clean_sales_data(raw)
    featured = build_features(clean, config)

    train_frame, test_frame = time_based_split(featured, config["model"]["test_days"])
    target = config["features"]["target"]
    X_train, y_train = make_model_frame(train_frame, target)
    X_test, y_test = make_model_frame(test_frame, target)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model_name, model = make_regressor(config["model"]["random_state"])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    history_columns = [column for column in test_frame.columns if column.startswith(f"{target}_lag_")]
    holdout = test_frame[["Date", "Store", target, *history_columns]].copy()
    holdout["PredictedSales"] = predictions
    holdout["Error"] = holdout[target] - holdout["PredictedSales"]
    holdout["AbsoluteError"] = holdout["Error"].abs()

    store_metrics = store_level_metrics(holdout, target)
    baselines = baseline_comparison(holdout, target)
    importance = feature_importance_frame(model, list(X_train.columns))

    model_dir = ensure_dir(project_path(config["paths"]["model_dir"]))
    model_path = project_path(config["paths"]["model_path"])
    metrics_path = project_path(config["paths"]["metrics_path"])
    predictions_path = project_path(config["paths"]["predictions_path"])
    store_metrics_path = project_path(config["paths"]["store_metrics_path"])
    importance_path = project_path(config["paths"]["feature_importance_path"])
    baseline_path = project_path(config["paths"]["baseline_comparison_path"])
    joblib.dump(
        {
            "model": model,
            "columns": list(X_train.columns),
            "model_name": model_name,
            "target": target,
            "config": config,
        },
        model_path,
    )
    holdout.to_csv(predictions_path, index=False)
    store_metrics.to_csv(store_metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    baselines.to_csv(baseline_path, index=False)
    save_json(
        {
            "model": model_name,
            "train_rows": int(len(train_frame)),
            "holdout_rows": int(len(test_frame)),
            "stores": int(featured["Store"].nunique()),
            "date_min": str(featured["Date"].min().date()),
            "date_max": str(featured["Date"].max().date()),
            **metrics,
        },
        metrics_path,
    )

    print(f"Saved model to {model_path.relative_to(project_path())}")
    print(f"Saved metrics to {metrics_path.relative_to(project_path())}")
    print(f"Saved holdout predictions to {predictions_path.relative_to(project_path())}")
    print(f"Model: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.3f}")
    return {"model_path": Path(model_path), "metrics": metrics}

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the demand forecasting model.")
    parser.add_argument("--sample", action="store_true", help="Use generated sample data.")
    args = parser.parse_args()
    train_pipeline(use_sample=args.sample)

if __name__ == "__main__":
    main()
