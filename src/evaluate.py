from __future__ import annotations
import argparse
import json

from src.train import train_pipeline
from src.utils import project_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Show saved model metrics.")
    parser.add_argument("--sample", action="store_true", help="Train on sample data first.")
    args = parser.parse_args()

    metrics_path = project_path("models/metrics.json")
    if args.sample or not metrics_path.exists():
        train_pipeline(use_sample=args.sample)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("Current evaluation metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.upper()}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    baseline_path = project_path("models/baseline_comparison.csv")
    if baseline_path.exists():
        print("\nBaseline comparison")
        print(baseline_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
