.PHONY: help sample eda eda-sample train train-sample forecast forecast-sample evaluate test dashboard docker-build docker-run clean

PYTHON ?= python3
help:
	@echo "Available commands:"
	@echo "  make sample        Generate sample data"
	@echo "  make eda           Generate EDA report and plots"
	@echo "  make eda-sample    Generate sample-data EDA report and plots"
	@echo "  make train-sample  Train on generated sample data"
	@echo "  make train         Train on Kaggle data, falling back to sample data"
	@echo "  make forecast      Create future/test-set forecasts"
	@echo "  make forecast-sample  Create sample future forecasts"
	@echo "  make evaluate      Print saved metrics"
	@echo "  make test          Run tests"
	@echo "  make dashboard     Launch Streamlit dashboard"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run dashboard in Docker"
	@echo "  make clean         Remove generated local artifacts"

sample:
	$(PYTHON) -m src.sample_data

eda:
	$(PYTHON) -m src.eda

eda-sample:
	$(PYTHON) -m src.eda --sample

train:
	$(PYTHON) -m src.train

train-sample:
	$(PYTHON) -m src.train --sample

forecast:
	$(PYTHON) -m src.forecast

forecast-sample:
	$(PYTHON) -m src.forecast --sample

evaluate:
	$(PYTHON) -m src.evaluate

test:
	$(PYTHON) -m pytest -q

dashboard:
	streamlit run app/dashboard.py

docker-build:
	docker build -t demand-forecasting .

docker-run:
	docker run --rm -p 8501:8501 -v "$$(pwd)/data:/app/data" -v "$$(pwd)/models:/app/models" -v "$$(pwd)/docs/assets/plots:/app/docs/assets/plots" demand-forecasting

clean:
	rm -rf models plots logs .pytest_cache
