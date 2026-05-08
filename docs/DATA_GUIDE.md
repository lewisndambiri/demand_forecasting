# Data Guide

This project is designed for the Rossmann Store Sales dataset from Kaggle. The repository also includes generated sample data so the code runs without downloading anything.

## Option 1: Run With Sample Data

Use this first to check that your environment is working.

```bash
cd demand-forecasting
source ../venv/bin/activate
make train-sample
make dashboard
```

The sample data lives here:

```text
data/sample/train.csv
data/sample/store.csv
```

It is synthetic, so it is useful for demos and tests, not for claiming real business accuracy.

## Option 2: Download the Kaggle Dataset Manually

1. Go to the Rossmann Store Sales competition page:
   https://www.kaggle.com/c/rossmann-store-sales/data
2. Sign in to Kaggle.
3. Accept the competition rules if Kaggle asks.
4. Download the dataset archive.
5. Extract it.
6. Copy these files into the project:

```text
demand-forecasting/data/train.csv
demand-forecasting/data/store.csv
demand-forecasting/data/test.csv
```

Only `train.csv` and `store.csv` are needed for the current training pipeline. `test.csv` is kept for future Kaggle-style submission work.

Then run:

```bash
make train
make evaluate
make dashboard
```

## Option 3: Download With the Kaggle CLI

Install the optional Kaggle helper:

```bash
pip install kaggle
```

Create an API token:

1. Open https://www.kaggle.com/settings/account
2. Find the API section.
3. Click `Create New Token`.
4. Kaggle downloads a `kaggle.json` file.

Move it into the expected location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download and extract:

```bash
mkdir -p data
kaggle competitions download -c rossmann-store-sales -p data
unzip data/rossmann-store-sales.zip -d data
```

If Kaggle provides zipped CSV files, extract them too:

```bash
unzip "data/*.zip" -d data
```

You should now have:

```text
data/train.csv
data/store.csv
data/test.csv
```

## Expected Columns

`train.csv` should include:

```text
Store, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
```

`store.csv` should include:

```text
Store, StoreType, Assortment, CompetitionDistance, Promo2
```

The validation step checks for these columns before training.

## Why the Real Data Is Not Committed

The real dataset belongs to Kaggle and may have its own usage rules. For GitHub, keep the real CSV files local and let `.gitignore` exclude them.
