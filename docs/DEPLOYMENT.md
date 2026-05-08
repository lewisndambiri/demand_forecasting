# Deployment Guide

This project supports two practical deployment paths:

- local Docker for reproducible demos
- Streamlit Community Cloud for a public portfolio dashboard

## Local Docker

Build the image:

```bash
docker build -t demand-forecasting .
```

Run the dashboard:

```bash
docker run --rm -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/docs/assets/plots:/app/docs/assets/plots" \
  demand-forecasting
```

Open:

```text
http://127.0.0.1:8501
```

## Docker Compose

```bash
docker compose up --build
```

The compose file mounts `data/`, `models/`, and `docs/assets/plots/` so local training and report artifacts can be reused by the container.

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud
3. Create a new app from the GitHub repo.
4. Set the entrypoint:

```text
app/dashboard.py
```
