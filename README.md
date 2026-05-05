---
title: AlphaPulse HK Stock Forecast AI
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AlphaPulse — HK Stock Forecast AI

A full-stack machine learning web app that predicts 7-day price direction for HKEX stocks and surfaces the top asymmetric opportunities across HSI constituents.

## Features

| Feature | Detail |
|---|---|
| **5-class prediction** | DOWN >5% / DOWN 3-5% / STABLE / UP 3-5% / UP >5% |
| **Dual confidence bars** | P(UP >3%) and P(UP >5%) visualised as animated bars |
| **Edge ratio** | P(up) / P(down) — highlights asymmetric setups |
| **HSI screener** | `/scan` ranks 43 HSI constituents by edge ratio, returns top 5 |
| **5-segment donut chart** | Full probability distribution per prediction |
| **1-hour scan cache** | Screener results cached in-memory; re-fetched hourly |

## Architecture

```
Browser (mobile-first dark UI)
    │  GET /predict/{ticker}
    │  GET /scan
    ▼
FastAPI (app.py)
    ├── get_price_data.py   — yfinance fetch + RSI/SMA/volatility features
    ├── stock_model.joblib  — CalibratedClassifierCV(RandomForest, isotonic, TimeSeriesSplit)
    └── static/             — index.html · style.css · script.js (Chart.js)
```

## API

| Endpoint | Description |
|---|---|
| `GET /predict/{ticker}` | Single-ticker prediction (e.g. `0700.HK`, `AAPL`) |
| `GET /scan` | Top 5 HSI picks by edge ratio (cached 1 h) |

**`/predict` response fields:**
```json
{
  "ticker": "0001.HK",
  "prediction": 0,
  "signal": "STABLE",
  "confidence_up_3pct": 0.22,
  "confidence_up_5pct": 0.16,
  "edge_ratio": 0.73,
  "probabilities": {"-2":0.20, "-1":0.11, "0":0.47, "1":0.06, "2":0.16},
  "current_price": 68.0,
  "last_updated": "2026-05-05"
}
```

## Project Structure

```
.
├── app.py                  # FastAPI routes + inference logic
├── get_price_data.py       # yfinance fetch + technical indicators
├── train_model.py          # Full training pipeline (run locally)
├── stock_model.joblib      # Trained model — NOT in git (hosted on HF Model Hub)
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   ├── test_app.py         # 11 unit tests
│   └── test_train_model.py # 22 unit tests
├── data/
│   └── hkex.csv            # HKEX equity ticker list (used by train_model.py)
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run training (30-60 min, requires network):
python train_model.py

# Start the app:
uvicorn app:app --reload --port 7860
# → http://localhost:7860
```

### Tests

```bash
pytest tests/ -v
```

## Deployment on Hugging Face Spaces

The trained model (`stock_model.joblib`) is hosted in a separate HF Model repository and downloaded automatically at container startup.

### Step 1 — Upload the model to HF Model Hub

```bash
pip install huggingface_hub
huggingface-cli login

# Create a model repo at huggingface.co/new, then upload:
huggingface-cli upload <your-username>/alphapulse-model stock_model.joblib
```

### Step 2 — Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK
3. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/alphapulse
   git push space main
   ```

### Step 3 — Set Space variables

In the Space → **Settings → Variables and Secrets**, add:

| Key | Value |
|---|---|
| `HF_MODEL_REPO` | `<your-username>/alphapulse-model` |
| `HF_TOKEN` | Your HF token (only needed if the model repo is private) |

The container downloads the model on first startup (~10-15 min for a large model).

> **RAM requirement:** The current model (~24 GB on disk) requires at least **32 GB RAM**. Use a Space with a **CPU Upgrade (L)** or retrain with `n_estimators=30` and `max_depth=15` for a smaller model that runs on the free tier.

## Deployment on Render

See [`render.yaml`](render.yaml). Set the `MODEL_URL` environment variable to a direct download URL for `stock_model.joblib` (e.g., the HF Hub raw URL).

## Model Details

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` in `CalibratedClassifierCV` |
| Calibration | Isotonic regression, `TimeSeriesSplit(n_splits=3)` |
| Features | SMA_5_ratio, SMA_20_ratio, RSI_14, Volatility_20, Returns_1d, Returns_5d |
| Training data | HKEX equity tickers, 5 years of daily OHLCV |
| Train/test split | 80 / 20 time-based |
| Class weights | `balanced` |
| Prediction horizon | 7 trading days |

## Disclaimer

Not financial advice. Predictions are based on historical price patterns only.
