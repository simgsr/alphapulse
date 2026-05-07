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

A full-stack machine learning web app that predicts 7-day price direction for HKEX stocks and surfaces the top asymmetric opportunities across a curated HK/SG watchlist.

## Features

| Feature | Detail |
|---|---|
| **3-class prediction** | DOWN >3% / STABLE / UP >3% over 7 trading days |
| **Confidence bars** | P(UP >3%) and P(DOWN >3%) visualised as animated bars |
| **Edge ratio** | P(up) / P(down) — highlights asymmetric setups |
| **Built-in screener** | `/scan` ranks 46 HK+SG stocks by edge ratio, returns top 5 (concurrent fetch, cached 1 h) |
| **My Watchlist** | Add tickers manually or import from CSV; scan your list with live progress |
| **3-segment donut chart** | Full probability distribution (DOWN / STABLE / UP) |

## Architecture

```
Browser (mobile-first dark UI)
    │  GET /predict/{ticker}
    │  GET /scan
    ▼
FastAPI (app.py)
    ├── get_price_data.py   — yfinance fetch + 14 technical features
    ├── stock_model.joblib  — RobustScaler → LGBMClassifier (3-class)
    └── static/             — index.html · style.css · script.js (Chart.js)
```

## API

| Endpoint | Description |
|---|---|
| `GET /predict/{ticker}` | Single-ticker prediction (e.g. `0700.HK`, `AAPL`) |
| `GET /scan` | Top 5 picks by edge ratio from built-in watchlist (cached 1 h) |

**`/predict` response fields:**
```json
{
  "ticker": "0700.HK",
  "prediction": 1,
  "signal": "UP > 3%",
  "confidence_up_3pct": 0.38,
  "confidence_down_3pct": 0.22,
  "edge_ratio": 1.73,
  "probabilities": {"-1": 0.22, "0": 0.40, "1": 0.38},
  "current_price": 413.96,
  "last_updated": "2026-05-06"
}
```

## Model Details

| Property | Value |
|---|---|
| Algorithm | `LGBMClassifier` (LightGBM) |
| Preprocessing | `RobustScaler` (resistant to HK small-cap outliers) |
| Features (14) | `SMA_5_ratio`, `SMA_20_ratio`, `SMA_50_ratio`, `RSI_14`, `RSI_7`, `MACD`, `MACD_hist`, `BB_pct_b`, `Volume_ratio_20`, `Volatility_20`, `Returns_1d`, `Returns_5d`, `Returns_10d`, `Returns_20d` |
| Training data | 2,626 HKEX equity tickers, 5 years of daily OHLCV (~2.8M rows) |
| Train/test split | 80 / 20 time-based |
| Class weights | `balanced` |
| Prediction horizon | 7 trading days |
| Test accuracy | ~45% (3-class; random baseline = 33%) |
| Test log-loss | ~1.04 |

## Project Structure

```
.
├── app.py                  # FastAPI routes + inference logic
├── get_price_data.py       # yfinance fetch + 14 technical indicators
├── train_model.py          # Full training pipeline (run locally)
├── stock_model.joblib      # Trained model (~6 MB, committed to repo)
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   ├── test_app.py
│   └── test_train_model.py
├── data/
│   └── alphapulse.csv      # HKEX equity ticker list (gitignored)
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (Optional) Retrain the model — takes ~30–60 min, requires network:
python train_model.py

# Start the app:
uvicorn app:app --reload --port 7860
# → http://localhost:7860
```

### Tests

```bash
pytest tests/ -v
```

## Deployment on Render

Render deploys automatically on every push to `master` via the Docker runtime defined in [`render.yaml`](render.yaml). No environment variables are required — `stock_model.joblib` is committed directly to the repository.

1. Connect the GitHub repo to a Render web service
2. Set runtime to **Docker**
3. Push to `master` — Render builds and deploys automatically

## Disclaimer

Not financial advice. Predictions are based on historical price patterns only.
