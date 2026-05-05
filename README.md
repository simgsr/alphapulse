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
├── stock_model.joblib      # Trained model — NOT in git (see Deployment)
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   ├── test_app.py         # 11 unit tests (build_prediction_response, rank_scan_results)
│   └── test_train_model.py # 22 unit tests (discretize_return, build_ticker_dataset, …)
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

# Place stock_model.joblib in the project root, then:
uvicorn app:app --reload --port 7860
# → http://localhost:7860
```

### Re-training the model

Requires ~30-60 min and network access (yfinance). Produces `stock_model.joblib`.

```bash
python train_model.py
```

### Tests

```bash
pytest tests/ -v
```

## Deployment on Render

The trained model (`stock_model.joblib`, ~24 GB) is too large for a Git repository. Host it externally and let the container download it at startup.

### Step 1 — Host the model

Upload `stock_model.joblib` to [Hugging Face Hub](https://huggingface.co/) (free, supports large files via LFS):

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <your-username>/<repo-name> stock_model.joblib
```

The direct download URL will be:
```
https://huggingface.co/<your-username>/<repo-name>/resolve/main/stock_model.joblib
```

### Step 2 — Deploy on Render

1. Push this repository to GitHub (the `.gitignore` already excludes the model).
2. In Render, click **New → Web Service → Connect a Git repository**.
3. Render auto-detects `render.yaml` and uses the Dockerfile.
4. In **Environment Variables**, add:
   - `MODEL_URL` = the Hugging Face direct download URL from Step 1
5. Deploy. The container downloads the model on first startup (~10-15 min depending on bandwidth).

> **Note:** The free Render plan has limited RAM. A 24 GB model requires at least the **Standard** plan (2 GB RAM). Consider retraining with `n_estimators=50` to reduce model size for a free-tier demo.

## Model Details

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` wrapped in `CalibratedClassifierCV` |
| Calibration | Isotonic regression, `TimeSeriesSplit(n_splits=3)` |
| Features | SMA_5_ratio, SMA_20_ratio, RSI_14, Volatility_20, Returns_1d, Returns_5d |
| Training data | HKEX equity tickers, 5 years of daily OHLCV |
| Train/test split | 80 / 20 time-based |
| Class weights | `balanced` (compensates for dominant STABLE class) |
| Prediction horizon | 7 trading days |

## Disclaimer

Not financial advice. Predictions are based on historical price patterns only.
