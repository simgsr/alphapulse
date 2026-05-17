# AlphaPulse — HK/SG Stock Forecast AI

A machine learning web app that predicts short- and medium-term price direction for HKEX and SGX stocks, surfacing the top asymmetric opportunities across a personalised watchlist.

## Features

| Feature | Detail |
|---|---|
| **Dual-model predictions** | 5-day (±2% threshold) and 14-day (±5% threshold) models run side-by-side |
| **3-class output** | DOWN / STABLE / UP for each horizon |
| **Confidence bars** | P(UP) and P(DOWN) visualised as animated bars |
| **Edge ratio** | P(up) / P(down) — highlights asymmetric risk/reward setups |
| **Watchlist screener** | Scan your watchlist concurrently, ranked by 5-day edge ratio |
| **Persistent watchlist** | Add/remove tickers; state saved to `watchlist.json` |

## Architecture

```
Browser (Gradio 6 dark UI)
    ▼
app.py
    ├── get_price_data.py      — yfinance fetch + 21 technical features
    ├── stock_model_5d.joblib  — 5-day RobustScaler → LGBMClassifier
    └── stock_model_14d.joblib — 14-day RobustScaler → LGBMClassifier
```

## Model Details

| Property | 5-Day Model | 14-Day Model |
|---|---|---|
| Horizon | 5 trading days | 14 trading days |
| UP threshold | > +2% | > +5% |
| DOWN threshold | > −2% | > −5% |
| Return clip | ±20% | ±30% |
| Algorithm | `LGBMClassifier` | `LGBMClassifier` |
| Preprocessing | `RobustScaler` | `RobustScaler` |
| Class weights | `balanced` | `balanced` |
| n_estimators | 500 | 500 |

**Features (21):** `SMA_5_ratio`, `SMA_20_ratio`, `SMA_50_ratio`, `RSI_14`, `RSI_7`, `MACD`, `MACD_hist`, `BB_pct_b`, `Volume_ratio_20`, `Volatility_20`, `Returns_1d`, `Returns_5d`, `Returns_10d`, `Returns_20d`, `Stoch_K`, `Stoch_D`, `ATR_ratio`, `ADX_14`, `OBV_ratio`, `CCI_20`, `CMF_20`

**Training data:** HKEX + SGX tickers, 5 years of daily OHLCV, 80/20 time-based split.

## Project Structure

```
.
├── app.py                   # Gradio UI + inference logic
├── get_price_data.py        # yfinance fetch + 21 technical indicators
├── train_model.py           # Full training pipeline (run locally)
├── predict_upstock.py       # CLI: rank top 5 stocks by P(UP>3%)
├── validate_tickers.py      # Utility: build valid_tickers.csv from raw list
├── llm_utils.py             # LLM interpretation layer
├── tests/
│   ├── test_app.py
│   ├── test_get_price_data.py
│   └── test_train_model.py
├── data/
│   └── valid_tickers.csv    # Pre-validated ticker list
├── requirements.txt
└── requirements-dev.txt
```

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Run tests:
pytest tests/ -v

# Start the app (models must exist — see Training below):
python app.py
# → http://localhost:7860
```

## Training

Models are not committed to the repository. Train them locally before running the app:

```bash
# Ensure ticker CSVs exist in data/
python train_model.py
# Produces: stock_model_5d.joblib, stock_model_14d.joblib (~30–60 min)
```

To include SGX tickers, place `data/sgx_tickers.csv` (one ticker per line) alongside `data/alphapulse.csv` — it is picked up automatically.

## Disclaimer

Not financial advice. Predictions are based on historical price patterns only and may not reflect future performance.
