# AlphaPulse — HK / SG / China A-Share Stock Forecast AI

A machine learning web app that predicts short- and medium-term price direction for HKEX, SGX, Shanghai (SS), and Shenzhen (SZ) stocks, surfacing the top opportunities across a personalised watchlist.

## Features

| Feature | Detail |
|---|---|
| **Dual-model predictions** | 7-day and 14-day binary models run side-by-side |
| **Binary output** | NOT-UP / UP >3% for each horizon |
| **Confidence score** | P(UP>3%) per ticker |
| **Edge ratio** | P(UP) / P(NOT-UP) — highlights asymmetric setups |
| **Watchlist screener** | Scan your watchlist concurrently, ranked by edge ratio |
| **Persistent watchlist** | Add/remove tickers; state saved to `watchlist.json` |

## Architecture

```
Browser (Gradio 6 dark UI)
    ▼
app.py
    ├── get_price_data.py       — yfinance fetch + 21 technical features
    ├── train_model.py          — full training pipeline (run locally)
    ├── predict_upstock.py      — CLI: rank top 5 stocks by P(UP>3%)
    ├── stock_model_7d.joblib   — 7-day  RobustScaler → LGBMClassifier
    └── stock_model_14d.joblib  — 14-day RobustScaler → LGBMClassifier
```

## Model Details

| Property | 7-Day Model | 14-Day Model |
|---|---|---|
| Horizon | 7 trading days | 14 trading days |
| Output | Binary: NOT-UP / UP | Binary: NOT-UP / UP |
| UP threshold | > +3% | > +3% |
| Return clip | ±20% | ±30% |
| Algorithm | `LGBMClassifier` | `LGBMClassifier` |
| Preprocessing | `RobustScaler` | `RobustScaler` |
| Imbalance handling | `is_unbalance=True` | `is_unbalance=True` |
| n_estimators (max) | 1000 (early-stop) | 1000 (early-stop) |
| Hyperparameter tuning | Optuna (50 trials) | Optuna (50 trials) |
| Min price filter | 1.0 (local currency) | 1.0 (local currency) |
| Min avg daily volume | 500,000 shares | 500,000 shares |

### Features (42 total)

**21 raw technical indicators:** `SMA_5_ratio`, `SMA_20_ratio`, `SMA_50_ratio`, `RSI_14`, `RSI_7`, `MACD`, `MACD_hist`, `BB_pct_b`, `Volume_ratio_20`, `Volatility_20`, `Returns_1d`, `Returns_5d`, `Returns_10d`, `Returns_20d`, `Stoch_K`, `Stoch_D`, `ATR_ratio`, `ADX_14`, `OBV_ratio`, `CCI_20`, `CMF_20`

**21 cross-sectional rank features:** each raw feature ranked percentile (0–1) within its exchange group on each trading day (suffix `_rank`). This lets the model see relative strength across peers regardless of market-wide level.

**Training data:** HKEX + SGX + Shanghai (SS) + Shenzhen (SZ) tickers, 5 years of daily OHLCV, 80/20 time-based split. Penny stocks and illiquid tickers are excluded via price/volume filters before training.

## Project Structure

```
.
├── app.py                   # Gradio UI + inference logic
├── get_price_data.py        # yfinance fetch + 21 technical indicators
├── train_model.py           # Full training pipeline (run locally)
├── predict_upstock.py       # CLI: rank top 5 stocks by P(UP>3%)
├── validate_tickers.py      # Utility: build valid ticker list from raw CSV
├── llm_utils.py             # LLM interpretation layer
├── csv/
│   ├── hk.csv               # HKEX ticker list
│   ├── si.csv               # SGX ticker list
│   ├── ss.csv               # Shanghai ticker list
│   └── sz.csv               # Shenzhen ticker list
├── data/
│   └── stock_model_7d_tickers.csv  # Tickers that passed filters at last train
├── tests/
│   ├── test_app.py
│   ├── test_get_price_data.py
│   └── test_train_model.py
├── watchlist.json           # Persisted user watchlist (auto-created)
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
# Ticker CSVs must exist in csv/  (hk.csv, si.csv, ss.csv, sz.csv)
python train_model.py
# Produces: stock_model_7d.joblib, stock_model_14d.joblib (~60–90 min with Optuna)
```

The training pipeline automatically:
- Skips tickers where median closing price < 1.0 or median daily volume < 500,000
- Runs Optuna (50 trials) to tune LightGBM hyperparameters
- Uses early stopping against an AUC-PR validation set
- Saves a `stock_model_7d_tickers.csv` with the tickers that survived all filters

To adjust the penny-stock filter, pass `min_price` and `min_avg_volume` into `train_and_save()`.

## CLI Screener

```bash
# Rank top 5 by P(UP>3%) across any ticker list:
python predict_upstock.py csv/hk.csv
```

## Disclaimer

Not financial advice. Predictions are based on historical price patterns only and may not reflect future performance.
