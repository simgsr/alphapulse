# AlphaPulse Dual-Model Improvement Design
**Date:** 2026-05-08
**Goal:** Improve model precision for UP signals; introduce dual-horizon models (5-day and 14-day) for side-by-side comparison in the UI; expand training universe to HKEX + SGX.

---

## 1. Features

21 total features (14 existing + 7 new). All computed in `get_price_data.py → calculate_technical_indicators()`.

### Existing (unchanged)
`SMA_5_ratio`, `SMA_20_ratio`, `SMA_50_ratio`, `RSI_14`, `RSI_7`, `MACD`, `MACD_hist`, `BB_pct_b`, `Volume_ratio_20`, `Volatility_20`, `Returns_1d`, `Returns_5d`, `Returns_10d`, `Returns_20d`

### New (7 additions)
| Feature | Calculation | Purpose |
|---|---|---|
| `Stoch_K` | Stochastic %K (14-day) | Momentum oscillator — oversold bounce detection |
| `Stoch_D` | 3-day SMA of `Stoch_K` | Smoothed momentum confirmation |
| `ATR_ratio` | ATR(14) / current price | Normalized volatility; breakout potential |
| `ADX_14` | Average Directional Index (14) | Trend strength filter |
| `OBV_ratio` | OBV / 20-day OBV SMA | Volume accumulation/distribution |
| `ROC_10` | (price / price[-10]) - 1 | Raw 10-day momentum |
| `CMF_20` | Chaikin Money Flow (20-day) | Buying/selling pressure via volume-weighted close |

**Minimum data requirement remains 252 rows** (unchanged).

---

## 2. Dual Models & Label Design

Two separate models trained and saved independently.

| Property | 5-day model | 14-day model |
|---|---|---|
| Output file | `stock_model_5d.joblib` | `stock_model_14d.joblib` |
| Forward return | `shift(-5)` | `shift(-14)` |
| UP label | > +2% | > +5% |
| DOWN label | < -2% | < -5% |
| STABLE | within ±2% | within ±5% |
| Clip range | ±20% | ±30% |

**Decision threshold tuning:** After each model is trained, `train_model.py` prints a precision/recall table for the UP class across thresholds 0.30–0.60. The threshold is not baked into the `.joblib` — it is applied at inference time in `app.py`. Default inference threshold: argmax (unchanged from current behavior unless manually overridden).

---

## 3. Training Pipeline (`train_model.py`)

### Interface changes
- `discretize_return(r, up_thresh, down_thresh)` — thresholds passed as arguments, not hardcoded
- `build_ticker_dataset(ticker, period, horizon, up_thresh, down_thresh)` — horizon and thresholds parameterized
- `build_full_dataset(csv_path, horizon, up_thresh, down_thresh)` — passes through to above
- `train_and_save(csv_path, horizon, up_thresh, down_thresh, output_path)` — full run for one model
- `FEATURE_NAMES` updated to list all 21 features

### `__main__` block
Calls `train_and_save` twice:
1. 5-day model → `stock_model_5d.joblib`
2. 14-day model → `stock_model_14d.joblib`

### LightGBM hyperparameters
- `n_estimators`: 300 → **500** (more features, more data)
- `num_leaves`: 63 → **95**
- All other params unchanged (`learning_rate=0.05`, `class_weight='balanced'`, `min_child_samples=20`, `random_state=42`)

### SGX ticker support
- If `data/sgx_tickers.csv` exists (one ticker per row, e.g. `U96.SI`), it is loaded and merged with the HKEX ticker list before training
- If the file does not exist, training proceeds on HKEX only — no error, no breaking change

---

## 4. UI Changes (`app.py`)

### Model loading
```python
model_5d  = load("stock_model_5d.joblib")   # None if file missing
model_14d = load("stock_model_14d.joblib")  # None if file missing
```
Each loaded independently. If one is missing, its card shows "Model not available" — the other still works.

### Analyze tab
Single ticker analysis returns **two result cards** rendered side by side:
- Left card: 5-day forecast (signal, UP% bar, DOWN% bar, edge ratio, price, date)
- Right card: 14-day forecast (same fields)
- Layout: `gr.Row()` with two `gr.HTML` outputs

### Scan tab
- Runs both models per ticker concurrently
- Ranks results by **5-day edge ratio** (primary sort)
- Scan table columns: `#`, `TICKER`, `PRICE`, `SIGNAL 5D`, `EDGE 5D`, `SIGNAL 14D`, `EDGE 14D`, `UP CONF 5D`
- Top 10 results displayed (unchanged count)

### Everything else unchanged
Watchlist management, CSV import, add/remove tickers, theme, CSS — no modifications.

---

## 5. Files Changed

| File | Change |
|---|---|
| `get_price_data.py` | Add 7 new feature calculations to `calculate_technical_indicators()` |
| `train_model.py` | Parameterize horizon/thresholds; update `FEATURE_NAMES`; dual `train_and_save` calls; precision/recall table printout |
| `app.py` | Load two models; dual-card analyze output; updated scan table with both horizons |
| `data/sgx_tickers.csv` | New file (user-supplied); optional SGX ticker list |
| `stock_model_5d.joblib` | New — replaces `stock_model.joblib` for 5-day predictions |
| `stock_model_14d.joblib` | New — 14-day predictions |
| `stock_model.joblib` | Deprecated — can be removed after retraining |

---

## 6. Out of Scope
- Hyperparameter search (Optuna/grid search) — not in this iteration
- Probability calibration layer (isotonic regression) — not in this iteration
- Binary reformulation (UP vs not-UP) — not in this iteration
- Any changes to Dockerfile, tests, or deployment config
