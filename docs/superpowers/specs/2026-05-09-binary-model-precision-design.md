# AlphaPulse Binary Model Precision Design

**Date:** 2026-05-09
**Goal:** Improve UP signal precision by (1) reformulating both models as binary classifiers (UP>3% vs NOT-UP) and (2) fixing OHLC data fetching so ATR and Stochastic use actual High/Low values instead of close-only approximations.

---

## 1. Label Design

Both the 5-day and 14-day models use a single binary label:

| Class | Condition | Meaning |
|---|---|---|
| `1` | forward return > 3% | UP signal |
| `0` | forward return ≤ 3% | NOT-UP (stable or falling) |

`discretize_return()` is renamed `binarize_return(r, thresh=0.03)` — single threshold, returns 0 or 1. The `down_thresh` parameter is removed everywhere it appears (`build_ticker_dataset`, `build_full_dataset`, `train_and_save`).

Both models use `up_thresh=0.03` (3%). The `horizon` parameter is the only difference between them (5 vs 14 days).

---

## 2. OHLC Data Fix

`fetch_latest_data()` in `get_price_data.py` changes from fetching `Close + Volume` only to fetching full OHLC: `Open`, `High`, `Low`, `Close`, `Volume`.

`calculate_technical_indicators()` is updated to use proper formulas where High/Low matter:

### ATR (fix)
True Range = max(H−L, |H−prev_Close|, |L−prev_Close|)

Current approximation uses `|diff(Close)|` which misses gap days entirely and consistently underestimates volatility.

### Stochastic %K/%D (fix)
`%K = (Close − rolling_Low_14) / (rolling_High_14 − rolling_Low_14) × 100`

Current approximation uses `rolling min/max of Close`, which understates the oscillator's range on days with large wicks.

### Unchanged indicators
MACD, RSI, SMA ratios, Bollinger Bands, Volume ratio, Volatility, Returns, OBV, CCI, CMF, ADX — all close-based by design, no changes needed.

`FEATURE_NAMES` list is unchanged (21 features).

---

## 3. Model Pipeline

### Class imbalance handling
Replace `class_weight='balanced'` with `scale_pos_weight`:

```python
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
```

Computed from the training set before fitting and passed into `LGBMClassifier`. More precise than `balanced` for binary problems.

### LightGBM hyperparameters
Unchanged from current values: `n_estimators=500`, `num_leaves=95`, `learning_rate=0.05`, `min_child_samples=20`, `random_state=42`, `n_jobs=2`.

### Saved model dict
```python
{
    'model': pipeline,
    'features': FEATURE_NAMES,
    'horizon': horizon,
    'up_thresh': 0.03,
    'binary': True,
    'description': '...',
}
```
`down_thresh` key removed. `binary: True` added for forward compatibility.

### Training output
`classification_report` label names: `['NOT-UP', 'UP>3%']`. Precision/recall sweep for the UP class is unchanged.

---

## 4. Inference & UI (`app.py`)

### Inference threshold
Default: argmax (`UP_prob > 0.5`). A config constant `UP_THRESHOLD` (default `0.5`) at the top of `app.py` allows manual override without touching model files. The precision/recall table printed at training time guides selection.

### Analyze tab result cards
- **Signal:** `UP >3%` or `NO SIGNAL`
- **UP probability bar:** unchanged
- **DOWN probability bar:** removed (no DOWN class)
- **Edge ratio:** `UP_prob / 0.5` — above 1.0 = model sees better-than-random odds

Both 5d and 14d cards follow this layout.

### Scan tab columns
`#`, `TICKER`, `PRICE`, `SIGNAL 5D`, `UP CONF 5D`, `SIGNAL 14D`, `UP CONF 14D`

`SIGNAL xD` = `UP >3%` or `—`. `UP CONF xD` = raw UP probability as a percentage.

### Unchanged
Watchlist management, CSV import, add/remove tickers, theme, CSS, scan count (top 10).

---

## 5. Files Changed

| File | Change |
|---|---|
| `get_price_data.py` | Fetch full OHLC; fix ATR (True Range) and Stochastic (actual H/L) |
| `train_model.py` | `binarize_return()` replaces `discretize_return()`; remove `down_thresh`; use `scale_pos_weight`; update label names; update `model_data` dict |
| `app.py` | Add `UP_THRESHOLD` constant; remove DOWN bar; update signal display and scan columns |

---

## 6. Out of Scope

- Hyperparameter search (Optuna) — future iteration
- Market-neutral labels (stock return vs index) — future iteration
- Cross-sectional rank features — future iteration
- Dockerfile, tests, deployment config — unchanged
