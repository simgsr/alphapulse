# AlphaPulse: Dual-Threshold Prediction Improvement Design

**Date:** 2026-05-02
**Status:** Approved

---

## Overview

Improve AlphaPulse's stock prediction by retraining the model with a 5-class discretized target (instead of 3), enabling the API to return separate confidence levels for price moving UP >3% and UP >5% within 7 trading days. The frontend is redesigned as a mobile-first, dark-theme financial dashboard.

---

## Section 1: Training Pipeline (`train_model.py`)

### Data Source
- Ticker universe: `data/hkex.csv` — ~2,697 HKEX tickers in yfinance format (e.g. `0001.hk`)
- Historical period: 5 years of daily OHLCV data per ticker via `yf.download()`

### Feature Engineering
Same 6 features as the existing `get_price_data.py`:
- `SMA_5_ratio`, `SMA_20_ratio`, `RSI_14`, `Volatility_20`, `Returns_1d`, `Returns_5d`

All features use only backward-looking rolling windows — no lookahead bias.

### Target Discretization
Compute 7-day forward return for each row:

```
forward_return = (Close[t+7] - Close[t]) / Close[t]
```

Discretize into 5 classes:

| Label | Condition |
|-------|-----------|
| `-2`  | forward_return < −5% |
| `-1`  | −5% ≤ forward_return < −3% |
| `0`   | −3% ≤ forward_return ≤ +3% |
| `1`   | +3% < forward_return ≤ +5% |
| `2`   | forward_return > +5% |

- Drop the last 7 rows of each ticker's history (target cannot be computed without future data)
- Cap extreme returns at ±30% before labeling to filter data errors and un-adjusted splits

### Data Quality Filters
- Skip tickers with fewer than 252 rows after feature engineering (~1 year of trading days)
- Log count of tickers skipped vs used

### Train / Test Split
- Time-based: earliest 80% of rows → train set; latest 20% → test set
- No random shuffling — preserves temporal order across all tickers combined

### ML Pipeline (sklearn `Pipeline`)
```
Pipeline([
    ('scaler', RobustScaler()),
    ('clf', CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        method='isotonic',
        cv=TimeSeriesSplit(n_splits=5)
    ))
])
```

- `RobustScaler`: handles outliers common in HK small-cap data; fit only on train set
- `class_weight='balanced'`: compensates for class 0 (STABLE) dominating the dataset
- `CalibratedClassifierCV(method='isotonic')`: produces meaningful probability outputs rather than raw vote fractions
- `TimeSeriesSplit`: cross-validation respects temporal order — no future data leaks into fold training

### Evaluation
Print to stdout before saving:
- Class distribution (train and test)
- Classification report (precision, recall, F1 per class)
- Log-loss on held-out test set

### Output
Save as `stock_model.joblib` with schema:
```python
{
    'model': pipeline,         # fitted Pipeline object
    'features': [...],         # list of 6 feature names
    'description': '...'       # human-readable summary
}
```

---

## Section 2: API Changes (`app.py`)

### Derived Confidence Values
From the 5-class `predict_proba` output:
```python
confidence_up_3pct = P(class=1) + P(class=2)
confidence_up_5pct = P(class=2)
```

These are always consistent: `confidence_up_5pct ≤ confidence_up_3pct`.

### Signal Labels (5 classes)

| Prediction | Signal |
|------------|--------|
| `2`  | UP > 5%     |
| `1`  | UP 3–5%     |
| `0`  | STABLE      |
| `-1` | DOWN 3–5%   |
| `-2` | DOWN > 5%   |

### Response Schema
```json
{
  "ticker": "0001.HK",
  "prediction": 2,
  "signal": "UP > 5%",
  "confidence_up_3pct": 0.62,
  "confidence_up_5pct": 0.31,
  "probabilities": {
    "-2": 0.05,
    "-1": 0.10,
    "0":  0.23,
    "1":  0.31,
    "2":  0.31
  },
  "current_price": 57.35,
  "last_updated": "2026-05-01"
}
```

The legacy `prediction` integer field is retained for backward compatibility.

> **Implementation note:** `model.classes_` on a `Pipeline` object delegates to the final estimator (sklearn ≥ 0.24). Verify sklearn version in `requirements.txt` is pinned to ≥ 0.24.

---

## Section 3: Frontend Redesign (Mobile-First)

### Design Principles
- Mobile-first (primary target: 375–430px viewport width)
- Dark financial theme
- Single-column layout, no horizontal scroll
- Minimum tap target: 48px
- Animated transitions on result load

### Color Palette
| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#0a0f1e` | Page background |
| `--card` | `#111c2d` | Card surfaces |
| `--accent` | `#00d4ff` | Input focus, buttons |
| `--up-strong` | `#00ff88` | UP > 5% |
| `--up-mild` | `#4ade80` | UP 3–5% |
| `--neutral` | `#fbbf24` | STABLE |
| `--down-mild` | `#f97316` | DOWN 3–5% |
| `--down-strong` | `#ef4444` | DOWN > 5% |

### Layout (top → bottom)
1. **Header**: App name + minimal settings icon
2. **Search**: Full-width ticker input + submit button (48px height)
3. **Result card** (fade-in on load):
   - Stock ticker + name
   - Current price (`32px bold`, tabular-nums) + signal badge (colored pill)
   - Confidence bar: "Confidence price UP >3% in 7 days" + animated progress bar + percentage
   - Confidence bar: "Confidence price UP >5% in 7 days" + animated progress bar + percentage
   - 5-segment donut chart (200×200px, `cutout: 72%`)
   - Inline legend row (5 color dots + labels)
   - Last updated (muted small text)
4. **Error state**: Inline red banner below input (replaces `alert()`)

### Chart
- 5 segments mapped to classes `-2, -1, 0, 1, 2`
- Segment colors match the palette above
- No hover tooltips on mobile; tap shows segment label via Chart.js `onClick`

### Files Changed
- `static/index.html`: restructure card layout, add confidence bar elements
- `static/style.css`: full rewrite for mobile-first dark theme
- `static/script.js`: update `updateUI()` to consume new API fields, 5-segment chart

---

## Affected Files

| File | Change |
|------|--------|
| `train_model.py` | **New** — full training pipeline |
| `stock_model.joblib` | **Replaced** — retrained 5-class model |
| `app.py` | **Modified** — new signal map, derived confidence fields |
| `static/index.html` | **Modified** — new mobile layout |
| `static/style.css` | **Rewritten** — dark mobile-first theme |
| `static/script.js` | **Modified** — consume new API shape, 5-segment chart |
| `data/hkex.csv` | Unchanged (input only) |
