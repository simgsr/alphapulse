# AlphaPulse: Cross-Sectional Rank Features + Optuna Tuning

**Date:** 2026-05-09
**Goal:** Improve UP signal precision by (1) adding cross-sectional percentile rank features so the model sees each stock relative to its peers, and (2) running Optuna hyperparameter search on the resulting 42-feature space.

---

## 1. Overview

Two improvements are applied in sequence, both targeting the same precision ceiling (~0.47 at threshold 0.65):

| Step | What changes | Expected impact |
|---|---|---|
| Cross-sectional rank features | Feature set 21 → 42 | Model sees relative strength, not just absolute indicators |
| Optuna tuning | LightGBM hyperparameters | Squeezes remaining precision from the new feature space |
| LLM interpretation | New `llm_utils.py` + UI buttons | Plain-English explanation of signals for the analyst |

Market-neutral labels are explicitly out of scope for this iteration.

---

## 2. Cross-Sectional Rank Features

### 2.1 Concept

For every trading date, each stock is ranked against all other stocks on the **same exchange** for each of the 21 existing features. The rank is expressed as a percentile (0.0–1.0):

- `RSI_14_rank = 0.80` means this stock's RSI is higher than 80% of same-exchange stocks on that date
- A stock with RSI=70 in a market where most stocks have RSI=80 is different from RSI=70 when most have RSI=50 — the rank captures this

### 2.2 Exchange Grouping

Exchange is derived from ticker suffix at dataset-build time:

| Suffix | Exchange key |
|---|---|
| `.HK` | `'HK'` |
| `.SI` | `'SGX'` |
| other | `'ALL'` (fallback) |

Each row in the combined DataFrame gets an `exchange` column. Cross-sectional ranks are computed per `(date, exchange)` group.

### 2.3 New Features

21 new `_rank` columns are appended to each row after the existing 21:

```python
RANK_FEATURE_NAMES = [f'{f}_rank' for f in FEATURE_NAMES]
ALL_FEATURE_NAMES = FEATURE_NAMES + RANK_FEATURE_NAMES  # 42 total
```

Rank values are computed with `pd.Series.rank(pct=True)` grouped by `(date, exchange)`.

### 2.4 Quantile Table (Inference)

At inference time only one stock is available, so same-day ranking is impossible. During training, a **quantile table** is built and stored in the model dict:

```python
quantile_table = {
    'HK':  {feature: sorted_np_array_of_training_values, ...},
    'SGX': {feature: sorted_np_array_of_training_values, ...},
    'ALL': {feature: sorted_np_array_of_training_values, ...},
}
```

At inference, `np.searchsorted(sorted_array, value) / len(sorted_array)` gives the percentile rank in O(log n).

The quantile table is saved in the model `.joblib` dict under key `'quantile_table'`.

### 2.5 Changes to `train_model.py`

- `build_ticker_dataset()`: add `exchange` column derived from ticker suffix
- `build_full_dataset()`: after concatenating frames, compute `_rank` columns per `(date, exchange)` group; build quantile table; return signature expands to `((X_train, y_train), (X_test, y_test), quantile_table)`
- `FEATURE_NAMES` → `ALL_FEATURE_NAMES` (42 features) used for `X` selection and model dict
- Model dict gains `'quantile_table'` key
- `'ALL'` key in quantile table is built from the union of all exchanges' training values for each feature (fallback for unrecognised ticker suffixes)

### 2.6 No changes to `get_price_data.py`

All 21 base features are already computed there. No modifications needed.

---

## 3. Optuna Hyperparameter Search

### 3.1 Parameters and Search Space

| Parameter | Type | Range |
|---|---|---|
| `num_leaves` | int | 50 – 300 |
| `learning_rate` | float (log) | 0.01 – 0.20 |
| `n_estimators` | int | 200 – 1000 |
| `min_child_samples` | int | 10 – 100 |
| `subsample` | float | 0.5 – 1.0 |
| `colsample_bytree` | float | 0.5 – 1.0 |
| `reg_alpha` | float | 0.0 – 5.0 |
| `reg_lambda` | float | 0.0 – 5.0 |

### 3.2 Objective Function

Maximize **precision at threshold 0.65** on a fixed validation slice. This directly optimises for the stated operational goal rather than a proxy metric.

### 3.3 Data Strategy

```
Full training set (80% of combined data, ~6.7M rows)
├── Optuna subsample: 30% stratified sample (~2M rows) → used for all 50 trials
│   ├── Trial train split: 80% of subsample
│   └── Trial val split:   20% of subsample (fixed across trials)
└── Final training: full training set using best params
```

The final held-out test set (20% of combined data) is **never touched** during Optuna.

### 3.4 Implementation

`train_and_save()` gains an `optuna_tune: bool = False` parameter. When `True`:

1. Draw stratified subsample from `X_train / y_train`
2. Split subsample 80/20 for trial train/val
3. Run `optuna.create_study(direction='maximize')` for 50 trials
4. Print best params and best precision
5. Build final pipeline using best params
6. Fit on full `X_train / y_train`

When `False` (default), behaviour is unchanged — uses the hardcoded hyperparameters from `build_pipeline()`.

### 3.5 Runtime Estimate

~1–2 min per trial × 50 trials × 2 models (5d + 14d) = **~2–4 hours** total.

---

## 4. Inference Changes (`app.py`)

### 4.1 Rank feature helper

A new function `add_rank_features(df, ticker, quantile_table)`:

1. Detects exchange from ticker suffix
2. Selects the appropriate sorted arrays from `quantile_table`
3. For each of 21 base features, computes percentile rank via `np.searchsorted`
4. Appends 21 `_rank` columns to `df`

Called in both the Analyze tab (single stock) and Scan tab (batch) before prediction.

### 4.2 Model loading

`_load_model()` currently unpacks to `(model, feature_names)`. It is updated to also return `quantile_table`, becoming `(model, feature_names, quantile_table)`. Both call sites in `app.py` (lines 26–27) are updated accordingly.

### 4.3 Signal display

Unchanged — signal cards, confidence bar, edge ratio, and scan columns are not modified.

---

## 5. LLM Interpretation Layer

### 5.1 New file: `llm_utils.py`

Contains all LLM logic, keeping `app.py` clean. Three components:

**`get_llm(provider, temperature=0)`** — supports:
- `"ollama_local"` (default): `ChatOllama(model=OLLAMA_LOCAL_MODEL)`
- `"groq"`: `ChatGroq` — requires `GROQ_API_KEY` env var
- `"gemini"`: `ChatGoogleGenerativeAI` — requires `GOOGLE_API_KEY` env var

Provider is read from the `LLM_PROVIDER` env var at startup (default: `"ollama_local"`). No UI dropdown.

**`explain_signal(ticker, price, result_5d, result_14d, indicators) -> str`**

Builds a structured prompt with:
- Ticker and current price
- 5d signal (UP/NOT-UP), confidence %, edge ratio
- 14d signal (UP/NOT-UP), confidence %, edge ratio
- 8 readable indicators: `RSI_14`, `MACD_hist`, `SMA_20_ratio`, `Volume_ratio_20`, `ATR_ratio`, `Returns_5d`, `Returns_20d`, `BB_pct_b`

System prompt: *"You are a financial signal interpreter. Summarise the model output in 3-5 plain-English sentences. Do not give buy/sell advice. Do not invent information not provided."*

Returns the LLM response string.

**`summarize_scan(scan_rows) -> str`**

Takes the list of scan result dicts (ticker, price, 5d signal + confidence, 14d signal + confidence) and returns a one-paragraph summary of patterns across the watchlist signals.

### 5.2 UI additions in `app.py`

**Analyze tab:**
- `gr.Button("Explain signal")` placed below the 5d/14d result cards
- `gr.Markdown()` box below the button, initially empty
- Button click calls `explain_signal()` with the last analysis context and renders the result

**Scan tab:**
- `gr.Button("Summarise scan")` placed below the results table
- `gr.Markdown()` box below the button, initially empty
- Button click calls `summarize_scan()` with the scan results and renders the result

### 5.3 State management

The last analysis result (ticker, price, signals, indicators) is stored in a `gr.State` so the "Explain signal" button can access it without re-running the model.

---

## 6. Files Changed

| File | Change |
|---|---|
| `train_model.py` | Exchange tagging; cross-sectional rank computation; quantile table build + save; `optuna_tune` flag; update `FEATURE_NAMES` → `ALL_FEATURE_NAMES` |
| `app.py` | Load `quantile_table`; add `add_rank_features()`; add "Explain signal" and "Summarise scan" buttons with `gr.State` for context |
| `llm_utils.py` | New file: `get_llm()`, `explain_signal()`, `summarize_scan()` |
| `get_price_data.py` | No changes |
| `requirements.txt` | Add `optuna`, `langchain-ollama`, `langchain-groq`, `langchain-google-genai` |

---

## 7. Out of Scope

- Market-neutral labels (stock return vs index) — future iteration
- Time-series cross-validation for Optuna — too slow at this data scale
- UI dropdown for LLM provider — use `LLM_PROVIDER` env var instead
- Re-training the old `stock_model.joblib` (legacy, unused)
