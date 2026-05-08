# AlphaPulse Dual-Model Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 new technical features, train two separate models (5-day and 14-day horizon), and update the Gradio UI to display both model predictions side-by-side.

**Architecture:** Feature engineering lives in `get_price_data.py`. Training logic in `train_model.py` is fully parameterized for horizon/thresholds and runs twice in `__main__` to produce two `.joblib` files. `app.py` loads both models at startup and renders dual result cards in analyze and dual-column rows in scan.

**Tech Stack:** Python 3.11, LightGBM, scikit-learn, yfinance, Gradio 5, joblib, pytest

---

## File Map

| File | Role |
|---|---|
| `get_price_data.py` | Add Stoch_K, Stoch_D, ATR_ratio, ADX_14, OBV_ratio, CCI_20, CMF_20 to `calculate_technical_indicators()` |
| `train_model.py` | Update `FEATURE_NAMES` (21 features); parameterize `discretize_return`, `build_ticker_dataset`, `build_full_dataset`, `train_and_save`; add precision/recall table; dual `__main__` calls |
| `app.py` | Load `stock_model_5d.joblib` + `stock_model_14d.joblib`; dual-card analyze UI; dual-horizon scan table |
| `tests/test_get_price_data.py` | Extend `test_new_columns_present` for the 7 new features; add bounds/sanity tests |
| `tests/test_train_model.py` | Add parameterized tests for new `discretize_return`/`build_ticker_dataset` signatures; add SGX merge test |
| `tests/test_app.py` | No changes required — existing tests cover `build_prediction_response`/`rank_scan_results` which are unchanged |

> **Note on ROC_10:** The spec listed ROC_10 = (price/price[-10])-1, which is identical to the existing `Returns_10d` feature. It has been replaced with **CCI_20** (Commodity Channel Index, 20-day), which is genuinely distinct from all existing features.

---

## Task 1: Add 7 new features to `get_price_data.py`

**Files:**
- Modify: `get_price_data.py`
- Test: `tests/test_get_price_data.py`

- [ ] **Step 1: Write the failing test — new columns present**

Add to `tests/test_get_price_data.py` (replace the existing `test_new_columns_present`):

```python
def test_new_columns_present():
    df = calculate_technical_indicators(_make_df())
    for col in [
        'SMA_5_ratio', 'SMA_20_ratio', 'SMA_50_ratio',
        'RSI_14', 'RSI_7',
        'MACD', 'MACD_hist',
        'BB_pct_b',
        'Volume_ratio_20',
        'Volatility_20', 'Returns_1d', 'Returns_5d', 'Returns_10d', 'Returns_20d',
        'Stoch_K', 'Stoch_D',
        'ATR_ratio',
        'ADX_14',
        'OBV_ratio',
        'CCI_20',
        'CMF_20',
    ]:
        assert col in df.columns, f"Missing column: {col}"
```

Add sanity tests below the existing tests:

```python
def test_stoch_k_bounded():
    df = calculate_technical_indicators(_make_df())
    assert df['Stoch_K'].between(0, 100).all()


def test_stoch_d_bounded():
    df = calculate_technical_indicators(_make_df())
    assert df['Stoch_D'].between(0, 100).all()


def test_atr_ratio_positive():
    df = calculate_technical_indicators(_make_df())
    assert (df['ATR_ratio'] > 0).all()


def test_adx_14_bounded():
    df = calculate_technical_indicators(_make_df())
    assert (df['ADX_14'] >= 0).all()
    assert (df['ADX_14'] <= 100).all()


def test_obv_ratio_no_nan():
    df = calculate_technical_indicators(_make_df())
    assert df['OBV_ratio'].isna().sum() == 0


def test_cci_20_no_nan():
    df = calculate_technical_indicators(_make_df())
    assert df['CCI_20'].isna().sum() == 0


def test_cmf_20_bounded():
    df = calculate_technical_indicators(_make_df())
    assert df['CMF_20'].between(-1, 1).all()
```

- [ ] **Step 2: Run failing tests**

```bash
cd /Users/simgsr/Documents/python_project/yf_price_prediction
source venv/bin/activate
pytest tests/test_get_price_data.py -v 2>&1 | tail -20
```

Expected: `test_new_columns_present` FAILS with `AssertionError: Missing column: Stoch_K`. Bounds tests also FAIL.

- [ ] **Step 3: Implement the 7 new features in `get_price_data.py`**

In `calculate_technical_indicators`, add the following block **before** the final `return df.dropna()`:

```python
    # Stochastic %K/%D (14-day, close-based approximation)
    low_14 = df['Adj_Close'].rolling(window=14).min()
    high_14 = df['Adj_Close'].rolling(window=14).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    df['Stoch_K'] = ((df['Adj_Close'] - low_14) / range_14 * 100).fillna(50)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ATR ratio (14-day, close-based simplified True Range)
    tr = df['Adj_Close'].diff().abs()
    atr_14 = tr.rolling(window=14).mean()
    df['ATR_ratio'] = (atr_14 / df['Adj_Close'].replace(0, np.nan)).fillna(0)

    # ADX (14-day, close-based approximation using directional movement)
    close_diff = df['Adj_Close'].diff()
    pos_dm = close_diff.clip(lower=0)
    neg_dm = (-close_diff).clip(lower=0)
    atr14_adx = close_diff.abs().rolling(14).mean().replace(0, np.nan)
    pdi = 100 * pos_dm.rolling(14).mean() / atr14_adx
    ndi = 100 * neg_dm.rolling(14).mean() / atr14_adx
    dx_denom = (pdi + ndi).replace(0, np.nan)
    dx = (100 * (pdi - ndi).abs() / dx_denom).fillna(0)
    df['ADX_14'] = dx.rolling(14).mean().fillna(0)

    # OBV ratio vs 20-day SMA of OBV
    direction = np.sign(df['Adj_Close'].diff().fillna(0))
    obv = (direction * df['Adj_Volume']).cumsum()
    obv_sma20 = obv.rolling(window=20).mean()
    df['OBV_ratio'] = (obv / obv_sma20.replace(0, np.nan)).fillna(1)

    # CCI_20 (Commodity Channel Index, 20-day)
    tp = df['Adj_Close']
    sma20_tp = tp.rolling(20).mean()
    mad20 = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI_20'] = ((tp - sma20_tp) / (0.015 * mad20.replace(0, np.nan))).fillna(0)

    # CMF_20 (Chaikin Money Flow, 20-day, close-only approximation)
    mfm = (2 * df['Adj_Close'] - df['Adj_Close'].rolling(2).min() - df['Adj_Close'].rolling(2).max())
    mfm = mfm / (df['Adj_Close'].rolling(2).max() - df['Adj_Close'].rolling(2).min()).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mf_vol = mfm * df['Adj_Volume']
    df['CMF_20'] = mf_vol.rolling(20).sum() / df['Adj_Volume'].rolling(20).sum().replace(0, np.nan)
    df['CMF_20'] = df['CMF_20'].fillna(0).clip(-1, 1)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_get_price_data.py -v 2>&1 | tail -25
```

Expected: all tests PASS including `test_no_nans_after_calculation`.

- [ ] **Step 5: Commit**

```bash
git add get_price_data.py tests/test_get_price_data.py
git commit -m "feat: add 7 new technical indicators (Stoch, ATR, ADX, OBV, CCI, CMF)"
```

---

## Task 2: Update `FEATURE_NAMES` in `train_model.py`

**Files:**
- Modify: `train_model.py`
- Test: `tests/test_train_model.py` (no changes needed — `_NEW_FEATURES` import still works)

- [ ] **Step 1: Replace `FEATURE_NAMES` in `train_model.py`**

Replace the existing `FEATURE_NAMES` list (lines 12–19) with:

```python
FEATURE_NAMES = [
    'SMA_5_ratio', 'SMA_20_ratio', 'SMA_50_ratio',
    'RSI_14', 'RSI_7',
    'MACD', 'MACD_hist',
    'BB_pct_b',
    'Volume_ratio_20',
    'Volatility_20', 'Returns_1d', 'Returns_5d', 'Returns_10d', 'Returns_20d',
    'Stoch_K', 'Stoch_D',
    'ATR_ratio',
    'ADX_14',
    'OBV_ratio',
    'CCI_20',
    'CMF_20',
]
```

- [ ] **Step 2: Run existing train_model tests to verify no regressions**

```bash
pytest tests/test_train_model.py -v 2>&1 | tail -30
```

Expected: all existing tests PASS. `TestBuildFullDataset` uses `_NEW_FEATURES` to build mock DataFrames — since it now includes 21 columns, the mock data is built with 21 ones-columns, which is still valid.

- [ ] **Step 3: Commit**

```bash
git add train_model.py
git commit -m "feat: update FEATURE_NAMES to 21 features"
```

---

## Task 3: Parameterize `discretize_return` in `train_model.py`

**Files:**
- Modify: `train_model.py`
- Test: `tests/test_train_model.py`

- [ ] **Step 1: Write new tests for parameterized thresholds**

Add to `tests/test_train_model.py` inside `TestDiscretizeReturn`:

```python
    def test_custom_up_thresh(self):
        # 5d model: up_thresh=0.02 — 0.025 should be UP
        assert discretize_return(0.025, up_thresh=0.02, down_thresh=0.02) == 1

    def test_custom_down_thresh(self):
        # 5d model: down_thresh=0.02 — -0.025 should be DOWN
        assert discretize_return(-0.025, up_thresh=0.02, down_thresh=0.02) == -1

    def test_stable_within_custom_thresholds(self):
        # 5d model: 0.015 is within ±2%, should be STABLE
        assert discretize_return(0.015, up_thresh=0.02, down_thresh=0.02) == 0

    def test_14d_up_thresh(self):
        # 14d model: up_thresh=0.05 — 0.04 should be STABLE
        assert discretize_return(0.04, up_thresh=0.05, down_thresh=0.05) == 0

    def test_14d_down_thresh(self):
        # 14d model: down_thresh=0.05 — -0.06 should be DOWN
        assert discretize_return(-0.06, up_thresh=0.05, down_thresh=0.05) == -1
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_train_model.py::TestDiscretizeReturn -v 2>&1 | tail -20
```

Expected: the 5 new tests FAIL with `TypeError: discretize_return() got unexpected keyword argument 'up_thresh'`.

- [ ] **Step 3: Update `discretize_return` signature**

Replace the existing `discretize_return` function in `train_model.py`:

```python
def discretize_return(r: float, up_thresh: float = 0.03, down_thresh: float = 0.03) -> int:
    """Map a forward return to a 3-class label.

    Classes:
        1  — UP   > up_thresh
        0  — STABLE (within thresholds)
       -1  — DOWN > down_thresh
    """
    if r > up_thresh:
        return 1
    elif r >= -down_thresh:
        return 0
    else:
        return -1
```

- [ ] **Step 4: Run all discretize_return tests**

```bash
pytest tests/test_train_model.py::TestDiscretizeReturn -v 2>&1 | tail -20
```

Expected: all 14 tests PASS (9 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: parameterize discretize_return with configurable thresholds"
```

---

## Task 4: Parameterize `build_ticker_dataset` in `train_model.py`

**Files:**
- Modify: `train_model.py`
- Test: `tests/test_train_model.py`

- [ ] **Step 1: Write new tests for parameterized build_ticker_dataset**

Add to `tests/test_train_model.py` inside `TestBuildTickerDataset`:

```python
    def test_5d_horizon_uses_shift_5(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk", horizon=5, up_thresh=0.02,
                                          down_thresh=0.02, clip=0.20)
        assert result is not None
        assert result["forward_return"].max() <= 0.20
        assert result["forward_return"].min() >= -0.20

    def test_14d_horizon_uses_shift_14(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk", horizon=14, up_thresh=0.05,
                                          down_thresh=0.05, clip=0.30)
        assert result is not None
        assert set(result["target"].unique()).issubset({-1, 0, 1})
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_train_model.py::TestBuildTickerDataset -v 2>&1 | tail -15
```

Expected: the 2 new tests FAIL with `TypeError: build_ticker_dataset() got unexpected keyword argument 'horizon'`.

- [ ] **Step 3: Update `build_ticker_dataset` in `train_model.py`**

Replace the existing `build_ticker_dataset` function:

```python
def build_ticker_dataset(
    ticker: str,
    period: str = '5y',
    horizon: int = 7,
    up_thresh: float = 0.03,
    down_thresh: float = 0.03,
    clip: float = 0.30,
) -> Optional[pd.DataFrame]:
    """Fetch, engineer features, and label one ticker.

    Returns None if data is unavailable or fewer than 252 labeled rows remain.
    """
    raw = fetch_latest_data(ticker, period=period)
    if raw is None:
        return None
    df = calculate_technical_indicators(raw)
    df = df.copy()
    df['forward_return'] = df['Adj_Close'].shift(-horizon) / df['Adj_Close'] - 1
    df = df.dropna(subset=['forward_return'])
    df['forward_return'] = df['forward_return'].clip(-clip, clip)
    df['target'] = df['forward_return'].apply(
        lambda r: discretize_return(r, up_thresh=up_thresh, down_thresh=down_thresh)
    )
    if len(df) < 252:
        return None
    return df
```

- [ ] **Step 4: Run all build_ticker_dataset tests**

```bash
pytest tests/test_train_model.py::TestBuildTickerDataset -v 2>&1 | tail -15
```

Expected: all 8 tests PASS (6 existing + 2 new).

Note: `test_forward_return_capped_at_30pct` still passes because it calls `build_ticker_dataset("0001.hk")` with default `clip=0.30`.

- [ ] **Step 5: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: parameterize build_ticker_dataset with horizon, thresholds, clip"
```

---

## Task 5: Parameterize `build_full_dataset` and `train_and_save`; add SGX support

**Files:**
- Modify: `train_model.py`
- Test: `tests/test_train_model.py`

- [ ] **Step 1: Write tests for parameterized build_full_dataset and SGX merge**

Add to `tests/test_train_model.py` (new class after `TestBuildFullDataset`):

```python
class TestBuildFullDatasetSGX:
    def _make_labeled_df(self, n_rows: int, start: str) -> pd.DataFrame:
        from train_model import FEATURE_NAMES as _FEATS
        idx = pd.date_range(start, periods=n_rows, freq="B")
        data = {f: np.ones(n_rows) for f in _FEATS}
        data['Adj_Close'] = np.ones(n_rows) * 100.0
        data['forward_return'] = np.zeros(n_rows)
        data['target'] = 0
        return pd.DataFrame(data, index=idx)

    def test_extra_csv_merges_tickers(self, tmp_path):
        hk_csv = tmp_path / "hkex.csv"
        hk_csv.write_text("0001.hk\n")
        sg_csv = tmp_path / "sgx_tickers.csv"
        sg_csv.write_text("D05.SI\n")

        df_a = self._make_labeled_df(300, "2020-01-01")
        df_b = self._make_labeled_df(300, "2020-06-01")

        call_log = []

        def mock_build(ticker, period='5y', **kwargs):
            call_log.append(ticker)
            return df_a if ticker == '0001.hk' else df_b

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            build_full_dataset(str(hk_csv), extra_csv_paths=[str(sg_csv)])

        assert '0001.hk' in call_log
        assert 'D05.SI' in call_log

    def test_no_extra_csv_runs_normally(self, tmp_path):
        hk_csv = tmp_path / "hkex.csv"
        hk_csv.write_text("0001.hk\n")

        df_a = self._make_labeled_df(300, "2020-01-01")

        with patch("train_model.build_ticker_dataset", return_value=df_a):
            from train_model import build_full_dataset
            (X_train, _), _ = build_full_dataset(str(hk_csv))

        assert len(X_train) > 0
```

Also add inside `TestBuildFullDataset` a test for the horizon param forwarding:

```python
    def test_horizon_param_forwarded(self, tmp_path):
        csv_content = "0001.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")
        received_kwargs = {}

        def mock_build(ticker, **kwargs):
            received_kwargs.update(kwargs)
            return df_a

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            build_full_dataset(str(csv_file), horizon=5, up_thresh=0.02, down_thresh=0.02)

        assert received_kwargs.get('horizon') == 5
        assert received_kwargs.get('up_thresh') == 0.02
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_train_model.py::TestBuildFullDatasetSGX tests/test_train_model.py::TestBuildFullDataset::test_horizon_param_forwarded -v 2>&1 | tail -15
```

Expected: all 3 tests FAIL with `TypeError`.

- [ ] **Step 3: Update `build_full_dataset` in `train_model.py`**

Replace the existing `build_full_dataset` function:

```python
def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    down_thresh: float = 0.03,
    extra_csv_paths: list = None,
) -> Tuple[Tuple, Tuple]:
    """Fetch all tickers, combine, and time-split 80/20.

    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    tickers = load_tickers(csv_path)
    if extra_csv_paths:
        for extra in extra_csv_paths:
            tickers += load_tickers(extra)
        tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    frames = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] Processing tickers...", flush=True)
        df = build_ticker_dataset(ticker, horizon=horizon,
                                  up_thresh=up_thresh, down_thresh=down_thresh)
        if df is not None:
            frames.append(df)
        else:
            skipped += 1

    print(f"\nUsed {len(frames)} tickers, skipped {skipped}")
    combined = pd.concat(frames).sort_index()

    print("\nClass distribution (full dataset):")
    print(combined['target'].value_counts().sort_index().to_string())

    X = combined[FEATURE_NAMES]
    y = combined['target']
    split = int(len(combined) * 0.8)
    return (X.iloc[:split], y.iloc[:split]), (X.iloc[split:], y.iloc[split:])
```

- [ ] **Step 4: Update `build_pipeline` hyperparameters in `train_model.py`**

Replace `n_estimators=300, num_leaves=63` with `n_estimators=500, num_leaves=95`:

```python
def build_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LGBMClassifier(
            n_estimators=500,
            num_leaves=95,
            learning_rate=0.05,
            class_weight='balanced',
            min_child_samples=20,
            n_jobs=2,
            random_state=42,
            verbose=-1,
        ))
    ])
```

- [ ] **Step 5: Update `train_and_save` signature and add precision/recall table**

Replace the existing `train_and_save` function:

```python
def train_and_save(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    down_thresh: float = 0.03,
    clip: float = 0.30,
    output_path: str = 'stock_model.joblib',
    extra_csv_paths: list = None,
) -> None:
    """Full training run: load data, fit pipeline, evaluate, save model."""
    print(f"\n=== AlphaPulse Model Training — {horizon}-day horizon ===")
    print(f"Ticker source : {csv_path}")
    print(f"Thresholds    : UP > {up_thresh*100:.0f}% / DOWN < -{down_thresh*100:.0f}%")
    print(f"Clip range    : ±{clip*100:.0f}%")
    print(f"Output path   : {output_path}\n")

    (X_train, y_train), (X_test, y_test) = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        down_thresh=down_thresh,
        extra_csv_paths=extra_csv_paths,
    )

    print(f"\nTraining rows : {len(X_train):,}")
    print(f"Test rows     : {len(X_test):,}")
    print("\nClass distribution (train set):")
    print(y_train.value_counts().sort_index().to_string())

    pipeline = build_pipeline()
    print("\nFitting pipeline — this may take a few minutes...")
    pipeline.fit(X_train, y_train)
    print("Fitting complete.")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print("\n=== Held-out Test Set Evaluation ===")
    label_names = [f"DOWN>{down_thresh*100:.0f}%", "STABLE", f"UP>{up_thresh*100:.0f}%"]
    print(classification_report(y_test, y_pred, target_names=label_names))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log-loss : {log_loss(y_test, y_proba):.4f}")

    # Precision/recall sweep for UP class
    classes_list = list(pipeline.classes_)
    up_idx = classes_list.index(1)
    y_proba_up = y_proba[:, up_idx]
    y_test_bin = (y_test == 1).astype(int)
    from sklearn.metrics import precision_score, recall_score
    thresholds = np.arange(0.30, 0.61, 0.05)
    print(f"\n=== UP class Precision / Recall sweep ===")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Signals':>10}")
    for t in thresholds:
        y_pred_t = (y_proba_up >= t).astype(int)
        n_sig = int(y_pred_t.sum())
        prec = precision_score(y_test_bin, y_pred_t, zero_division=0)
        rec = recall_score(y_test_bin, y_pred_t, zero_division=0)
        print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {n_sig:>10}")

    model_data = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'horizon': horizon,
        'up_thresh': up_thresh,
        'down_thresh': down_thresh,
        'description': (
            f'3-class stock predictor. '
            f'Classes: -1=DOWN>{down_thresh*100:.0f}%, 0=STABLE, 1=UP>{up_thresh*100:.0f}%. '
            f'Horizon: {horizon} trading days.'
        ),
    }
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")
```

- [ ] **Step 6: Update `__main__` block for dual model training with SGX support**

Replace the existing `if __name__ == '__main__':` block:

```python
if __name__ == '__main__':
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _hk_csv = find_ticker_csv(_data_dir)
    _sgx_csv = os.path.join(_data_dir, 'sgx_tickers.csv')
    _extra = [_sgx_csv] if os.path.exists(_sgx_csv) else None
    if _extra:
        print(f"SGX tickers found: {_sgx_csv}")

    train_and_save(
        _hk_csv,
        horizon=5,
        up_thresh=0.02,
        down_thresh=0.02,
        clip=0.20,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_model_5d.joblib'),
        extra_csv_paths=_extra,
    )
    train_and_save(
        _hk_csv,
        horizon=14,
        up_thresh=0.05,
        down_thresh=0.05,
        clip=0.30,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_model_14d.joblib'),
        extra_csv_paths=_extra,
    )
```

- [ ] **Step 7: Run all train_model tests**

```bash
pytest tests/test_train_model.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: dual-model training pipeline with SGX support and precision/recall sweep"
```

---

## Task 6: Update `app.py` — dual model loading and inference

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace model loading at the top of `app.py`**

Replace the existing model loading block (lines 15–20):

```python
MODEL_5D_PATH = os.path.join(_here, 'stock_model_5d.joblib')
MODEL_14D_PATH = os.path.join(_here, 'stock_model_14d.joblib')


def _load_model(path):
    try:
        d = joblib.load(path)
        return d['model'], d['features']
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None, []


model_5d, feature_names_5d = _load_model(MODEL_5D_PATH)
model_14d, feature_names_14d = _load_model(MODEL_14D_PATH)
```

Also **remove** the old `MODEL_PATH` line and the old `try/except joblib.load` block entirely.

- [ ] **Step 2: Update `_build_prediction` to accept model as argument**

Replace the existing `_build_prediction` function:

```python
def _build_prediction(ticker: str, mdl, feat_names: list) -> dict | None:
    try:
        data = fetch_latest_data(ticker)
        if data is None:
            return None
        processed = calculate_technical_indicators(data)
        if processed.empty:
            return None
        latest = processed[feat_names].iloc[-1:].values
        return build_prediction_response(ticker, mdl, latest, data)
    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None
```

- [ ] **Step 3: Add `_build_dual_prediction` for the scan**

Add after `_build_prediction`:

```python
def _build_dual_prediction(ticker: str) -> dict | None:
    try:
        data = fetch_latest_data(ticker)
        if data is None:
            return None
        processed = calculate_technical_indicators(data)
        if processed.empty:
            return None

        result = {
            "ticker": ticker,
            "current_price": float(data['Adj_Close'].iloc[-1]),
            "last_updated": str(data.index[-1].date()),
        }

        if model_5d is not None:
            latest = processed[feature_names_5d].iloc[-1:].values
            r5 = build_prediction_response(ticker, model_5d, latest, data)
            result.update({
                "signal_5d": r5["signal"],
                "confidence_up_5d": r5["confidence_up_3pct"],
                "confidence_down_5d": r5["confidence_down_3pct"],
                "edge_ratio_5d": r5["edge_ratio"],
                "prediction_5d": r5["prediction"],
            })

        if model_14d is not None:
            latest = processed[feature_names_14d].iloc[-1:].values
            r14 = build_prediction_response(ticker, model_14d, latest, data)
            result.update({
                "signal_14d": r14["signal"],
                "confidence_up_14d": r14["confidence_up_3pct"],
                "confidence_down_14d": r14["confidence_down_3pct"],
                "edge_ratio_14d": r14["edge_ratio"],
                "prediction_14d": r14["prediction"],
            })

        return result
    except Exception as e:
        print(f"Dual prediction error for {ticker}: {e}")
        return None
```

- [ ] **Step 4: Update `_result_html` to accept a `label` parameter**

Change the function signature from `def _result_html(r: dict) -> str:` to `def _result_html(r: dict, label: str = "") -> str:`.

In the header div of the HTML, add the label above the ticker. Replace the inner `<div>` block that shows the ticker:

```python
    label_html = (
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#888;margin-bottom:4px;">{label}</div>'
        if label else ""
    )
    return f"""
<div style="border:1px solid #b8b2aa;padding:18px 20px;background:#faf8f4;{_MONO}margin:4px 0;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              border-bottom:1px solid #d0cac2;padding-bottom:12px;margin-bottom:14px;">
    <div>
      {label_html}
      <div style="font-size:24px;font-weight:700;letter-spacing:1px;color:#1a1a18;">{r['ticker']}</div>
      <div style="font-size:16px;color:#555;margin-top:3px;">{r['current_price']:.3f}</div>
    </div>
    ...rest of existing HTML unchanged...
```

Fully replace `_result_html` with this complete version:

```python
def _result_html(r: dict, label: str = "") -> str:
    up_pct = round(r['confidence_up_3pct'] * 100)
    dn_pct = round(r['confidence_down_3pct'] * 100)
    if r['prediction'] == 1:
        sig_color = "#1e4d17"
    elif r['prediction'] == -1:
        sig_color = "#6b1616"
    else:
        sig_color = "#5a4a15"
    label_html = (
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#888;margin-bottom:4px;">{label}</div>'
        if label else ""
    )
    return f"""
<div style="border:1px solid #b8b2aa;padding:18px 20px;background:#faf8f4;{_MONO}margin:4px 0;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              border-bottom:1px solid #d0cac2;padding-bottom:12px;margin-bottom:14px;">
    <div>
      {label_html}
      <div style="font-size:24px;font-weight:700;letter-spacing:1px;color:#1a1a18;">{r['ticker']}</div>
      <div style="font-size:16px;color:#555;margin-top:3px;">{r['current_price']:.3f}</div>
    </div>
    <div style="text-align:right;">
      <div style="color:{sig_color};font-weight:700;font-size:16px;letter-spacing:1px;">{r['signal']}</div>
      <div style="font-size:13px;color:#888;margin-top:3px;">EDGE {r['edge_ratio']:.2f}&times;</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:15px;">
    <span style="width:180px;color:#555;flex-shrink:0;">UP &gt;3% in {r.get('horizon', 7)} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{up_pct}%;height:100%;background:#1e4d17;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{up_pct}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;font-size:15px;">
    <span style="width:180px;color:#555;flex-shrink:0;">DOWN &gt;3% in {r.get('horizon', 7)} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{dn_pct}%;height:100%;background:#6b1616;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{dn_pct}%</span>
  </div>
  <div style="font-size:12px;color:#aaa;text-align:right;margin-top:12px;
              border-top:1px solid #e5e0d9;padding-top:8px;">DATA AS OF {r['last_updated']}</div>
</div>"""
```

> Note: `build_prediction_response` doesn't currently add `horizon` to the result dict. The `r.get('horizon', 7)` fallback handles this gracefully with no changes needed to `build_prediction_response`.

- [ ] **Step 5: Update `analyze` function**

Replace the existing `analyze` function:

```python
def analyze(ticker: str, watchlist: list):
    ticker = _normalize_ticker(ticker)
    if not ticker:
        err = f'<p style="{_ERR_STYLE}">Enter a ticker symbol.</p>'
        return (err, "", *_wl_updates(watchlist))

    html_5d = html_14d = ""

    r5 = r14 = None
    if model_5d is not None:
        r5 = _build_prediction(ticker, model_5d, feature_names_5d)
        html_5d = (_result_html(r5, label="5-DAY FORECAST") if r5
                   else f'<p style="{_ERR_STYLE}">No data for {ticker} (5D).</p>')
    else:
        html_5d = f'<p style="{_ERR_STYLE}">5-day model not loaded.</p>'

    if model_14d is not None:
        r14 = _build_prediction(ticker, model_14d, feature_names_14d)
        html_14d = (_result_html(r14, label="14-DAY FORECAST") if r14
                    else f'<p style="{_ERR_STYLE}">No data for {ticker} (14D).</p>')
    else:
        html_14d = f'<p style="{_ERR_STYLE}">14-day model not loaded.</p>'

    if ticker not in watchlist and (r5 is not None or r14 is not None):
        watchlist = watchlist + [ticker]
        _save_watchlist(watchlist)

    return (html_5d, html_14d, *_wl_updates(watchlist))
```

- [ ] **Step 6: Update `scan_watchlist` to use dual predictions**

Replace the existing `scan_watchlist` function:

```python
def scan_watchlist(watchlist: list) -> str:
    if not watchlist:
        return f'<p style="{_ERR_STYLE}">Watchlist is empty.</p>'
    if model_5d is None and model_14d is None:
        return f'<p style="{_ERR_STYLE}">No models loaded.</p>'
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_build_dual_prediction, watchlist))
    valid = [r for r in results if r is not None]
    top10 = sorted(
        valid,
        key=lambda r: (r.get("confidence_up_5d", 0), r.get("edge_ratio_5d", 0)),
        reverse=True,
    )[:10]
    return _scan_html(top10)
```

- [ ] **Step 7: Replace `_scan_html` with dual-column version**

Replace the existing `_scan_html` function:

```python
def _scan_html(results: list) -> str:
    if not results:
        return f'<p style="{_ERR_STYLE}">No data available for any ticker in your watchlist.</p>'

    def _sig_color(pred):
        if pred == 1:
            return "#1e4d17"
        elif pred == -1:
            return "#6b1616"
        return "#5a4a15"

    rows = ""
    for i, r in enumerate(results):
        up_pct_5d = round(r.get("confidence_up_5d", 0) * 100)
        sig5_color = _sig_color(r.get("prediction_5d", 0))
        sig14_color = _sig_color(r.get("prediction_14d", 0))
        rows += (
            f'<tr style="border-bottom:1px solid #e5e0d9;">'
            f'<td style="padding:8px 10px;color:#aaa;">{i+1}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">{r["ticker"]}</td>'
            f'<td style="padding:8px 10px;color:#333;">{r["current_price"]:.3f}</td>'
            f'<td style="padding:8px 10px;color:{sig5_color};font-weight:700;">'
            f'{r.get("signal_5d", "N/A")}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">'
            f'{r.get("edge_ratio_5d", 0):.2f}&times;</td>'
            f'<td style="padding:8px 10px;color:{sig14_color};font-weight:700;">'
            f'{r.get("signal_14d", "N/A")}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">'
            f'{r.get("edge_ratio_14d", 0):.2f}&times;</td>'
            f'<td style="padding:8px 10px;color:#333;">{up_pct_5d}%</td>'
            f'</tr>'
        )
    th = ("padding:8px 10px;font-size:12px;text-transform:uppercase;"
          "letter-spacing:1px;text-align:left;border-bottom:2px solid #1a1a18;color:#1a1a18;")
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:14px;{_MONO}margin:4px 0;">'
        f'<thead><tr>'
        f'<th style="{th}">#</th>'
        f'<th style="{th}">TICKER</th>'
        f'<th style="{th}">PRICE</th>'
        f'<th style="{th}">SIGNAL 5D</th>'
        f'<th style="{th}">EDGE 5D</th>'
        f'<th style="{th}">SIGNAL 14D</th>'
        f'<th style="{th}">EDGE 14D</th>'
        f'<th style="{th}">UP CONF 5D</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
    )
```

- [ ] **Step 8: Run existing app tests to verify no regressions**

```bash
pytest tests/test_app.py -v 2>&1 | tail -20
```

Expected: all existing tests PASS. (`build_prediction_response`, `rank_scan_results`, and `SIGNAL_MAP` are all unchanged.)

- [ ] **Step 9: Commit**

```bash
git add app.py
git commit -m "feat: load dual models and render side-by-side 5D/14D predictions"
```

---

## Task 7: Update Gradio UI wiring in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update the Analyze section in the Gradio UI block**

Find the `# ── Analyze` section in the `with gr.Blocks(...) as demo:` block.

Replace the single `result_out` line:
```python
    result_out = gr.HTML(value="")
```

With two outputs inside a row:
```python
    with gr.Row():
        result_out_5d = gr.HTML(value="")
        result_out_14d = gr.HTML(value="")
```

- [ ] **Step 2: Update the wiring section**

Find the `# ── Wiring` section at the bottom of the `with gr.Blocks` block.

Replace the existing `_wl_outs` definition and analyze wiring:

Old:
```python
    _wl_outs = [watchlist_state, remove_cg]

    analyze_btn.click(
        fn=analyze,
        inputs=[ticker_in, watchlist_state],
        outputs=[result_out] + _wl_outs,
    )
    ticker_in.submit(
        fn=analyze,
        inputs=[ticker_in, watchlist_state],
        outputs=[result_out] + _wl_outs,
    )
```

New:
```python
    _wl_outs = [watchlist_state, remove_cg]
    _analyze_outs = [result_out_5d, result_out_14d] + _wl_outs

    analyze_btn.click(
        fn=analyze,
        inputs=[ticker_in, watchlist_state],
        outputs=_analyze_outs,
    )
    ticker_in.submit(
        fn=analyze,
        inputs=[ticker_in, watchlist_state],
        outputs=_analyze_outs,
    )
```

Leave all other wiring (add_btn, csv_btn, remove_btn, scan_btn) unchanged.

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: wire Gradio UI for dual 5D/14D result cards in analyze tab"
```

---

## Task 8: Final smoke test

- [ ] **Step 1: Run the complete test suite one final time**

```bash
pytest tests/ -v
```

Expected: all tests PASS with no warnings about missing model files.

- [ ] **Step 2: Verify `train_model.py` syntax**

```bash
python -c "import train_model; print('FEATURE_NAMES count:', len(train_model.FEATURE_NAMES))"
```

Expected output: `FEATURE_NAMES count: 21`

- [ ] **Step 3: Verify `app.py` imports cleanly**

```bash
python -c "import app; print('model_5d loaded:', app.model_5d is not None); print('model_14d loaded:', app.model_14d is not None)"
```

Expected output:
```
model_5d loaded: False
model_14d loaded: False
```
(False is correct — models don't exist yet until you retrain.)

- [ ] **Step 4: Final commit**

```bash
git add -A
git status
```

Review — should show no unexpected files. Then:

```bash
git commit -m "chore: finalize dual-model AlphaPulse improvement" --allow-empty
```

---

## Ready to Train

After implementation is complete, run training:

```bash
cd /Users/simgsr/Documents/python_project/yf_price_prediction
source venv/bin/activate

# Optional: add data/sgx_tickers.csv (one ticker per line, e.g. U96.SI) before running

python train_model.py
```

This will:
1. Train the 5-day model → `stock_model_5d.joblib`
2. Train the 14-day model → `stock_model_14d.joblib`
3. Print precision/recall sweep for each model so you can choose your UP confidence threshold

Then launch the app:
```bash
python app.py
```
