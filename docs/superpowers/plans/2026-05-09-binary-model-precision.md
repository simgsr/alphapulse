# Binary Model Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve UP signal precision by switching both models to binary classification (UP>3% vs NOT-UP) and fixing ATR/Stochastic to use actual OHLC High/Low instead of close-only approximations.

**Architecture:** Three files change in dependency order — `get_price_data.py` (data layer) → `train_model.py` (training layer) → `app.py` (inference/UI layer). Tests for each file are updated alongside the file they cover. No new files are created.

**Tech Stack:** Python, yfinance, LightGBM, scikit-learn, pandas, numpy, Gradio, pytest

---

## Task 1: Fix OHLC data fetching and ATR/Stochastic indicators

**Files:**
- Modify: `get_price_data.py`
- Modify: `tests/test_get_price_data.py`

- [ ] **Step 1: Update `_make_df` test helper to include High/Low columns**

In `tests/test_get_price_data.py`, replace the `_make_df` function:

```python
def _make_df(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(1)
    prices = 100.0 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    highs = prices * 1.01
    lows = prices * 0.99
    return pd.DataFrame(
        {
            "Adj_Open": prices * 0.999,
            "Adj_High": highs,
            "Adj_Low": lows,
            "Adj_Close": prices,
            "Adj_Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
```

Also replace the flat-price test helper inside `test_sma_50_ratio_near_one_for_flat_price`:

```python
def test_sma_50_ratio_near_one_for_flat_price():
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    flat = pd.DataFrame(
        {
            "Adj_Open": np.full(n, 100.0),
            "Adj_High": np.full(n, 100.0),
            "Adj_Low": np.full(n, 100.0),
            "Adj_Close": np.full(n, 100.0),
            "Adj_Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    df = calculate_technical_indicators(flat)
    assert (df['SMA_50_ratio'] - 1.0).abs().max() < 1e-9
```

- [ ] **Step 2: Run tests to confirm they fail on the old code**

```
cd /Users/simgsr/Documents/python_project/yf_price_prediction
python -m pytest tests/test_get_price_data.py -v 2>&1 | tail -20
```

Expected: Several ERRORS because `calculate_technical_indicators` tries `df['Adj_High']` which doesn't exist yet.

- [ ] **Step 3: Update `fetch_latest_data` to return full OHLC**

In `get_price_data.py`, replace the body of `fetch_latest_data`:

```python
def fetch_latest_data(ticker, period="120d"):
    """
    Fetch the latest historical data for a ticker.
    Returns a cleaned DataFrame with OHLCV columns suitable for feature engineering.
    120d (~84 trading days) ensures SMA_50 is computable.
    """
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return None

        # Handle MultiIndex columns from yfinance 1.3.0+
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.columns = ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']

        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None
```

- [ ] **Step 4: Fix ATR to use True Range (actual High/Low)**

In `get_price_data.py`, inside `calculate_technical_indicators`, replace the ATR block:

Old:
```python
    # ATR ratio (14-day, close-based simplified True Range)
    tr = df['Adj_Close'].diff().abs()
    atr_14 = tr.rolling(window=14).mean()
    df['ATR_ratio'] = (atr_14 / df['Adj_Close'].replace(0, np.nan)).fillna(0)
```

New:
```python
    # ATR ratio (14-day, True Range using actual High/Low)
    hl = df['Adj_High'] - df['Adj_Low']
    hc = (df['Adj_High'] - df['Adj_Close'].shift(1)).abs()
    lc = (df['Adj_Low'] - df['Adj_Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    df['ATR_ratio'] = (atr_14 / df['Adj_Close'].replace(0, np.nan)).fillna(0)
```

- [ ] **Step 5: Fix Stochastic %K/%D to use actual High/Low**

In `get_price_data.py`, inside `calculate_technical_indicators`, replace the Stochastic block:

Old:
```python
    # Stochastic %K/%D (14-day, close-based approximation)
    low_14 = df['Adj_Close'].rolling(window=14).min()
    high_14 = df['Adj_Close'].rolling(window=14).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    df['Stoch_K'] = ((df['Adj_Close'] - low_14) / range_14 * 100).fillna(50)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
```

New:
```python
    # Stochastic %K/%D (14-day, actual High/Low rolling range)
    low_14 = df['Adj_Low'].rolling(window=14).min()
    high_14 = df['Adj_High'].rolling(window=14).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    df['Stoch_K'] = ((df['Adj_Close'] - low_14) / range_14 * 100).fillna(50)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
```

- [ ] **Step 6: Run tests and confirm they pass**

```
python -m pytest tests/test_get_price_data.py -v
```

Expected: all green. ATR will be larger (True Range > |diff(Close)|), so `test_atr_ratio_positive` passes. Stoch range is wider, so bounds tests still pass.

- [ ] **Step 7: Commit**

```bash
git add get_price_data.py tests/test_get_price_data.py
git commit -m "feat: fetch full OHLC; fix ATR (True Range) and Stochastic with actual H/L"
```

---

## Task 2: Replace `discretize_return` with `binarize_return` in `train_model.py`

**Files:**
- Modify: `train_model.py`

- [ ] **Step 1: Replace `discretize_return` with `binarize_return`**

In `train_model.py`, replace the entire `discretize_return` function:

Old:
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

New:
```python
def binarize_return(r: float, thresh: float = 0.03) -> int:
    """Map a forward return to a binary label.

    Returns 1 if r > thresh (UP), 0 otherwise (NOT-UP).
    """
    return 1 if r > thresh else 0
```

- [ ] **Step 2: Remove `down_thresh` from `build_ticker_dataset`**

Replace the function signature and the `target` assignment:

Old signature:
```python
def build_ticker_dataset(
    ticker: str,
    period: str = '5y',
    horizon: int = 7,
    up_thresh: float = 0.03,
    down_thresh: float = 0.03,
    clip: float = 0.30,
) -> Optional[pd.DataFrame]:
```

New signature:
```python
def build_ticker_dataset(
    ticker: str,
    period: str = '5y',
    horizon: int = 7,
    up_thresh: float = 0.03,
    clip: float = 0.30,
) -> Optional[pd.DataFrame]:
```

Old target assignment (inside the function body):
```python
    df['target'] = df['forward_return'].apply(
        lambda r: discretize_return(r, up_thresh=up_thresh, down_thresh=down_thresh)
    )
```

New:
```python
    df['target'] = df['forward_return'].apply(
        lambda r: binarize_return(r, thresh=up_thresh)
    )
```

- [ ] **Step 3: Remove `down_thresh` from `build_full_dataset`**

Old signature:
```python
def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    down_thresh: float = 0.03,
    extra_csv_paths: list = None,
) -> Tuple[Tuple, Tuple]:
```

New signature:
```python
def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    extra_csv_paths: list = None,
) -> Tuple[Tuple, Tuple]:
```

Old call inside the function:
```python
        df = build_ticker_dataset(ticker, horizon=horizon,
                                  up_thresh=up_thresh, down_thresh=down_thresh)
```

New:
```python
        df = build_ticker_dataset(ticker, horizon=horizon, up_thresh=up_thresh)
```

- [ ] **Step 4: Update `build_pipeline` to accept `scale_pos_weight`**

Replace the `build_pipeline` function:

Old:
```python
def build_pipeline() -> Pipeline:
    """Build the unfitted sklearn Pipeline.

    RobustScaler is resistant to HK small-cap outliers.
    LGBMClassifier with class_weight='balanced' handles the dominant STABLE class.
    LightGBM produces well-calibrated probabilities natively (no CalibratedClassifierCV needed).
    """
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

New:
```python
def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build the unfitted sklearn Pipeline.

    RobustScaler is resistant to HK small-cap outliers.
    scale_pos_weight = count(NOT-UP) / count(UP) balances the binary imbalance.
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LGBMClassifier(
            n_estimators=500,
            num_leaves=95,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            min_child_samples=20,
            n_jobs=2,
            random_state=42,
            verbose=-1,
        ))
    ])
```

- [ ] **Step 5: Update `train_and_save` — remove `down_thresh`, compute `scale_pos_weight`, fix label names and sweep**

Old signature:
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
```

New signature:
```python
def train_and_save(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    clip: float = 0.30,
    output_path: str = 'stock_model.joblib',
    extra_csv_paths: list = None,
) -> None:
```

Inside `train_and_save`, replace the header print block:

Old:
```python
    print(f"\n=== AlphaPulse Model Training — {horizon}-day horizon ===")
    print(f"Ticker source : {csv_path}")
    print(f"Thresholds    : UP > {up_thresh*100:.0f}% / DOWN < -{down_thresh*100:.0f}%")
    print(f"Clip range    : ±{clip*100:.0f}%")
    print(f"Output path   : {output_path}\n")
```

New:
```python
    print(f"\n=== AlphaPulse Model Training — {horizon}-day horizon ===")
    print(f"Ticker source : {csv_path}")
    print(f"UP threshold  : > {up_thresh*100:.0f}%  (binary: NOT-UP otherwise)")
    print(f"Clip range    : ±{clip*100:.0f}%")
    print(f"Output path   : {output_path}\n")
```

Replace the `build_full_dataset` call:

Old:
```python
    (X_train, y_train), (X_test, y_test) = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        down_thresh=down_thresh,
        extra_csv_paths=extra_csv_paths,
    )
```

New:
```python
    (X_train, y_train), (X_test, y_test) = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        extra_csv_paths=extra_csv_paths,
    )
```

Replace the pipeline build and fit block:

Old:
```python
    pipeline = build_pipeline()
    print("\nFitting pipeline — this may take a few minutes...")
    pipeline.fit(X_train, y_train)
```

New:
```python
    spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"\nscale_pos_weight : {spw:.2f}  (NOT-UP / UP ratio)")
    pipeline = build_pipeline(scale_pos_weight=spw)
    print("Fitting pipeline — this may take a few minutes...")
    pipeline.fit(X_train, y_train)
```

Replace the classification report label names:

Old:
```python
    label_names = [f"DOWN>{down_thresh*100:.0f}%", "STABLE", f"UP>{up_thresh*100:.0f}%"]
    print(classification_report(y_test, y_pred, target_names=label_names))
```

New:
```python
    label_names = ["NOT-UP", f"UP>{up_thresh*100:.0f}%"]
    print(classification_report(y_test, y_pred, target_names=label_names))
```

Replace the UP class precision/recall sweep range (0.30–0.60 → 0.50–0.80):

Old:
```python
    thresholds = np.arange(0.30, 0.61, 0.05)
```

New:
```python
    thresholds = np.arange(0.50, 0.81, 0.05)
```

Replace the `up_idx` lookup (binary model has classes `[0, 1]` so index 1 is UP):

Old:
```python
    classes_list = list(pipeline.classes_)
    up_idx = classes_list.index(1)
    y_proba_up = y_proba[:, up_idx]
    y_test_bin = (y_test == 1).astype(int)
```

New:
```python
    classes_list = list(pipeline.classes_)
    up_idx = classes_list.index(1)
    y_proba_up = y_proba[:, up_idx]
    y_test_bin = y_test  # already binary
```

Replace the `model_data` dict:

Old:
```python
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
```

New:
```python
    model_data = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'horizon': horizon,
        'up_thresh': up_thresh,
        'binary': True,
        'description': (
            f'Binary stock predictor. '
            f'Classes: 0=NOT-UP, 1=UP>{up_thresh*100:.0f}%. '
            f'Horizon: {horizon} trading days.'
        ),
    }
```

- [ ] **Step 6: Update the `__main__` block — remove `down_thresh`, set `up_thresh=0.03` for both**

Old:
```python
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

New:
```python
    train_and_save(
        _hk_csv,
        horizon=5,
        up_thresh=0.03,
        clip=0.20,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_model_5d.joblib'),
        extra_csv_paths=_extra,
    )
    train_and_save(
        _hk_csv,
        horizon=14,
        up_thresh=0.03,
        clip=0.30,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_model_14d.joblib'),
        extra_csv_paths=_extra,
    )
```

- [ ] **Step 7: Commit**

```bash
git add train_model.py
git commit -m "feat: replace discretize_return with binarize_return; binary pipeline with scale_pos_weight"
```

---

## Task 3: Update `test_train_model.py` for binary labels

**Files:**
- Modify: `tests/test_train_model.py`

- [ ] **Step 1: Write failing tests for `binarize_return`**

Replace the entire `TestDiscretizeReturn` class with `TestBinarizeReturn`:

```python
class TestBinarizeReturn:
    def test_up_above_thresh(self):
        from train_model import binarize_return
        assert binarize_return(0.04) == 1

    def test_up_just_above_thresh(self):
        from train_model import binarize_return
        assert binarize_return(0.031) == 1

    def test_exact_thresh_is_not_up(self):
        from train_model import binarize_return
        assert binarize_return(0.03) == 0

    def test_zero_is_not_up(self):
        from train_model import binarize_return
        assert binarize_return(0.0) == 0

    def test_negative_is_not_up(self):
        from train_model import binarize_return
        assert binarize_return(-0.05) == 0

    def test_custom_thresh_up(self):
        from train_model import binarize_return
        assert binarize_return(0.021, thresh=0.02) == 1

    def test_custom_thresh_boundary(self):
        from train_model import binarize_return
        assert binarize_return(0.02, thresh=0.02) == 0
```

Also update the import at the top of the file — change `discretize_return` to `binarize_return`:

Old:
```python
from train_model import discretize_return, load_tickers, FEATURE_NAMES as _NEW_FEATURES
```

New:
```python
from train_model import binarize_return, load_tickers, FEATURE_NAMES as _NEW_FEATURES
```

- [ ] **Step 2: Run failing tests to confirm**

```
python -m pytest tests/test_train_model.py::TestBinarizeReturn -v
```

Expected: FAIL — `binarize_return` not importable yet (if Task 2 not done), or if Task 2 is done: all PASS. If all pass, move on.

- [ ] **Step 3: Update `TestBuildTickerDataset` — fix target class set and remove `down_thresh`**

Replace `test_target_values_are_valid_classes`:

Old:
```python
    def test_target_values_are_valid_classes(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert set(result["target"].unique()).issubset({-1, 0, 1})
```

New:
```python
    def test_target_values_are_binary(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert set(result["target"].unique()).issubset({0, 1})
```

Replace `test_5d_horizon_uses_shift_5` (remove `down_thresh`):

Old:
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
```

New:
```python
    def test_5d_horizon_uses_shift_5(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk", horizon=5, up_thresh=0.03, clip=0.20)
        assert result is not None
        assert result["forward_return"].max() <= 0.20
        assert result["forward_return"].min() >= -0.20
```

Replace `test_14d_horizon_uses_shift_14` (remove `down_thresh`, fix target set):

Old:
```python
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

New:
```python
    def test_14d_horizon_uses_shift_14(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk", horizon=14, up_thresh=0.03, clip=0.30)
        assert result is not None
        assert set(result["target"].unique()).issubset({0, 1})
```

- [ ] **Step 4: Update `TestBuildFullDataset.test_horizon_param_forwarded` — remove `down_thresh`**

Old:
```python
    def test_horizon_param_forwarded(self, tmp_path):
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text("0001.hk\n")

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

New:
```python
    def test_horizon_param_forwarded(self, tmp_path):
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text("0001.hk\n")

        df_a = self._make_labeled_df(300, "2020-01-01")
        received_kwargs = {}

        def mock_build(ticker, **kwargs):
            received_kwargs.update(kwargs)
            return df_a

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            build_full_dataset(str(csv_file), horizon=5, up_thresh=0.03)

        assert received_kwargs.get('horizon') == 5
        assert received_kwargs.get('up_thresh') == 0.03
```

- [ ] **Step 5: Update `_make_labeled_df` in `TestBuildFullDataset` — target values**

The `_make_labeled_df` helper sets `data['target'] = 0` which is valid for both 3-class and binary (0 = NOT-UP). No change needed here.

- [ ] **Step 6: Run all train_model tests and confirm green**

```
python -m pytest tests/test_train_model.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_train_model.py
git commit -m "test: update train_model tests for binary labels and binarize_return"
```

---

## Task 4: Update `app.py` for binary model inference and UI

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update `SIGNAL_MAP` and add `UP_THRESHOLD`**

Replace the `SIGNAL_MAP` dict and add `UP_THRESHOLD` below it:

Old:
```python
SIGNAL_MAP = {
    1: "UP > 3%",
    0: "STABLE",
    -1: "DOWN > 3%",
}
```

New:
```python
SIGNAL_MAP = {
    1: "UP >3%",
    0: "NO SIGNAL",
}

UP_THRESHOLD = 0.5
```

- [ ] **Step 2: Update `build_prediction_response` — remove DOWN, fix edge ratio**

Replace the entire `build_prediction_response` function:

Old:
```python
def build_prediction_response(ticker: str, mdl, features_array: np.ndarray, raw_data) -> dict:
    prediction = int(mdl.predict(features_array)[0])
    probabilities = mdl.predict_proba(features_array)[0].tolist()
    classes = mdl.classes_.tolist()
    prob_dict = {str(int(c)): round(p, 4) for c, p in zip(classes, probabilities)}

    confidence_up_3pct = round(prob_dict.get('1', 0.0), 4)
    confidence_down_3pct = round(prob_dict.get('-1', 0.0), 4)
    p_down = confidence_down_3pct
    edge_ratio = round(confidence_up_3pct / p_down, 2) if p_down > 0 else 99.0

    return {
        "ticker": ticker,
        "prediction": prediction,
        "signal": SIGNAL_MAP.get(prediction, "UNKNOWN"),
        "confidence_up_3pct": confidence_up_3pct,
        "confidence_down_3pct": confidence_down_3pct,
        "edge_ratio": edge_ratio,
        "probabilities": prob_dict,
        "current_price": float(raw_data['Adj_Close'].iloc[-1]),
        "last_updated": str(raw_data.index[-1].date()),
    }
```

New:
```python
def build_prediction_response(ticker: str, mdl, features_array: np.ndarray, raw_data) -> dict:
    prediction = int(mdl.predict(features_array)[0])
    probabilities = mdl.predict_proba(features_array)[0].tolist()
    classes = mdl.classes_.tolist()
    prob_dict = {str(int(c)): round(p, 4) for c, p in zip(classes, probabilities)}

    confidence_up_3pct = round(prob_dict.get('1', 0.0), 4)
    edge_ratio = round(confidence_up_3pct / UP_THRESHOLD, 2)

    return {
        "ticker": ticker,
        "prediction": prediction,
        "signal": SIGNAL_MAP.get(prediction, "NO SIGNAL"),
        "confidence_up_3pct": confidence_up_3pct,
        "edge_ratio": edge_ratio,
        "probabilities": prob_dict,
        "current_price": float(raw_data['Adj_Close'].iloc[-1]),
        "last_updated": str(raw_data.index[-1].date()),
    }
```

- [ ] **Step 3: Update `_build_dual_prediction` — remove DOWN confidence fields**

Replace the two `result.update(...)` blocks inside `_build_dual_prediction`:

Old (5d block):
```python
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
```

New (5d block):
```python
        if model_5d is not None:
            latest = processed[feature_names_5d].iloc[-1:].values
            r5 = build_prediction_response(ticker, model_5d, latest, data)
            result.update({
                "signal_5d": r5["signal"],
                "confidence_up_5d": r5["confidence_up_3pct"],
                "edge_ratio_5d": r5["edge_ratio"],
                "prediction_5d": r5["prediction"],
            })
```

Old (14d block):
```python
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
```

New (14d block):
```python
        if model_14d is not None:
            latest = processed[feature_names_14d].iloc[-1:].values
            r14 = build_prediction_response(ticker, model_14d, latest, data)
            result.update({
                "signal_14d": r14["signal"],
                "confidence_up_14d": r14["confidence_up_3pct"],
                "edge_ratio_14d": r14["edge_ratio"],
                "prediction_14d": r14["prediction"],
            })
```

- [ ] **Step 4: Update `_result_html` — remove DOWN bar, simplify signal color**

Replace the entire `_result_html` function:

Old:
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
    horizon = r.get('horizon', 7)
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
    <span style="width:180px;color:#555;flex-shrink:0;">UP &gt;3% in {horizon} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{up_pct}%;height:100%;background:#1e4d17;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{up_pct}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;font-size:15px;">
    <span style="width:180px;color:#555;flex-shrink:0;">DOWN &gt;3% in {horizon} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{dn_pct}%;height:100%;background:#6b1616;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{dn_pct}%</span>
  </div>
  <div style="font-size:12px;color:#aaa;text-align:right;margin-top:12px;
              border-top:1px solid #e5e0d9;padding-top:8px;">DATA AS OF {r['last_updated']}</div>
</div>"""
```

New:
```python
def _result_html(r: dict, label: str = "") -> str:
    up_pct = round(r['confidence_up_3pct'] * 100)
    sig_color = "#1e4d17" if r['prediction'] == 1 else "#5a4a15"
    label_html = (
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#888;margin-bottom:4px;">{label}</div>'
        if label else ""
    )
    horizon = r.get('horizon', 7)
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
  <div style="display:flex;align-items:center;gap:10px;font-size:15px;">
    <span style="width:180px;color:#555;flex-shrink:0;">UP &gt;3% in {horizon} days</span>
    <div style="flex:1;height:9px;background:#e5e0d9;border:1px solid #c8c2ba;">
      <div style="width:{up_pct}%;height:100%;background:#1e4d17;"></div>
    </div>
    <span style="width:42px;text-align:right;font-weight:700;color:#1a1a18;">{up_pct}%</span>
  </div>
  <div style="font-size:12px;color:#aaa;text-align:right;margin-top:12px;
              border-top:1px solid #e5e0d9;padding-top:8px;">DATA AS OF {r['last_updated']}</div>
</div>"""
```

- [ ] **Step 5: Update `_scan_html` — remove DOWN column, simplify signal color**

Replace the entire `_scan_html` function:

Old:
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

New:
```python
def _scan_html(results: list) -> str:
    if not results:
        return f'<p style="{_ERR_STYLE}">No data available for any ticker in your watchlist.</p>'

    rows = ""
    for i, r in enumerate(results):
        up_5d = round(r.get("confidence_up_5d", 0) * 100)
        up_14d = round(r.get("confidence_up_14d", 0) * 100)
        sig5_color = "#1e4d17" if r.get("prediction_5d") == 1 else "#5a4a15"
        sig14_color = "#1e4d17" if r.get("prediction_14d") == 1 else "#5a4a15"
        rows += (
            f'<tr style="border-bottom:1px solid #e5e0d9;">'
            f'<td style="padding:8px 10px;color:#aaa;">{i+1}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">{r["ticker"]}</td>'
            f'<td style="padding:8px 10px;color:#333;">{r["current_price"]:.3f}</td>'
            f'<td style="padding:8px 10px;color:{sig5_color};font-weight:700;">'
            f'{r.get("signal_5d", "N/A")}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">{up_5d}%</td>'
            f'<td style="padding:8px 10px;color:{sig14_color};font-weight:700;">'
            f'{r.get("signal_14d", "N/A")}</td>'
            f'<td style="padding:8px 10px;font-weight:700;color:#1a1a18;">{up_14d}%</td>'
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
        f'<th style="{th}">UP CONF 5D</th>'
        f'<th style="{th}">SIGNAL 14D</th>'
        f'<th style="{th}">UP CONF 14D</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
    )
```

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: update app.py for binary model — remove DOWN bar, fix edge ratio, simplify scan"
```

---

## Task 5: Update `test_app.py` for binary model

**Files:**
- Modify: `tests/test_app.py`

- [ ] **Step 1: Update mock model and test helpers for binary classes**

Replace `_make_mock_model` at the top of the file:

Old:
```python
def _make_mock_model(prediction: int = 1):
    """Return a mock Pipeline whose predict/predict_proba mimic 3-class output."""
    m = MagicMock()
    m.predict.return_value = np.array([prediction])
    m.predict_proba.return_value = np.array([[0.15, 0.23, 0.62]])
    m.classes_ = np.array([-1.0, 0.0, 1.0])
    return m
```

New:
```python
def _make_mock_model(prediction: int = 1):
    """Return a mock Pipeline whose predict/predict_proba mimic binary output."""
    m = MagicMock()
    m.predict.return_value = np.array([prediction])
    m.predict_proba.return_value = np.array([[0.38, 0.62]])
    m.classes_ = np.array([0.0, 1.0])
    return m
```

- [ ] **Step 2: Run tests to confirm they fail on the old app.py**

```
python -m pytest tests/test_app.py -v 2>&1 | tail -20
```

Expected: multiple FAILs (SIGNAL_MAP assertions, DOWN probability, etc.).

- [ ] **Step 3: Update all `TestBuildPredictionResponse` tests**

Replace the entire `TestBuildPredictionResponse` class:

```python
class TestBuildPredictionResponse:
    def test_confidence_up_3pct_is_class_1(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        # proba: [0]=0.38, [1]=0.62 → confidence_up_3pct = P(1) = 0.62
        assert abs(result["confidence_up_3pct"] - 0.62) < 1e-4

    def test_signal_is_up_for_prediction_one(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert result["signal"] == "UP >3%"

    def test_signal_is_no_signal_for_prediction_zero(self):
        from app import build_prediction_response
        m = MagicMock()
        m.predict.return_value = np.array([0])
        m.predict_proba.return_value = np.array([[0.75, 0.25]])
        m.classes_ = np.array([0.0, 1.0])
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", m, features, mock_data)

        assert result["signal"] == "NO SIGNAL"

    def test_signal_map_is_binary(self):
        from app import SIGNAL_MAP
        assert SIGNAL_MAP[1] == "UP >3%"
        assert SIGNAL_MAP[0] == "NO SIGNAL"
        assert len(SIGNAL_MAP) == 2

    def test_response_contains_required_fields(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        for key in ("ticker", "prediction", "signal",
                    "confidence_up_3pct", "edge_ratio",
                    "probabilities", "current_price", "last_updated"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_down_not_in_response(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert "confidence_down_3pct" not in result

    def test_probabilities_keyed_by_binary_classes(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert set(result["probabilities"].keys()) == {"0", "1"}

    def test_edge_ratio_is_up_over_threshold(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        # P(UP) = 0.62, UP_THRESHOLD = 0.5 → edge_ratio = 0.62 / 0.5 = 1.24
        assert abs(result["edge_ratio"] - round(0.62 / 0.5, 2)) < 1e-4
```

- [ ] **Step 4: Run all app tests and confirm green**

```
python -m pytest tests/test_app.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all PASS across all three test files.

- [ ] **Step 6: Commit**

```bash
git add tests/test_app.py
git commit -m "test: update app tests for binary model — binary mock, edge ratio, SIGNAL_MAP"
```

---

## Self-review checklist

**Spec coverage:**
- §1 binary labels (binarize_return, thresh=0.03) → Task 2 Steps 1–2 ✓
- §2 OHLC fix (fetch_latest_data, ATR True Range, Stochastic H/L) → Task 1 Steps 3–5 ✓
- §3 scale_pos_weight (remove class_weight='balanced') → Task 2 Steps 4–5 ✓
- §3 binary=True in model_data → Task 2 Step 5 ✓
- §3 sweep range 0.50–0.80 → Task 2 Step 5 ✓
- §4 UP_THRESHOLD=0.5 constant → Task 4 Step 1 ✓
- §4 edge_ratio = confidence_up / UP_THRESHOLD → Task 4 Step 2 ✓
- §4 DOWN bar removed from result card → Task 4 Step 4 ✓
- §4 scan columns (SIGNAL xD, UP CONF xD) → Task 4 Step 5 ✓

**No placeholders** — every step has exact code.

**Type consistency** — `binarize_return` defined in Task 2 Step 1, used in Step 2 ✓; `UP_THRESHOLD` defined in Task 4 Step 1, used in Step 2 ✓; `scale_pos_weight` param added to `build_pipeline` in Step 4, passed in Step 5 ✓.
