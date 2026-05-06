# Feature Expansion + LightGBM Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand from 6 to 14 technical features and replace RandomForest with LightGBM to improve 3-class stock prediction accuracy (current: 41%, UP>3% recall: 0.01).

**Architecture:** Feature engineering lives in `get_price_data.py`; the feature name contract is declared in `train_model.py::FEATURE_NAMES`; the model pipeline is built in `train_model.py::build_pipeline()`. All three files change, plus `requirements.txt` and the test suite.

**Tech Stack:** Python 3.x, yfinance, scikit-learn Pipeline, LightGBM (`lgbm>=4.0.0`), pytest

---

## File Map

| File | Change |
|---|---|
| `get_price_data.py` | Add 8 new indicator columns; increase default fetch window to 120d |
| `train_model.py` | Expand `FEATURE_NAMES` to 14; replace RandomForest+CalibratedClassifierCV with LGBMClassifier |
| `requirements.txt` | Add `lightgbm>=4.0.0` |
| `tests/test_get_price_data.py` | New file — tests for every new indicator column |
| `tests/test_train_model.py` | Update `_make_labeled_df` and `test_features_subset_only` to use new 14-feature list |

---

## New Feature List (14 total)

```python
FEATURE_NAMES = [
    'SMA_5_ratio', 'SMA_20_ratio', 'SMA_50_ratio',   # trend position
    'RSI_14', 'RSI_7',                                 # momentum (two speeds)
    'MACD', 'MACD_hist',                               # trend momentum
    'BB_pct_b',                                        # mean-reversion signal
    'Volume_ratio_20',                                 # volume surge
    'Volatility_20',                                   # risk
    'Returns_1d', 'Returns_5d', 'Returns_10d', 'Returns_20d',  # multi-horizon momentum
]
```

---

## Task 1: Install LightGBM dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add lightgbm to requirements.txt**

Open `requirements.txt` and add the line:
```
lightgbm>=4.0.0
```
The full file should look like:
```
fastapi
uvicorn
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
lightgbm>=4.0.0
huggingface_hub>=0.23.0
python-multipart
pytest>=7.0.0
httpx>=0.24.0
```

- [ ] **Step 2: Install the dependency**

```bash
pip install lightgbm>=4.0.0
```

Expected: `Successfully installed lightgbm-...` (or "already satisfied")

- [ ] **Step 3: Verify import works**

```bash
python -c "import lightgbm; print(lightgbm.__version__)"
```

Expected: a version string like `4.3.0`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add lightgbm dependency"
```

---

## Task 2: Add new technical indicators to get_price_data.py

**Files:**
- Modify: `get_price_data.py`
- Create: `tests/test_get_price_data.py`

The problem: `calculate_technical_indicators` only produces 6 columns. We need 14. Also the default fetch window of `60d` (~42 trading days) is too short to compute SMA_50 (needs 50 days); increase to `120d`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_get_price_data.py`:

```python
import numpy as np
import pandas as pd
import pytest
from get_price_data import calculate_technical_indicators


def _make_df(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(1)
    prices = 100.0 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    return pd.DataFrame(
        {"Adj_Close": prices, "Adj_Volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def test_new_columns_present():
    df = calculate_technical_indicators(_make_df())
    for col in ['RSI_7', 'SMA_50_ratio', 'MACD', 'MACD_hist',
                'BB_pct_b', 'Volume_ratio_20', 'Returns_10d', 'Returns_20d']:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nans_after_calculation():
    df = calculate_technical_indicators(_make_df())
    assert df.isnull().sum().sum() == 0


def test_rsi_7_bounded():
    df = calculate_technical_indicators(_make_df())
    assert df['RSI_7'].between(0, 100).all()


def test_rsi_14_bounded():
    df = calculate_technical_indicators(_make_df())
    assert df['RSI_14'].between(0, 100).all()


def test_bb_pct_b_mostly_in_range():
    df = calculate_technical_indicators(_make_df(500))
    in_range = df['BB_pct_b'].between(0, 1).mean()
    assert in_range >= 0.85


def test_macd_hist_nonzero():
    df = calculate_technical_indicators(_make_df())
    assert df['MACD_hist'].abs().mean() > 0


def test_volume_ratio_20_positive():
    df = calculate_technical_indicators(_make_df())
    assert (df['Volume_ratio_20'] > 0).all()


def test_sma_50_ratio_near_one_for_flat_price():
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    flat = pd.DataFrame(
        {"Adj_Close": np.full(n, 100.0), "Adj_Volume": np.full(n, 1_000_000.0)},
        index=idx,
    )
    df = calculate_technical_indicators(flat)
    assert (df['SMA_50_ratio'] - 1.0).abs().max() < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/simgsr/Documents/simgsr-ntu/capstone/ntu_yfinance_deployment
python -m pytest tests/test_get_price_data.py -v
```

Expected: All 8 tests FAIL with `KeyError` or `AttributeError` (columns don't exist yet).

- [ ] **Step 3: Implement new indicators in get_price_data.py**

Replace the entire `calculate_technical_indicators` function and update `fetch_latest_data` default period:

```python
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_latest_data(ticker, period="120d"):
    """
    Fetch the latest historical data for a ticker.
    Returns a cleaned DataFrame suitable for feature engineering.
    120d (~84 trading days) ensures SMA_50 is computable.
    """
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return None

        # Handle MultiIndex columns from yfinance 1.3.0+
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Close', 'Volume']].copy()
        data.columns = ['Adj_Close', 'Adj_Volume']

        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def calculate_technical_indicators(df):
    """
    Calculate the 14 features used during model training and inference.
    Requires at least ~70 rows of data (SMA_50 + Returns_20d).
    """
    df = df.copy()
    delta = df['Adj_Close'].diff()

    # RSI (two speeds)
    for window, col in [(14, 'RSI_14'), (7, 'RSI_7')]:
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        df[col] = 100 - (100 / (1 + gain / loss))

    # SMA ratios (3 time scales)
    for window, col in [(5, 'SMA_5'), (20, 'SMA_20'), (50, 'SMA_50')]:
        df[col] = df['Adj_Close'].rolling(window=window).mean()
        df[f'{col}_ratio'] = df['Adj_Close'] / df[col]

    # MACD (12-26-9)
    ema12 = df['Adj_Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Adj_Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    macd_signal = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - macd_signal

    # Bollinger Band %B (20-day, 2 std)
    bb_std = df['Adj_Close'].rolling(window=20).std()
    bb_upper = df['SMA_20'] + 2 * bb_std
    bb_lower = df['SMA_20'] - 2 * bb_std
    df['BB_pct_b'] = (df['Adj_Close'] - bb_lower) / (bb_upper - bb_lower)

    # Volume ratio vs 20-day average
    df['Volume_ratio_20'] = df['Adj_Volume'] / df['Adj_Volume'].rolling(window=20).mean()

    # Volatility and multi-horizon returns
    df['Volatility_20'] = df['Adj_Close'].pct_change().rolling(window=20).std()
    df['Returns_1d'] = df['Adj_Close'].pct_change()
    df['Returns_5d'] = df['Adj_Close'].pct_change(5)
    df['Returns_10d'] = df['Adj_Close'].pct_change(10)
    df['Returns_20d'] = df['Adj_Close'].pct_change(20)

    return df.dropna()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_get_price_data.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add get_price_data.py tests/test_get_price_data.py
git commit -m "feat: expand technical indicators to 14 features, increase fetch window to 120d"
```

---

## Task 3: Update FEATURE_NAMES and switch to LightGBM in train_model.py

**Files:**
- Modify: `train_model.py`

The current `build_pipeline()` uses `RandomForestClassifier` wrapped in `CalibratedClassifierCV`. Replace with `LGBMClassifier` which has better native probability calibration and typically outperforms RF on tabular data.

- [ ] **Step 1: Update train_model.py**

Replace the existing `FEATURE_NAMES` list and `build_pipeline()` function. Also update the import block.

Full updated `train_model.py`:

```python
import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, log_loss, accuracy_score
from lightgbm import LGBMClassifier
from get_price_data import fetch_latest_data, calculate_technical_indicators

FEATURE_NAMES = [
    'SMA_5_ratio', 'SMA_20_ratio', 'SMA_50_ratio',
    'RSI_14', 'RSI_7',
    'MACD', 'MACD_hist',
    'BB_pct_b',
    'Volume_ratio_20',
    'Volatility_20', 'Returns_1d', 'Returns_5d', 'Returns_10d', 'Returns_20d',
]


def discretize_return(r: float) -> int:
    """Map a 7-day forward return to a 3-class label.

    Classes:
        1  — UP > 3%
        0  — STABLE (within ±3%)
       -1  — DOWN > 3%
    """
    if r > 0.03:
        return 1
    elif r >= -0.03:
        return 0
    else:
        return -1


def load_tickers(csv_path: str) -> list:
    """Return ticker symbols from a single-column CSV (no header required)."""
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].dropna().str.strip().tolist()


def build_ticker_dataset(ticker: str, period: str = '5y') -> Optional[pd.DataFrame]:
    """Fetch, engineer features, and label one ticker.

    Returns None if data is unavailable or fewer than 252 labeled rows remain.
    The 7-day forward return is clipped to ±30% to filter data errors and
    unadjusted corporate actions.
    """
    raw = fetch_latest_data(ticker, period=period)
    if raw is None:
        return None
    df = calculate_technical_indicators(raw)
    df = df.copy()
    df['forward_return'] = df['Adj_Close'].shift(-7) / df['Adj_Close'] - 1
    df = df.dropna(subset=['forward_return'])
    df['forward_return'] = df['forward_return'].clip(-0.30, 0.30)
    df['target'] = df['forward_return'].apply(discretize_return)
    if len(df) < 252:
        return None
    return df


def build_full_dataset(csv_path: str) -> Tuple[Tuple, Tuple]:
    """Fetch all HKEX equity tickers, combine, and time-split 80/20.

    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    tickers = load_tickers(csv_path)
    frames = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] Processing tickers...", flush=True)
        df = build_ticker_dataset(ticker)
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


def build_pipeline() -> Pipeline:
    """Build the unfitted sklearn Pipeline.

    RobustScaler is resistant to HK small-cap outliers.
    LGBMClassifier with class_weight='balanced' handles the dominant STABLE class.
    LightGBM produces well-calibrated probabilities natively (no CalibratedClassifierCV needed).
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.05,
            class_weight='balanced',
            min_child_samples=20,
            n_jobs=2,
            random_state=42,
            verbose=-1,
        ))
    ])


def find_ticker_csv(data_dir: str) -> str:
    """Return the path of the first CSV found in data_dir."""
    csvs = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    if not csvs:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")
    return os.path.join(data_dir, csvs[0])


def train_and_save(csv_path: str, output_path: str = 'stock_model.joblib') -> None:
    """Full training run: load data, fit pipeline, evaluate, save model."""
    print("=== AlphaPulse Model Training ===")
    print(f"Ticker source : {csv_path}")
    print(f"Output path   : {output_path}\n")

    (X_train, y_train), (X_test, y_test) = build_full_dataset(csv_path)

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
    label_names = ["DOWN>3%", "STABLE", "UP>3%"]
    print(classification_report(y_test, y_pred, target_names=label_names))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log-loss : {log_loss(y_test, y_proba):.4f}")

    model_data = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'description': (
            '3-class HKEX stock predictor. '
            'Classes: -1=DOWN>3%, 0=STABLE, 1=UP>3%. '
            'Horizon: 7 trading days.'
        ),
    }
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == '__main__':
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _csv = find_ticker_csv(_data_dir)
    train_and_save(_csv)
```

- [ ] **Step 2: Verify existing tests still import cleanly**

```bash
python -m pytest tests/test_train_model.py::TestDiscretizeReturn -v
```

Expected: All 9 tests PASS (discretize_return is unchanged).

- [ ] **Step 3: Commit**

```bash
git add train_model.py
git commit -m "feat: expand FEATURE_NAMES to 14, replace RandomForest with LightGBM"
```

---

## Task 4: Update tests/test_train_model.py for the new feature list

**Files:**
- Modify: `tests/test_train_model.py`

Two things break with the new feature list:
1. `_make_labeled_df` hardcodes the old 6-feature column list
2. `test_features_subset_only` asserts the old 6-feature list

- [ ] **Step 1: Run the broken test to confirm it fails**

```bash
python -m pytest tests/test_train_model.py::TestBuildFullDataset -v
```

Expected: `test_features_subset_only` FAILS with AssertionError (old feature list vs new).

- [ ] **Step 2: Update _make_labeled_df and test_features_subset_only**

In `tests/test_train_model.py`, find the `TestBuildFullDataset` class and replace `_make_labeled_df` and `test_features_subset_only`:

```python
_NEW_FEATURES = [
    'SMA_5_ratio', 'SMA_20_ratio', 'SMA_50_ratio',
    'RSI_14', 'RSI_7',
    'MACD', 'MACD_hist',
    'BB_pct_b',
    'Volume_ratio_20',
    'Volatility_20', 'Returns_1d', 'Returns_5d', 'Returns_10d', 'Returns_20d',
]
```

Replace `_make_labeled_df`:
```python
def _make_labeled_df(self, n_rows: int, start: str) -> pd.DataFrame:
    """Minimal labeled DataFrame as returned by build_ticker_dataset."""
    idx = pd.date_range(start, periods=n_rows, freq="B")
    data = {f: np.ones(n_rows) for f in _NEW_FEATURES}
    data['Adj_Close'] = np.ones(n_rows) * 100.0
    data['forward_return'] = np.zeros(n_rows)
    data['target'] = 0
    return pd.DataFrame(data, index=idx)
```

Replace `test_features_subset_only`:
```python
def test_features_subset_only(self, tmp_path):
    csv_content = (
        "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
        "0001.hk,00001,TICKER A,Equity,500,HK0001,\n"
    )
    csv_file = tmp_path / "hkex.csv"
    csv_file.write_text(csv_content)

    df_a = self._make_labeled_df(300, "2020-01-01")

    with patch("train_model.build_ticker_dataset", return_value=df_a):
        from train_model import build_full_dataset
        (X_train, y_train), (X_test, y_test) = build_full_dataset(str(csv_file))

    assert list(X_train.columns) == _NEW_FEATURES
    assert list(X_test.columns) == _NEW_FEATURES
```

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS. Watch especially for:
- `tests/test_train_model.py::TestBuildFullDataset::test_features_subset_only` → PASS
- `tests/test_get_price_data.py` → all PASS
- `tests/test_app.py` → all PASS (unaffected; uses mocked model)

- [ ] **Step 4: Commit**

```bash
git add tests/test_train_model.py
git commit -m "test: update feature list in train_model tests to 14-feature set"
```

---

## Task 5: Smoke-test the full pipeline end-to-end

**Files:** No code changes — this task is verification only.

- [ ] **Step 1: Run a quick smoke test with a single ticker**

```bash
python - <<'EOF'
import numpy as np
from get_price_data import fetch_latest_data, calculate_technical_indicators
from train_model import FEATURE_NAMES, build_pipeline

# Fetch one ticker
data = fetch_latest_data("0001.HK", period="5y")
if data is None:
    print("FAIL: Could not fetch 0001.HK")
else:
    df = calculate_technical_indicators(data)
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"FAIL: Missing columns: {missing}")
    else:
        X = df[FEATURE_NAMES].iloc[-1:].values
        pipeline = build_pipeline()
        print(f"PASS: {len(FEATURE_NAMES)} features present, pipeline builds OK")
        print(f"Sample feature vector shape: {X.shape}")
        print(f"Feature names: {FEATURE_NAMES}")
EOF
```

Expected output:
```
PASS: 14 features present, pipeline builds OK
Sample feature vector shape: (1, 14)
Feature names: ['SMA_5_ratio', 'SMA_20_ratio', 'SMA_50_ratio', ...]
```

- [ ] **Step 2: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS, 0 failures.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: feature expansion + LightGBM — 14 features, ready to retrain"
```

---

## After Completion: Retrain the model

Run training (takes 30–60 min on all HKEX tickers):

```bash
python train_model.py
```

What to watch for in the output:
- `UP>3% recall` should be >0.10 (was 0.01 with old model)
- `Accuracy` should be >0.44 (was 0.41)
- `Log-loss` should be <1.00 (was 1.08)

If UP recall is still near zero, consider further tuning: lower `min_child_samples`, increase `n_estimators` to 500, or add `subsample=0.8`.
