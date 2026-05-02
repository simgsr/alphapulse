# AlphaPulse Dual-Threshold Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain AlphaPulse with a 5-class calibrated model (DOWN >5%, DOWN 3–5%, STABLE, UP 3–5%, UP >5%), expose `confidence_up_3pct` and `confidence_up_5pct` in the API, and redesign the frontend as a mobile-first dark dashboard.

**Architecture:** `train_model.py` builds a calibrated `sklearn.Pipeline` (RobustScaler → CalibratedClassifierCV(RandomForest)) from ~2,700 HKEX tickers using 5 years of daily data, then saves it as `stock_model.joblib`. `app.py` derives the two confidence values from 5-class `predict_proba` output. The static frontend is rebuilt mobile-first with two animated confidence bars and a 5-segment donut chart.

**Tech Stack:** Python 3.8+, scikit-learn 1.8.0, yfinance, pandas, FastAPI, pytest, httpx, Chart.js (CDN)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `train_model.py` | **Create** | Training pipeline: load tickers, features, target labels, fit pipeline, evaluate, save |
| `tests/__init__.py` | **Create** | Empty — marks `tests/` as a package |
| `tests/test_train_model.py` | **Create** | Unit tests: `discretize_return`, `load_tickers`, `build_ticker_dataset` |
| `tests/test_app.py` | **Create** | Unit tests: `SIGNAL_MAP`, `build_prediction_response` |
| `app.py` | **Modify** | Add `SIGNAL_MAP`, extract `build_prediction_response()`, update endpoint to return new fields |
| `static/index.html` | **Modify** | Mobile layout: two confidence bar elements, error banner, updated stats section |
| `static/style.css` | **Rewrite** | Dark mobile-first theme, confidence bar styles, 5-class badge variants |
| `static/script.js` | **Modify** | Consume new API fields, 5-segment chart, animated confidence bars, inline error banner |
| `requirements.txt` | **Modify** | Add `pytest>=7.0.0`, `httpx>=0.24.0` |

---

## Task 1: Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add test dependencies to requirements.txt**

Open `requirements.txt` and append:
```
pytest>=7.0.0
httpx>=0.24.0
```

- [ ] **Step 2: Create the tests package**

Create `tests/__init__.py` as an empty file.

- [ ] **Step 3: Verify pytest runs**

```bash
cd /Users/simgsr/Documents/simgsr-ntu/capstone/ntu_yfinance_deployment
pip install pytest httpx --quiet
pytest tests/ -v
```

Expected output: `no tests ran` (0 tests collected, no errors).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "test: add pytest and httpx for test suite"
```

---

## Task 2: `discretize_return()` with TDD

**Files:**
- Create: `train_model.py` (skeleton + this function only)
- Create: `tests/test_train_model.py` (this test class only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_model.py`:

```python
import pytest
from train_model import discretize_return


class TestDiscreetizeReturn:
    def test_strong_up(self):
        assert discretize_return(0.06) == 2

    def test_mild_up(self):
        assert discretize_return(0.04) == 1

    def test_exact_5pct_is_mild_up(self):
        # +5% is the upper boundary of class 1 (UP 3-5%)
        assert discretize_return(0.05) == 1

    def test_exact_3pct_is_stable(self):
        # +3% is the upper boundary of class 0 (STABLE)
        assert discretize_return(0.03) == 0

    def test_stable_zero(self):
        assert discretize_return(0.0) == 0

    def test_stable_small_positive(self):
        assert discretize_return(0.02) == 0

    def test_stable_small_negative(self):
        assert discretize_return(-0.02) == 0

    def test_exact_neg_3pct_is_stable(self):
        # -3% is the lower boundary of class 0 (STABLE)
        assert discretize_return(-0.03) == 0

    def test_mild_down(self):
        assert discretize_return(-0.04) == -1

    def test_exact_neg_5pct_is_mild_down(self):
        # -5% is the upper boundary of class -1 (DOWN 3-5%)
        assert discretize_return(-0.05) == -1

    def test_strong_down(self):
        assert discretize_return(-0.06) == -2
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_train_model.py -v
```

Expected: `ImportError: cannot import name 'discretize_return' from 'train_model'`

- [ ] **Step 3: Create `train_model.py` with `discretize_return` only**

Create `train_model.py`:

```python
import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, log_loss
from get_price_data import fetch_latest_data, calculate_technical_indicators

FEATURE_NAMES = [
    'SMA_5_ratio', 'SMA_20_ratio', 'RSI_14',
    'Volatility_20', 'Returns_1d', 'Returns_5d'
]


def discretize_return(r: float) -> int:
    """Map a 7-day forward return to a 5-class label.

    Classes:
        2  — UP > 5%
        1  — UP 3–5%
        0  — STABLE (within ±3%)
       -1  — DOWN 3–5%
       -2  — DOWN > 5%
    """
    if r > 0.05:
        return 2
    elif r > 0.03:
        return 1
    elif r >= -0.03:
        return 0
    elif r >= -0.05:
        return -1
    else:
        return -2
```

- [ ] **Step 4: Run to verify PASS**

```bash
pytest tests/test_train_model.py::TestDiscreetizeReturn -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add discretize_return with 5-class label logic"
```

---

## Task 3: `load_tickers()` with TDD

**Files:**
- Modify: `train_model.py` (add `load_tickers`)
- Modify: `tests/test_train_model.py` (add `TestLoadTickers`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_model.py`:

```python
import os


class TestLoadTickers:
    def test_returns_equity_tickers_only(self, tmp_path):
        csv_content = (
            "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
            "0001.hk,00001,CKH HOLDINGS,Equity,500,KYG217651051,\n"
            "0002.hk,00002,CLP HOLDINGS,Equity,500,HK0002007356,\n"
            "BOND1.hk,B001,SOME BOND,Bond,1000,HK000BOND01,\n"
        )
        csv_file = tmp_path / "test_hkex.csv"
        csv_file.write_text(csv_content)

        tickers = load_tickers(str(csv_file))

        assert "0001.hk" in tickers
        assert "0002.hk" in tickers
        assert "BOND1.hk" not in tickers

    def test_returns_list_of_strings(self, tmp_path):
        csv_content = (
            "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
            "0001.hk,00001,CKH HOLDINGS,Equity,500,KYG217651051,\n"
        )
        csv_file = tmp_path / "test_hkex.csv"
        csv_file.write_text(csv_content)

        tickers = load_tickers(str(csv_file))

        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)
```

Update the import line at the top of `tests/test_train_model.py` to:
```python
from train_model import discretize_return, load_tickers
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_train_model.py::TestLoadTickers -v
```

Expected: `ImportError: cannot import name 'load_tickers'`

- [ ] **Step 3: Add `load_tickers` to `train_model.py`**

Append after the `discretize_return` function:

```python
def load_tickers(csv_path: str) -> list:
    """Return all Equity ticker symbols from the HKEX CSV."""
    df = pd.read_csv(csv_path)
    return df[df['Category'] == 'Equity']['Tickers'].tolist()
```

- [ ] **Step 4: Run to verify PASS**

```bash
pytest tests/test_train_model.py::TestLoadTickers -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add load_tickers filtering HKEX equity tickers"
```

---

## Task 4: `build_ticker_dataset()` with TDD

**Files:**
- Modify: `train_model.py` (add `build_ticker_dataset`)
- Modify: `tests/test_train_model.py` (add `TestBuildTickerDataset`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_model.py`:

```python
import pandas as pd
from unittest.mock import patch


def _make_synthetic_df(n_rows: int) -> pd.DataFrame:
    """Synthetic OHLCV-style DataFrame with Adj_Close and Adj_Volume."""
    np.random.seed(0)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    prices = 100.0 * np.cumprod(1 + np.random.normal(0, 0.01, n_rows))
    return pd.DataFrame(
        {"Adj_Close": prices, "Adj_Volume": np.full(n_rows, 500_000.0)},
        index=idx,
    )


class TestBuildTickerDataset:
    def test_returns_none_when_fetch_fails(self):
        with patch("train_model.fetch_latest_data", return_value=None):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result is None

    def test_returns_none_when_too_few_rows(self):
        short_df = _make_synthetic_df(50)
        with patch("train_model.fetch_latest_data", return_value=short_df), \
             patch("train_model.calculate_technical_indicators", return_value=short_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result is None

    def test_returns_dataframe_with_target_column(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result is not None
        assert "target" in result.columns
        assert "forward_return" in result.columns

    def test_target_values_are_valid_classes(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert set(result["target"].unique()).issubset({-2, -1, 0, 1, 2})

    def test_no_nan_in_result(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result["forward_return"].isna().sum() == 0

    def test_forward_return_capped_at_30pct(self):
        # inject a spike row
        long_df = _make_synthetic_df(400)
        long_df = long_df.copy()
        long_df.iloc[100, long_df.columns.get_loc("Adj_Close")] = 1e9  # extreme spike
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result["forward_return"].max() <= 0.30
        assert result["forward_return"].min() >= -0.30
```

Update the import at the top of `tests/test_train_model.py` to add `numpy`:
```python
import numpy as np
import os
import pandas as pd
from unittest.mock import patch
import pytest
from train_model import discretize_return, load_tickers
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_train_model.py::TestBuildTickerDataset -v
```

Expected: `ImportError: cannot import name 'build_ticker_dataset'`

- [ ] **Step 3: Add `build_ticker_dataset` to `train_model.py`**

Append after `load_tickers`:

```python
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
    # Forward return: (Close[t+7] - Close[t]) / Close[t]
    df['forward_return'] = df['Adj_Close'].shift(-7) / df['Adj_Close'] - 1
    df = df.dropna(subset=['forward_return'])
    df['forward_return'] = df['forward_return'].clip(-0.30, 0.30)
    df['target'] = df['forward_return'].apply(discretize_return)
    if len(df) < 252:
        return None
    return df
```

- [ ] **Step 4: Run to verify PASS**

```bash
pytest tests/test_train_model.py::TestBuildTickerDataset -v
```

Expected: `6 passed`

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest tests/test_train_model.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add build_ticker_dataset with forward return labeling"
```

---

## Task 5: `build_full_dataset()` with TDD

**Files:**
- Modify: `train_model.py` (add `build_full_dataset`)
- Modify: `tests/test_train_model.py` (add `TestBuildFullDataset`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_model.py`:

```python
class TestBuildFullDataset:
    def _make_labeled_df(self, n_rows: int, start: str) -> pd.DataFrame:
        """Minimal labeled DataFrame as returned by build_ticker_dataset."""
        idx = pd.date_range(start, periods=n_rows, freq="B")
        data = {f: np.ones(n_rows) for f in [
            'SMA_5_ratio', 'SMA_20_ratio', 'RSI_14',
            'Volatility_20', 'Returns_1d', 'Returns_5d'
        ]}
        data['Adj_Close'] = np.ones(n_rows) * 100.0
        data['forward_return'] = np.zeros(n_rows)
        data['target'] = 0
        return pd.DataFrame(data, index=idx)

    def test_80_20_split_sizes(self, tmp_path):
        csv_content = (
            "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
            "0001.hk,00001,TICKER A,Equity,500,HK0001,\n"
            "0002.hk,00002,TICKER B,Equity,500,HK0002,\n"
        )
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")
        df_b = self._make_labeled_df(300, "2020-01-01")

        def mock_build(ticker, period='5y'):
            return df_a if ticker == '0001.hk' else df_b

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            (X_train, y_train), (X_test, y_test) = build_full_dataset(str(csv_file))

        total = len(X_train) + len(X_test)
        assert abs(len(X_train) / total - 0.8) < 0.01

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

        expected_features = [
            'SMA_5_ratio', 'SMA_20_ratio', 'RSI_14',
            'Volatility_20', 'Returns_1d', 'Returns_5d'
        ]
        assert list(X_train.columns) == expected_features
        assert list(X_test.columns) == expected_features

    def test_skips_none_tickers(self, tmp_path):
        csv_content = (
            "Tickers,Stock Code,Name of Securities,Category,Board Lot,ISIN,RMB Counter\n"
            "0001.hk,00001,TICKER A,Equity,500,HK0001,\n"
            "0002.hk,00002,TICKER BAD,Equity,500,HK0002,\n"
        )
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")

        def mock_build(ticker, period='5y'):
            return df_a if ticker == '0001.hk' else None

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            (X_train, y_train), _ = build_full_dataset(str(csv_file))

        assert len(X_train) > 0
```

Update the import at the top to add `build_full_dataset` later (leave import as-is; pytest will trigger the ImportError naturally).

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_train_model.py::TestBuildFullDataset -v
```

Expected: `ImportError: cannot import name 'build_full_dataset'`

- [ ] **Step 3: Add `build_full_dataset` to `train_model.py`**

Append after `build_ticker_dataset`:

```python
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
```

- [ ] **Step 4: Run to verify PASS**

```bash
pytest tests/test_train_model.py::TestBuildFullDataset -v
```

Expected: `3 passed`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_train_model.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add build_full_dataset with 80/20 time-based split"
```

---

## Task 6: `build_pipeline()` + `train_and_save()` + Script Entrypoint

**Files:**
- Modify: `train_model.py` (add remaining functions + `__main__` block)

> No unit test for `train_and_save` — fitting on 2M+ rows takes 30–60 min and is validated by the evaluation output printed to stdout. The integration is verified by running the script.

- [ ] **Step 1: Add `build_pipeline` and `train_and_save` to `train_model.py`**

Append to `train_model.py`:

```python
def build_pipeline() -> Pipeline:
    """Build the unfitted sklearn Pipeline.

    RobustScaler is resistant to HK small-cap outliers.
    CalibratedClassifierCV with isotonic regression corrects RandomForest's
    tendency to produce overconfident probability estimates.
    TimeSeriesSplit ensures the calibration CV folds respect temporal order.
    class_weight='balanced' compensates for the dominant STABLE class.
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf', CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
            ),
            method='isotonic',
            cv=TimeSeriesSplit(n_splits=5),
        ))
    ])


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
    print("\nFitting pipeline — this may take 30–60 minutes...")
    pipeline.fit(X_train, y_train)
    print("Fitting complete.")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print("\n=== Held-out Test Set Evaluation ===")
    label_names = ["DOWN>5%", "DOWN 3-5%", "STABLE", "UP 3-5%", "UP>5%"]
    print(classification_report(y_test, y_pred, target_names=label_names))
    print(f"Log-loss: {log_loss(y_test, y_proba):.4f}")

    model_data = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'description': (
            '5-class HKEX stock predictor. '
            'Classes: -2=DOWN>5%, -1=DOWN3-5%, 0=STABLE, 1=UP3-5%, 2=UP>5%. '
            'Horizon: 7 trading days.'
        ),
    }
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == '__main__':
    _csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'hkex.csv')
    train_and_save(_csv)
```

- [ ] **Step 2: Verify the script parses without error**

```bash
python3 -c "import train_model; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
pytest tests/test_train_model.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add train_model.py
git commit -m "feat: add build_pipeline and train_and_save with evaluation output"
```

> **Note:** Run `python3 train_model.py` after Task 7 (once app.py is updated) so the new `stock_model.joblib` is compatible with the updated endpoint. Training takes 30–60 min.

---

## Task 7: Update `app.py` with TDD

**Files:**
- Modify: `app.py`
- Create: `tests/test_app.py`

The key change: extract prediction logic into a pure `build_prediction_response()` function so it can be tested without loading the model file.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_mock_model(prediction: int = 2):
    """Return a mock Pipeline whose predict/predict_proba mimic 5-class output."""
    m = MagicMock()
    m.predict.return_value = np.array([prediction])
    m.predict_proba.return_value = np.array([[0.05, 0.10, 0.23, 0.31, 0.31]])
    m.classes_ = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return m


def _make_mock_data():
    import pandas as pd
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    return pd.DataFrame(
        {"Adj_Close": [57.35] * 30, "Adj_Volume": [500_000.0] * 30},
        index=idx,
    )


class TestBuildPredictionResponse:
    def test_confidence_up_3pct_is_sum_of_class_1_and_2(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=2)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        # proba: [-2]=0.05, [-1]=0.10, [0]=0.23, [1]=0.31, [2]=0.31
        # confidence_up_3pct = P(1) + P(2) = 0.31 + 0.31 = 0.62
        assert abs(result["confidence_up_3pct"] - 0.62) < 1e-4

    def test_confidence_up_5pct_is_class_2_only(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=2)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert abs(result["confidence_up_5pct"] - 0.31) < 1e-4

    def test_confidence_5pct_never_exceeds_3pct(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert result["confidence_up_5pct"] <= result["confidence_up_3pct"]

    def test_signal_map_all_classes(self):
        from app import SIGNAL_MAP
        assert SIGNAL_MAP[2] == "UP > 5%"
        assert SIGNAL_MAP[1] == "UP 3-5%"
        assert SIGNAL_MAP[0] == "STABLE"
        assert SIGNAL_MAP[-1] == "DOWN 3-5%"
        assert SIGNAL_MAP[-2] == "DOWN > 5%"

    def test_response_contains_required_fields(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        for key in ("ticker", "prediction", "signal",
                    "confidence_up_3pct", "confidence_up_5pct",
                    "probabilities", "current_price", "last_updated"):
            assert key in result, f"Missing key: {key}"

    def test_probabilities_keyed_by_string_int(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert set(result["probabilities"].keys()) == {"-2", "-1", "0", "1", "2"}
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_app.py -v
```

Expected: `ImportError: cannot import name 'build_prediction_response' from 'app'`

- [ ] **Step 3: Update `app.py`**

Replace the entire content of `app.py` with:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os
import numpy as np
from get_price_data import fetch_latest_data, calculate_technical_indicators

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SIGNAL_MAP = {
    2: "UP > 5%",
    1: "UP 3-5%",
    0: "STABLE",
    -1: "DOWN 3-5%",
    -2: "DOWN > 5%",
}

MODEL_PATH = 'stock_model.joblib'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stock_model.joblib')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_names = model_data['features']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []


def build_prediction_response(ticker: str, mdl, features_array: np.ndarray, raw_data) -> dict:
    """Pure function: run inference and build the API response dict."""
    prediction = int(mdl.predict(features_array)[0])
    probabilities = mdl.predict_proba(features_array)[0].tolist()
    classes = mdl.classes_.tolist()
    prob_dict = {str(int(c)): round(p, 4) for c, p in zip(classes, probabilities)}

    confidence_up_3pct = round(prob_dict.get('1', 0.0) + prob_dict.get('2', 0.0), 4)
    confidence_up_5pct = round(prob_dict.get('2', 0.0), 4)

    return {
        "ticker": ticker,
        "prediction": prediction,
        "signal": SIGNAL_MAP.get(prediction, "UNKNOWN"),
        "confidence_up_3pct": confidence_up_3pct,
        "confidence_up_5pct": confidence_up_5pct,
        "probabilities": prob_dict,
        "current_price": float(raw_data['Adj_Close'].iloc[-1]),
        "last_updated": str(raw_data.index[-1].date()),
    }


@app.get("/predict/{ticker}")
async def predict(ticker: str):
    ticker = ticker.upper()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    data = fetch_latest_data(ticker)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    processed = calculate_technical_indicators(data)
    if processed.empty:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")

    latest_features = processed[feature_names].iloc[-1:].values
    return build_prediction_response(ticker, model, latest_features, data)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

- [ ] **Step 4: Run to verify PASS**

```bash
pytest tests/test_app.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add SIGNAL_MAP and build_prediction_response for 5-class output"
```

---

## Task 8: Update `static/index.html`

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Replace the content of `static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <title>AlphaPulse | HK Stock AI</title>
    <link rel="stylesheet" href="style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="app">

        <header class="header">
            <span class="header__logo">Alpha<span class="header__logo--accent">Pulse</span></span>
            <span class="header__sub">HK Stock Forecast AI</span>
        </header>

        <section class="search-section">
            <div class="search-row">
                <input
                    type="text"
                    id="tickerInput"
                    class="search-input"
                    placeholder="e.g. 0001.HK or AAPL"
                    autocomplete="off"
                    autocapitalize="characters"
                >
                <button id="predictBtn" class="search-btn" aria-label="Analyze">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="2.5"
                         stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    <span>Analyze</span>
                </button>
            </div>
            <div id="errorBanner" class="error-banner hidden">
                <span id="errorText">An error occurred.</span>
            </div>
        </section>

        <div id="loader" class="loader hidden">
            <div class="spinner"></div>
            <p class="loader__text">Analyzing market data…</p>
        </div>

        <section id="resultContainer" class="result hidden">

            <div class="result__header">
                <div class="result__ticker-block">
                    <h2 id="tickerDisplay" class="result__ticker">---</h2>
                    <p id="priceDisplay" class="result__price">$0.00</p>
                </div>
                <div id="signalBadge" class="badge">---</div>
            </div>

            <div class="confidence-stack">
                <div class="confidence-card">
                    <div class="confidence-card__header">
                        <span class="confidence-card__label">Confidence price UP &gt;3% in 7 days</span>
                        <span id="conf3Pct" class="confidence-card__value">0%</span>
                    </div>
                    <div class="confidence-bar">
                        <div id="conf3Bar" class="confidence-bar__fill confidence-bar__fill--mild" style="width:0%"></div>
                    </div>
                </div>

                <div class="confidence-card">
                    <div class="confidence-card__header">
                        <span class="confidence-card__label">Confidence price UP &gt;5% in 7 days</span>
                        <span id="conf5Pct" class="confidence-card__value">0%</span>
                    </div>
                    <div class="confidence-bar">
                        <div id="conf5Bar" class="confidence-bar__fill confidence-bar__fill--strong" style="width:0%"></div>
                    </div>
                </div>
            </div>

            <div class="chart-section">
                <div class="chart-wrap">
                    <canvas id="probChart"></canvas>
                </div>
                <div class="chart-legend" id="chartLegend"></div>
            </div>

            <p id="lastUpdated" class="result__updated">Updated: —</p>

            <p class="disclaimer">Not financial advice. Based on historical patterns only.</p>

        </section>

    </div>
    <script src="script.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify the file saves correctly**

```bash
grep -c "confidence-card" static/index.html
```

Expected: `4` (two card divs × 2 occurrences of class each = at least 4)

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign index.html with mobile-first layout and dual confidence bars"
```

---

## Task 9: Rewrite `static/style.css`

**Files:**
- Modify: `static/style.css`

- [ ] **Step 1: Replace the entire content of `static/style.css`**

```css
:root {
    --bg:            #0a0f1e;
    --card:          #111c2d;
    --card-border:   rgba(255,255,255,0.07);
    --accent:        #00d4ff;
    --text:          #f0f4ff;
    --text-muted:    rgba(240,244,255,0.45);
    --up-strong:     #00ff88;
    --up-mild:       #4ade80;
    --neutral:       #fbbf24;
    --down-mild:     #f97316;
    --down-strong:   #ef4444;
    --radius-card:   18px;
    --radius-btn:    14px;
    --radius-bar:    6px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
    background: var(--bg);
    color: var(--text);
    min-height: 100dvh;
    background-image:
        radial-gradient(ellipse at 15% 10%, rgba(0,212,255,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 90%, rgba(0,255,136,0.06) 0%, transparent 50%);
}

.app {
    width: 100%;
    max-width: 480px;
    margin: 0 auto;
    padding: 24px 16px 40px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

/* ── Header ── */
.header {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding-top: 8px;
}
.header__logo {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header__logo--accent { color: var(--accent); }
.header__sub {
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Search ── */
.search-section { display: flex; flex-direction: column; gap: 10px; }
.search-row { display: flex; gap: 10px; }

.search-input {
    flex: 1;
    min-width: 0;
    height: 52px;
    padding: 0 16px;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-btn);
    color: var(--text);
    font-size: 16px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
}
.search-input::placeholder { color: var(--text-muted); }
.search-input:focus { border-color: var(--accent); }

.search-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 52px;
    padding: 0 20px;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: var(--radius-btn);
    font-size: 15px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    white-space: nowrap;
    transition: opacity 0.2s, transform 0.1s;
    -webkit-tap-highlight-color: transparent;
}
.search-btn:active { opacity: 0.85; transform: scale(0.97); }

/* ── Error banner ── */
.error-banner {
    padding: 12px 16px;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    color: var(--down-strong);
    font-size: 14px;
}
.hidden { display: none !important; }

/* ── Loader ── */
.loader { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 32px 0; }
.spinner {
    width: 40px; height: 40px;
    border: 3px solid rgba(255,255,255,0.08);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}
.loader__text { color: var(--text-muted); font-size: 14px; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Result card ── */
.result {
    display: flex;
    flex-direction: column;
    gap: 16px;
    animation: fadeUp 0.35s ease-out both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Result header ── */
.result__header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-card);
    padding: 20px;
}
.result__ticker {
    font-size: 26px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.result__price {
    font-size: 22px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Signal badge ── */
.badge {
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
    white-space: nowrap;
    border: 1px solid currentColor;
}
.badge--up2   { color: var(--up-strong);   background: rgba(0,255,136,0.12); }
.badge--up1   { color: var(--up-mild);     background: rgba(74,222,128,0.12); }
.badge--neutral { color: var(--neutral);   background: rgba(251,191,36,0.12); }
.badge--down1 { color: var(--down-mild);   background: rgba(249,115,22,0.12); }
.badge--down2 { color: var(--down-strong); background: rgba(239,68,68,0.12); }

/* ── Confidence cards ── */
.confidence-stack { display: flex; flex-direction: column; gap: 12px; }

.confidence-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-card);
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.confidence-card__header { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.confidence-card__label { font-size: 13px; color: var(--text-muted); }
.confidence-card__value { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; flex-shrink: 0; }

.confidence-bar {
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: var(--radius-bar);
    overflow: hidden;
}
.confidence-bar__fill {
    height: 100%;
    border-radius: var(--radius-bar);
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
    width: 0%;
}
.confidence-bar__fill--mild   { background: linear-gradient(90deg, #22c55e, var(--up-mild)); }
.confidence-bar__fill--strong { background: linear-gradient(90deg, #16a34a, var(--up-strong)); }

/* ── Chart section ── */
.chart-section {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-card);
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
}
.chart-wrap { width: 180px; height: 180px; position: relative; }

.chart-legend {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px 14px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: var(--text-muted);
}
.legend-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Footer ── */
.result__updated { font-size: 12px; color: var(--text-muted); text-align: right; }
.disclaimer { font-size: 11px; color: var(--text-muted); text-align: center; opacity: 0.6; }
```

- [ ] **Step 2: Verify the file was written**

```bash
grep "confidence-bar__fill" static/style.css | wc -l
```

Expected: at least `3`

- [ ] **Step 3: Commit**

```bash
git add static/style.css
git commit -m "feat: rewrite style.css as mobile-first dark theme with confidence bars"
```

---

## Task 10: Update `static/script.js`

**Files:**
- Modify: `static/script.js`

- [ ] **Step 1: Replace the entire content of `static/script.js`**

```js
const tickerInput   = document.getElementById('tickerInput');
const predictBtn    = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultContainer');
const loader        = document.getElementById('loader');
const errorBanner   = document.getElementById('errorBanner');
const errorText     = document.getElementById('errorText');

const tickerDisplay = document.getElementById('tickerDisplay');
const priceDisplay  = document.getElementById('priceDisplay');
const signalBadge   = document.getElementById('signalBadge');
const conf3Pct      = document.getElementById('conf3Pct');
const conf5Pct      = document.getElementById('conf5Pct');
const conf3Bar      = document.getElementById('conf3Bar');
const conf5Bar      = document.getElementById('conf5Bar');
const lastUpdated   = document.getElementById('lastUpdated');
const chartLegend   = document.getElementById('chartLegend');

let probChart = null;

// JS object keys are always strings; String(data.prediction) matches all keys here
const SIGNAL_CLASSES = {
    '2':  'badge--up2',
    '1':  'badge--up1',
    '0':  'badge--neutral',
    '-1': 'badge--down1',
    '-2': 'badge--down2',
};

const CHART_CONFIG = [
    { key: '-2', label: 'DOWN >5%',  color: '#ef4444' },
    { key: '-1', label: 'DOWN 3-5%', color: '#f97316' },
    { key: '0',  label: 'STABLE',    color: '#fbbf24' },
    { key: '1',  label: 'UP 3-5%',   color: '#4ade80' },
    { key: '2',  label: 'UP >5%',    color: '#00ff88' },
];

function showError(msg) {
    errorText.textContent = msg;
    errorBanner.classList.remove('hidden');
}

function clearError() {
    errorBanner.classList.add('hidden');
}

async function performAnalysis() {
    const ticker = tickerInput.value.trim();
    if (!ticker) return;

    clearError();
    resultSection.classList.add('hidden');
    loader.classList.remove('hidden');
    predictBtn.disabled = true;

    try {
        const res = await fetch(`/predict/${encodeURIComponent(ticker)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
            throw new Error(err.detail || 'Analysis failed');
        }
        const data = await res.json();
        updateUI(data);
    } catch (e) {
        showError(e.message);
    } finally {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }
}

function updateUI(data) {
    // Header
    tickerDisplay.textContent = data.ticker;
    priceDisplay.textContent  = '$' + Number(data.current_price).toLocaleString(undefined, {
        minimumFractionDigits: 2, maximumFractionDigits: 2
    });

    // Signal badge
    signalBadge.textContent  = data.signal;
    signalBadge.className    = 'badge';
    const predKey = String(data.prediction);
    if (SIGNAL_CLASSES[predKey]) signalBadge.classList.add(SIGNAL_CLASSES[predKey]);

    // Confidence bars (trigger CSS transition by setting width after next frame)
    const pct3 = Math.round(data.confidence_up_3pct * 100);
    const pct5 = Math.round(data.confidence_up_5pct * 100);
    conf3Pct.textContent = pct3 + '%';
    conf5Pct.textContent = pct5 + '%';
    requestAnimationFrame(() => {
        conf3Bar.style.width = pct3 + '%';
        conf5Bar.style.width = pct5 + '%';
    });

    // Chart
    renderChart(data.probabilities);

    // Footer
    lastUpdated.textContent = 'Updated: ' + data.last_updated;

    resultSection.classList.remove('hidden');
}

function renderChart(probs) {
    const ctx = document.getElementById('probChart').getContext('2d');
    if (probChart) probChart.destroy();

    const values = CHART_CONFIG.map(c => probs[c.key] ?? 0);
    const colors = CHART_CONFIG.map(c => c.color);

    probChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: CHART_CONFIG.map(c => c.label),
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 6,
            }]
        },
        options: {
            responsive: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${(ctx.raw * 100).toFixed(1)}%`
                    }
                }
            },
            cutout: '70%',
            onClick: (evt, elements) => {
                if (elements.length) {
                    const idx = elements[0].index;
                    // brief visual feedback only — no action needed
                }
            }
        }
    });

    // Build legend
    chartLegend.innerHTML = CHART_CONFIG.map((c, i) => `
        <span class="legend-item">
            <span class="legend-dot" style="background:${c.color}"></span>
            ${c.label} <strong>${(values[i] * 100).toFixed(1)}%</strong>
        </span>
    `).join('');
}

predictBtn.addEventListener('click', performAnalysis);
tickerInput.addEventListener('keydown', e => { if (e.key === 'Enter') performAnalysis(); });
```

- [ ] **Step 2: Verify no syntax errors**

```bash
node --input-type=module < static/script.js 2>&1 | head -5
```

Expected: either no output or only warnings about missing DOM globals (not syntax errors). If `node` is not available, skip to Step 3.

- [ ] **Step 3: Commit**

```bash
git add static/script.js
git commit -m "feat: update script.js for 5-class chart and dual confidence bars"
```

---

## Task 11: Train the New Model

> This task requires network access (yfinance) and takes 30–60 minutes. Run manually.

- [ ] **Step 1: Run the training script**

```bash
cd /Users/simgsr/Documents/simgsr-ntu/capstone/ntu_yfinance_deployment
python3 train_model.py
```

Watch for:
- Progress logs every 100 tickers
- Class distribution print (class 0 should dominate — expected)
- Classification report after fitting
- Log-loss printed to confirm calibration quality
- `Model saved to: stock_model.joblib`

- [ ] **Step 2: Verify model file updated**

```bash
python3 -c "
import joblib, datetime, os
m = joblib.load('stock_model.joblib')
mtime = datetime.datetime.fromtimestamp(os.path.getmtime('stock_model.joblib'))
print('Classes :', m['model'].classes_)
print('Features:', m['features'])
print('Saved at:', mtime)
"
```

Expected output includes `Classes : [-2. -1.  0.  1.  2.]`

- [ ] **Step 3: Commit the retrained model**

```bash
git add stock_model.joblib
git commit -m "feat: retrain model with 5-class discretized targets from HKEX universe"
```

---

## Task 12: End-to-End Smoke Test

- [ ] **Step 1: Start the server**

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

- [ ] **Step 2: Test a known HKEX ticker**

In a second terminal:

```bash
curl -s http://localhost:7860/predict/0001.hk | python3 -m json.tool
```

Verify the response contains:
```json
{
  "ticker": "0001.HK",
  "prediction": <int>,
  "signal": "<one of UP > 5% / UP 3-5% / STABLE / DOWN 3-5% / DOWN > 5%>",
  "confidence_up_3pct": <float 0–1>,
  "confidence_up_5pct": <float 0–1>,
  "probabilities": {"-2": ..., "-1": ..., "0": ..., "1": ..., "2": ...},
  "current_price": <float>,
  "last_updated": "<date>"
}
```

- [ ] **Step 3: Test case-insensitive ticker input**

```bash
curl -s http://localhost:7860/predict/0001.hk | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['ticker']=='0001.HK', d['ticker']"
echo "Case normalization: OK"
```

- [ ] **Step 4: Open the mobile UI in a browser**

Open `http://localhost:7860` in a browser (or use DevTools device emulation at 390px width).

Verify:
- Dark background loads
- Input and Analyze button are full-width and 52px tall
- After entering `0001.HK` and clicking Analyze, confidence bars animate to correct percentages
- Signal badge is color-coded
- Donut chart shows 5 segments with legend
- No horizontal scroll at 390px viewport width

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete AlphaPulse dual-threshold prediction upgrade"
```
