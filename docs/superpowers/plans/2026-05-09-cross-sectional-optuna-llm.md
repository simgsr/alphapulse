# Cross-Sectional Rank Features + Optuna + LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-sectional percentile rank features (21→42), Optuna hyperparameter search, and an LLM interpretation layer to AlphaPulse.

**Architecture:** Cross-sectional ranks are computed at training time via `(date, exchange)` grouping and stored as a quantile table in the model `.joblib` for O(log n) inference-time lookup. Optuna tunes on a 30% subsample, then the final model trains on the full dataset. LLM logic lives in a new `llm_utils.py`; `app.py` adds two buttons wired to `gr.State`.

**Tech Stack:** LightGBM, scikit-learn, Optuna, LangChain (Ollama/Groq/Gemini), Gradio, pytest

---

## File Map

| File | Action | What changes |
|---|---|---|
| `train_model.py` | Modify | `_ticker_exchange`, `RANK_FEATURE_NAMES`, `ALL_FEATURE_NAMES`, `_compute_cross_sectional_ranks`, `_build_quantile_table`, `_optuna_best_params`; update `build_ticker_dataset`, `build_full_dataset`, `train_and_save` |
| `app.py` | Modify | `_load_model` (3-tuple), `add_rank_features`, `INDICATOR_KEYS`, `_build_prediction`, `_build_dual_prediction`, `analyze`, `scan_watchlist`, `_explain`, `_summarize`; add UI buttons + states |
| `llm_utils.py` | Create | `get_llm`, `explain_signal`, `summarize_scan` |
| `tests/test_train_model.py` | Modify | Add exchange/rank/quantile/optuna tests; fix existing unpackings |
| `tests/test_app.py` | Modify | Update `_load_model` test; add `add_rank_features` tests |
| `tests/test_llm_utils.py` | Create | Tests for all three `llm_utils` functions |
| `requirements.txt` | Modify | Add `optuna`, `langchain-ollama`, `langchain-groq`, `langchain-google-genai` |

---

## Task 1: Exchange tagging in `train_model.py`

**Files:**
- Modify: `train_model.py`
- Modify: `tests/test_train_model.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train_model.py` after the existing imports:

```python
class TestTickerExchange:
    def test_hk_suffix_returns_hk(self):
        from train_model import _ticker_exchange
        assert _ticker_exchange("0001.HK") == "HK"

    def test_si_suffix_returns_sgx(self):
        from train_model import _ticker_exchange
        assert _ticker_exchange("D05.SI") == "SGX"

    def test_unknown_suffix_returns_all(self):
        from train_model import _ticker_exchange
        assert _ticker_exchange("AAPL") == "ALL"

    def test_case_insensitive(self):
        from train_model import _ticker_exchange
        assert _ticker_exchange("0001.hk") == "HK"


class TestBuildTickerDatasetExchange:
    def test_hk_ticker_adds_exchange_column(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.HK")
        assert "exchange" in result.columns
        assert (result["exchange"] == "HK").all()

    def test_si_ticker_exchange_is_sgx(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("D05.SI")
        assert (result["exchange"] == "SGX").all()

    def test_unknown_ticker_exchange_is_all(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("AAPL")
        assert (result["exchange"] == "ALL").all()
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/simgsr/Documents/python_project/yf_price_prediction && \
source venv/bin/activate && \
pytest tests/test_train_model.py::TestTickerExchange tests/test_train_model.py::TestBuildTickerDatasetExchange -v
```

Expected: `ImportError` or `AttributeError` — `_ticker_exchange` not defined.

- [ ] **Step 3: Implement in `train_model.py`**

After the `FEATURE_NAMES` list, add:

```python
RANK_FEATURE_NAMES = [f'{f}_rank' for f in FEATURE_NAMES]
ALL_FEATURE_NAMES = FEATURE_NAMES + RANK_FEATURE_NAMES


def _ticker_exchange(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith('.HK'):
        return 'HK'
    if t.endswith('.SI'):
        return 'SGX'
    return 'ALL'
```

In `build_ticker_dataset`, after `df = df.copy()` (line 57), add:

```python
    df['exchange'] = _ticker_exchange(ticker)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_train_model.py::TestTickerExchange tests/test_train_model.py::TestBuildTickerDatasetExchange -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add exchange tagging and RANK_FEATURE_NAMES constants"
```

---

## Task 2: Cross-sectional rank helpers in `train_model.py`

**Files:**
- Modify: `train_model.py`
- Modify: `tests/test_train_model.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train_model.py`:

```python
class TestComputeCrossSectionalRanks:
    def _make_combined(self):
        from train_model import FEATURE_NAMES
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        data = {f: [1.0, 2.0, 3.0] for f in FEATURE_NAMES}
        data['exchange'] = 'HK'
        data['target'] = 0
        return pd.DataFrame(data, index=idx)

    def test_adds_rank_columns_for_all_features(self):
        from train_model import _compute_cross_sectional_ranks, FEATURE_NAMES
        df = self._make_combined()
        result = _compute_cross_sectional_ranks(df)
        for feat in FEATURE_NAMES:
            assert f'{feat}_rank' in result.columns

    def test_rank_values_between_zero_and_one(self):
        from train_model import _compute_cross_sectional_ranks, FEATURE_NAMES
        df = self._make_combined()
        result = _compute_cross_sectional_ranks(df)
        for feat in FEATURE_NAMES:
            vals = result[f'{feat}_rank'].dropna()
            assert (vals >= 0.0).all() and (vals <= 1.0).all()

    def test_original_columns_preserved(self):
        from train_model import _compute_cross_sectional_ranks, FEATURE_NAMES
        df = self._make_combined()
        result = _compute_cross_sectional_ranks(df)
        for feat in FEATURE_NAMES:
            assert feat in result.columns


class TestBuildQuantileTable:
    def _make_df(self):
        from train_model import FEATURE_NAMES
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        data = {f: [float(i) for i in range(5)] for f in FEATURE_NAMES}
        data['exchange'] = ['HK', 'HK', 'SGX', 'HK', 'SGX']
        return pd.DataFrame(data, index=idx)

    def test_returns_hk_sgx_all_keys(self):
        from train_model import _build_quantile_table
        table = _build_quantile_table(self._make_df())
        assert 'HK' in table and 'SGX' in table and 'ALL' in table

    def test_arrays_are_sorted(self):
        from train_model import _build_quantile_table, FEATURE_NAMES
        table = _build_quantile_table(self._make_df())
        arr = table['HK'][FEATURE_NAMES[0]]
        assert list(arr) == sorted(arr)

    def test_all_key_contains_all_rows(self):
        from train_model import _build_quantile_table, FEATURE_NAMES
        table = _build_quantile_table(self._make_df())
        assert len(table['ALL'][FEATURE_NAMES[0]]) == 5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_train_model.py::TestComputeCrossSectionalRanks tests/test_train_model.py::TestBuildQuantileTable -v
```

Expected: `ImportError` — functions not defined.

- [ ] **Step 3: Implement in `train_model.py`**

Add after `_ticker_exchange`:

```python
def _compute_cross_sectional_ranks(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    for feat in FEATURE_NAMES:
        combined[f'{feat}_rank'] = combined.groupby(
            [combined.index, 'exchange']
        )[feat].rank(pct=True)
    return combined


def _build_quantile_table(df: pd.DataFrame) -> dict:
    table = {}
    for exch in ('HK', 'SGX', 'ALL'):
        subset = df if exch == 'ALL' else df[df['exchange'] == exch]
        table[exch] = {
            feat: np.sort(subset[feat].dropna().values)
            for feat in FEATURE_NAMES
        }
    return table
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_train_model.py::TestComputeCrossSectionalRanks tests/test_train_model.py::TestBuildQuantileTable -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add _compute_cross_sectional_ranks and _build_quantile_table"
```

---

## Task 3: Update `build_full_dataset` + fix existing tests

**Files:**
- Modify: `train_model.py`
- Modify: `tests/test_train_model.py`

- [ ] **Step 1: Update `_make_labeled_df` helpers in tests to add `exchange` column**

In `tests/test_train_model.py`, both `TestBuildFullDataset._make_labeled_df` and `TestBuildFullDatasetSGX._make_labeled_df` need an `exchange` column. Update both:

```python
# In TestBuildFullDataset._make_labeled_df:
def _make_labeled_df(self, n_rows: int, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_rows, freq="B")
    data = {f: np.ones(n_rows) for f in _NEW_FEATURES}
    data['Adj_Close'] = np.ones(n_rows) * 100.0
    data['forward_return'] = np.zeros(n_rows)
    data['target'] = 0
    data['exchange'] = 'HK'          # ADD THIS LINE
    return pd.DataFrame(data, index=idx)
```

Apply the same `data['exchange'] = 'HK'` addition to `TestBuildFullDatasetSGX._make_labeled_df`.

- [ ] **Step 2: Fix all existing unpackings from 2-tuple to 3-tuple**

In `tests/test_train_model.py`, find every line that unpacks `build_full_dataset` and add `_, ` for the quantile_table:

```python
# test_80_20_split_sizes:
(X_train, y_train), (X_test, y_test), _ = build_full_dataset(str(csv_file))

# test_features_subset_only (also update the assertion):
(X_train, y_train), (X_test, y_test), _ = build_full_dataset(str(csv_file))
# Change assertion from:
#   assert list(X_train.columns) == expected_features
# to:
from train_model import ALL_FEATURE_NAMES
assert list(X_train.columns) == ALL_FEATURE_NAMES

# test_skips_none_tickers:
(X_train, y_train), _, _ = build_full_dataset(str(csv_file))

# test_no_extra_csv_runs_normally:
(X_train, _), _, _ = build_full_dataset(str(hk_csv))
```

Also update the import at the top to add `ALL_FEATURE_NAMES`:
```python
from train_model import binarize_return, load_tickers, FEATURE_NAMES as _NEW_FEATURES, ALL_FEATURE_NAMES
```

- [ ] **Step 3: Add new test for quantile table return**

Add to `TestBuildFullDataset`:

```python
def test_returns_quantile_table_as_third_element(self, tmp_path):
    csv_content = "0001.hk\n"
    csv_file = tmp_path / "hkex.csv"
    csv_file.write_text(csv_content)
    df_a = self._make_labeled_df(300, "2020-01-01")

    with patch("train_model.build_ticker_dataset", return_value=df_a):
        from train_model import build_full_dataset
        _, _, quantile_table = build_full_dataset(str(csv_file))

    assert isinstance(quantile_table, dict)
    assert 'HK' in quantile_table
    assert 'ALL' in quantile_table
```

- [ ] **Step 4: Run existing tests to see current failures**

```bash
pytest tests/test_train_model.py::TestBuildFullDataset tests/test_train_model.py::TestBuildFullDatasetSGX -v
```

Expected: multiple failures due to 2-tuple unpacking and missing `exchange` column.

- [ ] **Step 5: Update `build_full_dataset` in `train_model.py`**

Replace the body of `build_full_dataset` from the `combined = pd.concat(...)` line onwards:

```python
    combined = pd.concat(frames).sort_index()

    print("\nClass distribution (full dataset):")
    print(combined['target'].value_counts().sort_index().to_string())

    combined = _compute_cross_sectional_ranks(combined)

    X = combined[ALL_FEATURE_NAMES]
    y = combined['target']
    split = int(len(combined) * 0.8)

    quantile_table = _build_quantile_table(combined.iloc[:split])

    return (X.iloc[:split], y.iloc[:split]), (X.iloc[split:], y.iloc[split:]), quantile_table
```

Also update the return type hint in the signature:
```python
def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    extra_csv_paths: list = None,
) -> Tuple[Tuple, Tuple, dict]:
```

- [ ] **Step 6: Update `train_and_save` to unpack 3 values and save quantile table**

In `train_and_save`, change the unpacking line:
```python
    (X_train, y_train), (X_test, y_test), quantile_table = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        extra_csv_paths=extra_csv_paths,
    )
```

Update `model_data` dict to include `quantile_table` and use `ALL_FEATURE_NAMES`:
```python
    model_data = {
        'model': pipeline,
        'features': ALL_FEATURE_NAMES,
        'quantile_table': quantile_table,
        'horizon': horizon,
        'up_thresh': up_thresh,
        'binary': True,
        'description': (
            f'Binary stock predictor. '
            f'Classes: 0=NOT-UP, 1=UP>{up_thresh*100:.0f}%. '
            f'Horizon: {horizon} trading days. '
            f'Features: {len(ALL_FEATURE_NAMES)} (21 base + 21 cross-sectional ranks).'
        ),
    }
```

- [ ] **Step 7: Run all train_model tests**

```bash
pytest tests/test_train_model.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: cross-sectional rank features in build_full_dataset; quantile table saved to model"
```

---

## Task 4: Optuna hyperparameter search

**Files:**
- Modify: `train_model.py`
- Modify: `requirements.txt`
- Modify: `tests/test_train_model.py`

- [ ] **Step 1: Add `optuna` to requirements**

In `requirements.txt`, append:
```
optuna>=3.0.0
```

Install it:
```bash
pip install optuna>=3.0.0
```

- [ ] **Step 2: Add `import optuna` at top of `train_model.py`**

Add after the existing imports:
```python
import optuna
```

- [ ] **Step 3: Write failing tests**

Add to `tests/test_train_model.py`:

```python
class TestOptunaBestParams:
    def test_returns_dict_with_all_expected_keys(self):
        mock_study = MagicMock()
        mock_study.best_value = 0.45
        mock_study.best_params = {
            'num_leaves': 100, 'learning_rate': 0.05, 'n_estimators': 300,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.0, 'reg_lambda': 0.0,
        }
        import pandas as pd
        from train_model import ALL_FEATURE_NAMES
        X = pd.DataFrame({f: np.ones(200) for f in ALL_FEATURE_NAMES})
        y = pd.Series([0]*120 + [1]*80)

        with patch("train_model.optuna") as mock_optuna:
            mock_optuna.logging = MagicMock()
            mock_optuna.logging.WARNING = 30
            mock_optuna.create_study.return_value = mock_study
            from train_model import _optuna_best_params
            result = _optuna_best_params(X, y, scale_pos_weight=1.5, n_trials=1)

        mock_optuna.create_study.assert_called_once_with(direction='maximize')
        mock_study.optimize.assert_called_once()
        assert result == mock_study.best_params

    def test_optuna_tune_false_skips_search(self, tmp_path):
        from train_model import ALL_FEATURE_NAMES
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text("0001.hk\n")

        X_tr = pd.DataFrame({f: np.ones(100) for f in ALL_FEATURE_NAMES})
        y_tr = pd.Series([0]*60 + [1]*40)
        X_te = pd.DataFrame({f: np.ones(25) for f in ALL_FEATURE_NAMES})
        y_te = pd.Series([0]*15 + [1]*10)

        mock_pipeline = MagicMock()
        mock_pipeline.fit.return_value = mock_pipeline
        mock_pipeline.predict.return_value = np.zeros(25, dtype=int)
        mock_pipeline.predict_proba.return_value = np.column_stack(
            [np.full(25, 0.6), np.full(25, 0.4)]
        )
        mock_pipeline.classes_ = np.array([0, 1])

        with patch("train_model.build_full_dataset", return_value=((X_tr, y_tr), (X_te, y_te), {})), \
             patch("train_model._optuna_best_params") as mock_opt, \
             patch("train_model.build_pipeline", return_value=mock_pipeline), \
             patch("joblib.dump"):
            from train_model import train_and_save
            train_and_save(str(csv_file), optuna_tune=False,
                           output_path=str(tmp_path / "model.joblib"))

        mock_opt.assert_not_called()
```

Add `from unittest.mock import MagicMock, patch` to the existing import if not already there (it is already there).

- [ ] **Step 4: Run to verify failure**

```bash
pytest tests/test_train_model.py::TestOptunaBestParams -v
```

Expected: `ImportError` — `_optuna_best_params` not defined.

- [ ] **Step 5: Implement `_optuna_best_params` in `train_model.py`**

Add after `_build_quantile_table`:

```python
def _optuna_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_trials: int = 50,
) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=0.30, stratify=y_train, random_state=42
    )
    split = int(len(X_sub) * 0.8)
    X_tr, X_val = X_sub.iloc[:split], X_sub.iloc[split:]
    y_tr, y_val = y_sub.iloc[:split], y_sub.iloc[split:]

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        }
        clf = LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_jobs=2,
            random_state=42,
            verbose=-1,
            **params,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        return precision_score(y_val, (proba >= 0.65).astype(int), zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"\nOptuna best precision@0.65: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params
```

- [ ] **Step 6: Add `optuna_tune` parameter to `train_and_save`**

Update signature:
```python
def train_and_save(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    clip: float = 0.30,
    output_path: str = 'stock_model.joblib',
    extra_csv_paths: list = None,
    optuna_tune: bool = False,
    n_trials: int = 50,
) -> None:
```

Replace the `pipeline = build_pipeline(scale_pos_weight=spw)` line with:

```python
    if optuna_tune:
        print(f"\nRunning Optuna ({n_trials} trials) — optimising precision@0.65 ...")
        best_params = _optuna_best_params(X_train, y_train, spw, n_trials=n_trials)
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', LGBMClassifier(
                scale_pos_weight=spw,
                n_jobs=2,
                random_state=42,
                verbose=-1,
                **best_params,
            ))
        ])
    else:
        pipeline = build_pipeline(scale_pos_weight=spw)
```

- [ ] **Step 7: Run all tests**

```bash
pytest tests/test_train_model.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add train_model.py requirements.txt tests/test_train_model.py
git commit -m "feat: add Optuna hyperparameter search with optuna_tune flag in train_and_save"
```

---

## Task 5: `add_rank_features` and inference updates in `app.py`

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_app.py`:

```python
class TestLoadModel:
    def test_returns_three_tuple(self, tmp_path):
        import joblib
        model_path = tmp_path / "model.joblib"
        mock_model = MagicMock()
        joblib.dump({
            'model': mock_model,
            'features': ['f1', 'f2'],
            'quantile_table': {'HK': {'f1': [1.0, 2.0]}},
        }, str(model_path))

        from app import _load_model
        result = _load_model(str(model_path))

        assert len(result) == 3
        assert result[1] == ['f1', 'f2']
        assert 'HK' in result[2]

    def test_missing_quantile_table_returns_empty_dict(self, tmp_path):
        import joblib
        model_path = tmp_path / "model.joblib"
        joblib.dump({'model': MagicMock(), 'features': []}, str(model_path))

        from app import _load_model
        _, _, qt = _load_model(str(model_path))
        assert qt == {}


class TestAddRankFeatures:
    def _make_feature_df(self):
        import pandas as pd
        from train_model import FEATURE_NAMES
        return pd.DataFrame({f: [50.0] for f in FEATURE_NAMES})

    def test_adds_rank_columns_for_all_features(self):
        import numpy as np
        from train_model import FEATURE_NAMES
        from app import add_rank_features
        df = self._make_feature_df()
        quantile_table = {
            'HK': {f: np.array([0.0, 50.0, 100.0]) for f in FEATURE_NAMES}
        }
        result = add_rank_features(df, '0001.HK', quantile_table)
        for feat in FEATURE_NAMES:
            assert f'{feat}_rank' in result.columns

    def test_rank_values_between_zero_and_one(self):
        import numpy as np
        from train_model import FEATURE_NAMES
        from app import add_rank_features
        df = self._make_feature_df()
        quantile_table = {
            'HK': {f: np.array([0.0, 50.0, 100.0]) for f in FEATURE_NAMES}
        }
        result = add_rank_features(df, '0001.HK', quantile_table)
        for feat in FEATURE_NAMES:
            val = result[f'{feat}_rank'].iloc[0]
            assert 0.0 <= val <= 1.0

    def test_empty_quantile_table_returns_df_unchanged(self):
        from app import add_rank_features
        df = self._make_feature_df()
        result = add_rank_features(df, '0001.HK', {})
        assert list(result.columns) == list(df.columns)

    def test_si_ticker_uses_sgx_table(self):
        import numpy as np
        from train_model import FEATURE_NAMES
        from app import add_rank_features
        df = self._make_feature_df()
        quantile_table = {
            'SGX': {f: np.array([0.0, 50.0, 100.0]) for f in FEATURE_NAMES},
        }
        result = add_rank_features(df, 'D05.SI', quantile_table)
        assert f'{FEATURE_NAMES[0]}_rank' in result.columns

    def test_unknown_ticker_falls_back_to_all(self):
        import numpy as np
        from train_model import FEATURE_NAMES
        from app import add_rank_features
        df = self._make_feature_df()
        quantile_table = {
            'ALL': {f: np.array([0.0, 50.0, 100.0]) for f in FEATURE_NAMES},
        }
        result = add_rank_features(df, 'AAPL', quantile_table)
        assert f'{FEATURE_NAMES[0]}_rank' in result.columns
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_app.py::TestLoadModel tests/test_app.py::TestAddRankFeatures -v
```

Expected: failures — `_load_model` returns 2-tuple, `add_rank_features` not defined.

- [ ] **Step 3: Update `_load_model` in `app.py`**

Replace the existing `_load_model` function:

```python
def _load_model(path):
    try:
        d = joblib.load(path)
        return d['model'], d['features'], d.get('quantile_table', {})
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None, [], {}
```

Update the two call sites (lines 26–27):

```python
model_5d, feature_names_5d, quantile_table_5d = _load_model(MODEL_5D_PATH)
model_14d, feature_names_14d, quantile_table_14d = _load_model(MODEL_14D_PATH)
```

- [ ] **Step 4: Add `add_rank_features` and `INDICATOR_KEYS` to `app.py`**

After the `_load_model` call sites, add:

```python
INDICATOR_KEYS = [
    'RSI_14', 'MACD_hist', 'SMA_20_ratio', 'Volume_ratio_20',
    'ATR_ratio', 'Returns_5d', 'Returns_20d', 'BB_pct_b',
]


def add_rank_features(df: pd.DataFrame, ticker: str, quantile_table: dict) -> pd.DataFrame:
    if not quantile_table:
        return df
    t = ticker.upper()
    if t.endswith('.HK'):
        exch = 'HK'
    elif t.endswith('.SI'):
        exch = 'SGX'
    else:
        exch = 'ALL'
    table = quantile_table.get(exch) or quantile_table.get('ALL', {})
    if not table:
        return df
    df = df.copy()
    for feat, sorted_arr in table.items():
        if feat in df.columns and len(sorted_arr) > 0:
            df[f'{feat}_rank'] = (
                np.searchsorted(sorted_arr, df[feat].values).astype(float) / len(sorted_arr)
            )
        else:
            df[f'{feat}_rank'] = 0.5
    return df
```

- [ ] **Step 5: Update `_build_prediction` to use rank features and capture indicators**

Replace the existing `_build_prediction`:

```python
def _build_prediction(ticker: str, mdl, feat_names: list, quantile_table: dict = None) -> dict | None:
    try:
        data = fetch_latest_data(ticker)
        if data is None:
            return None
        processed = calculate_technical_indicators(data)
        if processed.empty:
            return None
        if quantile_table:
            processed = add_rank_features(processed, ticker, quantile_table)
        latest = processed[feat_names].iloc[-1:].values
        result = build_prediction_response(ticker, mdl, latest, data)
        last = processed.iloc[-1]
        result['indicators'] = {
            k: round(float(last[k]), 6) for k in INDICATOR_KEYS if k in last.index
        }
        return result
    except Exception as e:
        print(f"Prediction error for {ticker}: {e}")
        return None
```

- [ ] **Step 6: Update `_build_dual_prediction` to use rank features**

Replace the `_build_dual_prediction` body:

```python
def _build_dual_prediction(ticker: str) -> dict | None:
    try:
        data = fetch_latest_data(ticker)
        if data is None:
            return None
        processed = calculate_technical_indicators(data)
        if processed.empty:
            return None

        qt = quantile_table_5d or quantile_table_14d or {}
        processed_r = add_rank_features(processed, ticker, qt)

        result = {
            "ticker": ticker,
            "current_price": float(data['Adj_Close'].iloc[-1]),
            "last_updated": str(data.index[-1].date()),
        }

        if model_5d is not None:
            latest = processed_r[feature_names_5d].iloc[-1:].values
            r5 = build_prediction_response(ticker, model_5d, latest, data)
            result.update({
                "signal_5d": r5["signal"],
                "confidence_up_5d": r5["confidence_up_3pct"],
                "edge_ratio_5d": r5["edge_ratio"],
                "prediction_5d": r5["prediction"],
            })

        if model_14d is not None:
            latest = processed_r[feature_names_14d].iloc[-1:].values
            r14 = build_prediction_response(ticker, model_14d, latest, data)
            result.update({
                "signal_14d": r14["signal"],
                "confidence_up_14d": r14["confidence_up_3pct"],
                "edge_ratio_14d": r14["edge_ratio"],
                "prediction_14d": r14["prediction"],
            })

        last = processed_r.iloc[-1]
        result['indicators'] = {
            k: round(float(last[k]), 6) for k in INDICATOR_KEYS if k in last.index
        }
        return result
    except Exception as e:
        print(f"Dual prediction error for {ticker}: {e}")
        return None
```

- [ ] **Step 7: Update `analyze` to pass `quantile_table` and return context state**

Replace `analyze`:

```python
def analyze(ticker: str, watchlist: list):
    ticker = _normalize_ticker(ticker)
    if not ticker:
        err = f'<p style="{_ERR_STYLE}">Enter a ticker symbol.</p>'
        return (err, "", *_wl_updates(watchlist), {}, "")

    html_5d = html_14d = ""
    r5 = r14 = None

    if model_5d is not None:
        r5 = _build_prediction(ticker, model_5d, feature_names_5d, quantile_table_5d)
        html_5d = (_result_html(r5, label="5-DAY FORECAST") if r5
                   else f'<p style="{_ERR_STYLE}">No data for {ticker} (5D).</p>')
    else:
        html_5d = f'<p style="{_ERR_STYLE}">5-day model not loaded.</p>'

    if model_14d is not None:
        r14 = _build_prediction(ticker, model_14d, feature_names_14d, quantile_table_14d)
        html_14d = (_result_html(r14, label="14-DAY FORECAST") if r14
                    else f'<p style="{_ERR_STYLE}">No data for {ticker} (14D).</p>')
    else:
        html_14d = f'<p style="{_ERR_STYLE}">14-day model not loaded.</p>'

    analyze_ctx = {}
    if r5 or r14:
        best = r5 or r14
        analyze_ctx = {
            'ticker': ticker,
            'price': best.get('current_price', 0),
            'result_5d': r5 or {},
            'result_14d': r14 or {},
            'indicators': best.get('indicators', {}),
        }

    if ticker not in watchlist and (r5 is not None or r14 is not None):
        watchlist = watchlist + [ticker]
        _save_watchlist(watchlist)

    return (html_5d, html_14d, *_wl_updates(watchlist), analyze_ctx, "")
```

- [ ] **Step 8: Update `scan_watchlist` to return raw results in state**

Replace `scan_watchlist`:

```python
def scan_watchlist(watchlist: list):
    if not watchlist:
        return f'<p style="{_ERR_STYLE}">Watchlist is empty.</p>', []
    if model_5d is None and model_14d is None:
        return f'<p style="{_ERR_STYLE}">No models loaded.</p>', []
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_build_dual_prediction, watchlist))
    valid = [r for r in results if r is not None]
    top10 = sorted(
        valid,
        key=lambda r: (r.get("confidence_up_5d", 0), r.get("edge_ratio_5d", 0)),
        reverse=True,
    )[:10]
    return _scan_html(top10), top10
```

- [ ] **Step 9: Run all tests**

```bash
pytest tests/test_app.py -v
```

Expected: all tests PASS. Note: `_analyze_outs` wiring in the UI block will need updating in Task 7.

- [ ] **Step 10: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add_rank_features and INDICATOR_KEYS in app.py; update _load_model, _build_prediction, _build_dual_prediction"
```

---

## Task 6: `llm_utils.py` and LangChain dependencies

**Files:**
- Create: `llm_utils.py`
- Create: `tests/test_llm_utils.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add LangChain deps to `requirements.txt`**

Append:
```
langchain-ollama>=0.2.0
langchain-groq>=0.2.0
langchain-google-genai>=2.0.0
```

Install:
```bash
pip install langchain-ollama langchain-groq langchain-google-genai
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_llm_utils.py`:

```python
import pytest
from unittest.mock import MagicMock, patch


class TestGetLlm:
    def test_ollama_local_instantiates_chatollama(self):
        with patch("llm_utils.ChatOllama") as mock_cls:
            from llm_utils import get_llm
            get_llm("ollama_local")
        mock_cls.assert_called_once()

    def test_groq_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from llm_utils import get_llm
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            get_llm("groq")

    def test_gemini_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from llm_utils import get_llm
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            get_llm("gemini")


class TestExplainSignal:
    def _make_mock_llm(self, content="Test summary."):
        mock_resp = MagicMock()
        mock_resp.content = content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        return mock_llm

    def test_returns_llm_response_content(self):
        mock_llm = self._make_mock_llm("The model sees moderate UP momentum.")
        with patch("llm_utils.get_llm", return_value=mock_llm):
            from llm_utils import explain_signal
            result = explain_signal(
                ticker="0001.HK", price=57.35,
                result_5d={"signal": "UP >3%", "confidence_up_3pct": 0.62, "edge_ratio": 1.24},
                result_14d={"signal": "NO SIGNAL", "confidence_up_3pct": 0.40, "edge_ratio": 0.80},
                indicators={"RSI_14": 58.0, "MACD_hist": 0.02},
            )
        assert result == "The model sees moderate UP momentum."

    def test_prompt_includes_ticker_and_price(self):
        mock_llm = self._make_mock_llm()
        with patch("llm_utils.get_llm", return_value=mock_llm):
            from llm_utils import explain_signal
            explain_signal(
                ticker="0001.HK", price=57.35,
                result_5d={"signal": "UP >3%", "confidence_up_3pct": 0.62, "edge_ratio": 1.24},
                result_14d={"signal": "NO SIGNAL", "confidence_up_3pct": 0.40, "edge_ratio": 0.80},
                indicators={},
            )
        messages = mock_llm.invoke.call_args[0][0]
        combined = " ".join(m.content for m in messages)
        assert "0001.HK" in combined
        assert "57.35" in combined

    def test_prompt_includes_both_horizon_signals(self):
        mock_llm = self._make_mock_llm()
        with patch("llm_utils.get_llm", return_value=mock_llm):
            from llm_utils import explain_signal
            explain_signal(
                ticker="0001.HK", price=10.0,
                result_5d={"signal": "UP >3%", "confidence_up_3pct": 0.62, "edge_ratio": 1.24},
                result_14d={"signal": "NO SIGNAL", "confidence_up_3pct": 0.40, "edge_ratio": 0.80},
                indicators={},
            )
        messages = mock_llm.invoke.call_args[0][0]
        combined = " ".join(m.content for m in messages)
        assert "5-day" in combined
        assert "14-day" in combined


class TestSummarizeScan:
    def test_empty_rows_returns_no_results_message(self):
        from llm_utils import summarize_scan
        result = summarize_scan([])
        assert "No scan results" in result

    def test_returns_llm_content(self):
        mock_resp = MagicMock()
        mock_resp.content = "Two HK financials showing UP signals."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        scan_rows = [
            {"ticker": "0001.HK", "signal_5d": "UP >3%", "confidence_up_5d": 0.62,
             "signal_14d": "UP >3%", "confidence_up_14d": 0.67},
        ]
        with patch("llm_utils.get_llm", return_value=mock_llm):
            from llm_utils import summarize_scan
            result = summarize_scan(scan_rows)
        assert result == "Two HK financials showing UP signals."

    def test_prompt_includes_ticker_names(self):
        mock_resp = MagicMock()
        mock_resp.content = "summary"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        scan_rows = [
            {"ticker": "0001.HK", "signal_5d": "UP >3%", "confidence_up_5d": 0.62,
             "signal_14d": "NO SIGNAL", "confidence_up_14d": 0.38},
        ]
        with patch("llm_utils.get_llm", return_value=mock_llm):
            from llm_utils import summarize_scan
            summarize_scan(scan_rows)
        messages = mock_llm.invoke.call_args[0][0]
        combined = " ".join(m.content for m in messages)
        assert "0001.HK" in combined
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_llm_utils.py -v
```

Expected: `ModuleNotFoundError` — `llm_utils` not found.

- [ ] **Step 4: Create `llm_utils.py`**

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import os

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3-flash-preview"
OLLAMA_LOCAL_MODEL = "macdev/gpt-oss20b-large-ctx"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama_local")

_INDICATOR_LABELS = {
    'RSI_14': 'RSI (14-day)',
    'MACD_hist': 'MACD Histogram',
    'SMA_20_ratio': 'Price/SMA20 ratio',
    'Volume_ratio_20': 'Volume ratio (20d avg)',
    'ATR_ratio': 'ATR ratio',
    'Returns_5d': '5-day return',
    'Returns_20d': '20-day return',
    'BB_pct_b': 'Bollinger %B',
}

_EXPLAIN_SYSTEM = (
    "You are a financial signal interpreter. "
    "Summarise the model output in 3-5 plain-English sentences. "
    "Do not give buy/sell advice. Do not invent information not provided."
)

_SCAN_SYSTEM = (
    "You are a financial signal interpreter. "
    "Summarise these watchlist scan results in one short paragraph. "
    "Note any patterns, sector clusters, or strong signals. "
    "Do not give buy/sell advice. Do not invent information not provided."
)


def get_llm(provider="ollama_local", temperature=0):
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, temperature=temperature, google_api_key=api_key
        )

    return ChatOllama(model=OLLAMA_LOCAL_MODEL, temperature=temperature)


def explain_signal(
    ticker: str,
    price: float,
    result_5d: dict,
    result_14d: dict,
    indicators: dict,
) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Current price: {price:.3f}",
        "",
        "5-day model:",
        f"  Signal: {result_5d.get('signal', 'N/A')}",
        f"  UP confidence: {result_5d.get('confidence_up_3pct', 0)*100:.1f}%",
        f"  Edge ratio: {result_5d.get('edge_ratio', 0):.2f}x",
        "",
        "14-day model:",
        f"  Signal: {result_14d.get('signal', 'N/A')}",
        f"  UP confidence: {result_14d.get('confidence_up_3pct', 0)*100:.1f}%",
        f"  Edge ratio: {result_14d.get('edge_ratio', 0):.2f}x",
        "",
        "Technical indicators:",
    ]
    for key, label in _INDICATOR_LABELS.items():
        val = indicators.get(key)
        if val is not None:
            lines.append(f"  {label}: {val:.4f}")

    llm = get_llm(LLM_PROVIDER)
    response = llm.invoke([
        SystemMessage(content=_EXPLAIN_SYSTEM),
        HumanMessage(content="\n".join(lines)),
    ])
    return response.content


def summarize_scan(scan_rows: list) -> str:
    if not scan_rows:
        return "No scan results to summarise."

    lines = ["Scan results:"]
    for r in scan_rows:
        sig5 = r.get('signal_5d', 'N/A')
        conf5 = round(r.get('confidence_up_5d', 0) * 100, 1)
        sig14 = r.get('signal_14d', 'N/A')
        conf14 = round(r.get('confidence_up_14d', 0) * 100, 1)
        lines.append(
            f"  {r['ticker']}: 5d={sig5} ({conf5}%), 14d={sig14} ({conf14}%)"
        )

    llm = get_llm(LLM_PROVIDER)
    response = llm.invoke([
        SystemMessage(content=_SCAN_SYSTEM),
        HumanMessage(content="\n".join(lines)),
    ])
    return response.content
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_llm_utils.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add llm_utils.py tests/test_llm_utils.py requirements.txt
git commit -m "feat: add llm_utils.py with get_llm, explain_signal, summarize_scan"
```

---

## Task 7: LLM UI buttons in `app.py`

**Files:**
- Modify: `app.py`

This task wires the LLM functions into the Gradio UI. Unit tests cover the wrapper functions; the UI layout is verified manually by launching the app.

- [ ] **Step 1: Add wrapper functions to `app.py`**

Add after `scan_watchlist`:

```python
def _explain(ctx: dict) -> str:
    if not ctx:
        return "*Analyse a stock first, then click Explain.*"
    from llm_utils import explain_signal
    try:
        return explain_signal(
            ticker=ctx['ticker'],
            price=ctx['price'],
            result_5d=ctx['result_5d'],
            result_14d=ctx['result_14d'],
            indicators=ctx['indicators'],
        )
    except Exception as e:
        return f"LLM error: {e}"


def _summarize(scan_rows: list) -> str:
    if not scan_rows:
        return "*Run a scan first, then click Summarise.*"
    from llm_utils import summarize_scan
    try:
        return summarize_scan(scan_rows)
    except Exception as e:
        return f"LLM error: {e}"
```

- [ ] **Step 2: Add states and UI elements to the Gradio block**

In the `with gr.Blocks(...) as demo:` block, add two new state variables near the top (after `watchlist_state`):

```python
    analyze_ctx_state = gr.State({})
    scan_results_state = gr.State([])
```

After the `with gr.Row():` block that holds `result_out_5d` and `result_out_14d`, add:

```python
    explain_btn = gr.Button("EXPLAIN SIGNAL", variant="secondary")
    explain_out = gr.Markdown(value="")
```

After `scan_out = gr.HTML(value="")`, add:

```python
    summarize_btn = gr.Button("SUMMARISE SCAN", variant="secondary")
    scan_summary_out = gr.Markdown(value="")
```

- [ ] **Step 3: Update `_analyze_outs` and rewire all analyze handlers**

Replace the wiring section:

```python
    _wl_outs = [watchlist_state, remove_cg]
    _analyze_outs = [result_out_5d, result_out_14d] + _wl_outs + [analyze_ctx_state, explain_out]

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

Add wiring for the new buttons:

```python
    explain_btn.click(
        fn=_explain,
        inputs=[analyze_ctx_state],
        outputs=[explain_out],
    )

    scan_btn.click(
        fn=scan_watchlist,
        inputs=[watchlist_state],
        outputs=[scan_out, scan_results_state],
    )

    summarize_btn.click(
        fn=_summarize,
        inputs=[scan_results_state],
        outputs=[scan_summary_out],
    )
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS (UI wiring is not unit-tested).

- [ ] **Step 5: Manual smoke test**

```bash
python app.py
```

Open `http://localhost:7860`. Verify:
1. Analyse a ticker → result cards appear → click "EXPLAIN SIGNAL" → LLM text appears below
2. Scan watchlist → table appears → click "SUMMARISE SCAN" → LLM summary appears below
3. Analysing a second ticker clears the previous explain text

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add Explain Signal and Summarise Scan LLM buttons to app.py UI"
```

---

## Self-Review Notes

- **Spec coverage:** All three major sections covered — cross-sectional ranks (Tasks 1–3), Optuna (Task 4), LLM layer (Tasks 6–7). Inference path update in Task 5 bridges training and UI.
- **Type consistency:** `quantile_table` is `dict` throughout; `_load_model` returns `(Pipeline|None, list, dict)`; `add_rank_features` accepts/returns `pd.DataFrame`.
- **No placeholders:** All code blocks are complete and runnable.
- **Breaking change handled:** Existing `build_full_dataset` 2-tuple unpackings are fixed in Task 3 Step 2 before the implementation changes in Step 5.
