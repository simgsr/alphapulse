import numpy as np
import os
import pandas as pd
from unittest.mock import patch
import pytest
from train_model import binarize_return, load_tickers, FEATURE_NAMES as _NEW_FEATURES, ALL_FEATURE_NAMES


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


class TestLoadTickers:
    def test_returns_tickers_from_single_column_csv(self, tmp_path):
        csv_content = "0001.hk\n0002.hk\n0003.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        tickers = load_tickers(str(csv_file))

        assert "0001.hk" in tickers
        assert "0002.hk" in tickers
        assert "0003.hk" in tickers

    def test_returns_list_of_strings(self, tmp_path):
        csv_content = "0001.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        tickers = load_tickers(str(csv_file))

        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)


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

    def test_target_values_are_binary(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert set(result["target"].unique()).issubset({0, 1})

    def test_no_nan_in_result(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result["forward_return"].isna().sum() == 0

    def test_forward_return_capped_at_30pct(self):
        long_df = _make_synthetic_df(400)
        long_df = long_df.copy()
        long_df.iloc[100, long_df.columns.get_loc("Adj_Close")] = 1e9  # extreme spike
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert result["forward_return"].max() <= 0.30
        assert result["forward_return"].min() >= -0.30

    def test_5d_horizon_uses_shift_5(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk", horizon=5, up_thresh=0.03, clip=0.20)
        assert result is not None
        assert result["forward_return"].max() <= 0.20
        assert result["forward_return"].min() >= -0.20

    def test_14d_horizon_uses_shift_14(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk", horizon=14, up_thresh=0.03, clip=0.30)
        assert result is not None
        assert set(result["target"].unique()).issubset({0, 1})


class TestBuildFullDataset:
    def _make_labeled_df(self, n_rows: int, start: str, exchange: str = 'HK') -> pd.DataFrame:
        """Minimal labeled DataFrame as returned by build_ticker_dataset."""
        idx = pd.date_range(start, periods=n_rows, freq="B")
        data = {f: np.random.rand(n_rows) for f in _NEW_FEATURES}
        data['Adj_Close'] = np.ones(n_rows) * 100.0
        data['forward_return'] = np.zeros(n_rows)
        data['target'] = 0
        data['exchange'] = exchange
        return pd.DataFrame(data, index=idx)

    def test_80_20_split_sizes(self, tmp_path):
        csv_content = "0001.hk\n0002.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")
        df_b = self._make_labeled_df(300, "2020-01-01")

        def mock_build(ticker, period='5y', **kwargs):
            return df_a if ticker == '0001.hk' else df_b

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            (X_train, y_train), (X_test, y_test), _, _ = build_full_dataset(str(csv_file))

        total = len(X_train) + len(X_test)
        assert abs(len(X_train) / total - 0.8) < 0.01

    def test_features_subset_only(self, tmp_path):
        csv_content = "0001.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")

        with patch("train_model.build_ticker_dataset", return_value=df_a):
            from train_model import build_full_dataset
            (X_train, y_train), (X_test, y_test), _, _ = build_full_dataset(str(csv_file))

        assert list(X_train.columns) == ALL_FEATURE_NAMES
        assert list(X_test.columns) == ALL_FEATURE_NAMES

    def test_skips_none_tickers(self, tmp_path):
        csv_content = "0001.hk\n0002.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")

        def mock_build(ticker, period='5y', **kwargs):
            return df_a if ticker == '0001.hk' else None

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            (X_train, y_train), _, _qt, _ = build_full_dataset(str(csv_file))

        assert len(X_train) > 0

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

    def test_returns_quantile_table(self, tmp_path):
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text("0001.hk\n")

        df_a = self._make_labeled_df(300, "2020-01-01", exchange='HK')

        with patch("train_model.build_ticker_dataset", return_value=df_a):
            from train_model import build_full_dataset
            _, _, qt, _ = build_full_dataset(str(csv_file))

        assert isinstance(qt, dict)
        assert 'ALL' in qt
        assert 'HK' in qt
        for feat in _NEW_FEATURES:
            assert feat in qt['ALL']


class TestBuildFullDatasetSGX:
    def _make_labeled_df(self, n_rows: int, start: str, exchange: str = 'HK') -> pd.DataFrame:
        from train_model import FEATURE_NAMES as _FEATS
        idx = pd.date_range(start, periods=n_rows, freq="B")
        data = {f: np.random.rand(n_rows) for f in _FEATS}
        data['Adj_Close'] = np.ones(n_rows) * 100.0
        data['forward_return'] = np.zeros(n_rows)
        data['target'] = 0
        data['exchange'] = exchange
        return pd.DataFrame(data, index=idx)

    def test_extra_csv_merges_tickers(self, tmp_path):
        hk_csv = tmp_path / "hkex.csv"
        hk_csv.write_text("0001.hk\n")
        sg_csv = tmp_path / "sgx_tickers.csv"
        sg_csv.write_text("D05.SI\n")

        df_a = self._make_labeled_df(300, "2020-01-01", exchange='HK')
        df_b = self._make_labeled_df(300, "2020-06-01", exchange='SGX')
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
            (X_train, _), _, _qt, _ = build_full_dataset(str(hk_csv))

        assert len(X_train) > 0


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
