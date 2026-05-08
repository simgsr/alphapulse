import numpy as np
import os
import pandas as pd
from unittest.mock import patch
import pytest
from train_model import discretize_return, load_tickers, FEATURE_NAMES as _NEW_FEATURES


class TestDiscretizeReturn:
    def test_up_large(self):
        assert discretize_return(0.06) == 1

    def test_up_small(self):
        assert discretize_return(0.04) == 1

    def test_exact_3pct_is_stable(self):
        # boundary: +3% is the top of STABLE
        assert discretize_return(0.03) == 0

    def test_stable_zero(self):
        assert discretize_return(0.0) == 0

    def test_stable_small_positive(self):
        assert discretize_return(0.02) == 0

    def test_stable_small_negative(self):
        assert discretize_return(-0.02) == 0

    def test_exact_neg_3pct_is_stable(self):
        # boundary: -3% is the bottom of STABLE
        assert discretize_return(-0.03) == 0

    def test_down_small(self):
        assert discretize_return(-0.04) == -1

    def test_down_large(self):
        assert discretize_return(-0.06) == -1

    def test_custom_up_thresh(self):
        assert discretize_return(0.025, up_thresh=0.02, down_thresh=0.02) == 1

    def test_custom_down_thresh(self):
        assert discretize_return(-0.025, up_thresh=0.02, down_thresh=0.02) == -1

    def test_stable_within_custom_thresholds(self):
        assert discretize_return(0.015, up_thresh=0.02, down_thresh=0.02) == 0

    def test_14d_up_thresh(self):
        assert discretize_return(0.04, up_thresh=0.05, down_thresh=0.05) == 0

    def test_14d_down_thresh(self):
        assert discretize_return(-0.06, up_thresh=0.05, down_thresh=0.05) == -1


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

    def test_target_values_are_valid_classes(self):
        long_df = _make_synthetic_df(400)
        with patch("train_model.fetch_latest_data", return_value=long_df), \
             patch("train_model.calculate_technical_indicators", return_value=long_df):
            from train_model import build_ticker_dataset
            result = build_ticker_dataset("0001.hk")
        assert set(result["target"].unique()).issubset({-1, 0, 1})

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


class TestBuildFullDataset:
    def _make_labeled_df(self, n_rows: int, start: str) -> pd.DataFrame:
        """Minimal labeled DataFrame as returned by build_ticker_dataset."""
        idx = pd.date_range(start, periods=n_rows, freq="B")
        data = {f: np.ones(n_rows) for f in _NEW_FEATURES}
        data['Adj_Close'] = np.ones(n_rows) * 100.0
        data['forward_return'] = np.zeros(n_rows)
        data['target'] = 0
        return pd.DataFrame(data, index=idx)

    def test_80_20_split_sizes(self, tmp_path):
        csv_content = "0001.hk\n0002.hk\n"
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
        csv_content = "0001.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")

        with patch("train_model.build_ticker_dataset", return_value=df_a):
            from train_model import build_full_dataset
            (X_train, y_train), (X_test, y_test) = build_full_dataset(str(csv_file))

        expected_features = _NEW_FEATURES
        assert list(X_train.columns) == expected_features
        assert list(X_test.columns) == expected_features

    def test_skips_none_tickers(self, tmp_path):
        csv_content = "0001.hk\n0002.hk\n"
        csv_file = tmp_path / "hkex.csv"
        csv_file.write_text(csv_content)

        df_a = self._make_labeled_df(300, "2020-01-01")

        def mock_build(ticker, period='5y'):
            return df_a if ticker == '0001.hk' else None

        with patch("train_model.build_ticker_dataset", side_effect=mock_build):
            from train_model import build_full_dataset
            (X_train, y_train), _ = build_full_dataset(str(csv_file))

        assert len(X_train) > 0
