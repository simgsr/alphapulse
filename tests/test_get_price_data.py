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
