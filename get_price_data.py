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

    # RSI (two speeds); fillna(50) handles flat-price edge case where gain=loss=0
    for window, col in [(14, 'RSI_14'), (7, 'RSI_7')]:
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        df[col] = (100 - (100 / (1 + gain / loss))).fillna(50)

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

    # Bollinger Band %B (20-day, 2 std); fillna(0.5) handles flat-price edge case
    # where std=0 so upper=lower, producing 0/0
    bb_std = df['Adj_Close'].rolling(window=20).std()
    bb_upper = df['SMA_20'] + 2 * bb_std
    bb_lower = df['SMA_20'] - 2 * bb_std
    band_range = (bb_upper - bb_lower).replace(0, np.nan)
    df['BB_pct_b'] = ((df['Adj_Close'] - bb_lower) / band_range).fillna(0.5)

    # Volume ratio vs 20-day average
    df['Volume_ratio_20'] = df['Adj_Volume'] / df['Adj_Volume'].rolling(window=20).mean()

    # Volatility and multi-horizon returns
    df['Volatility_20'] = df['Adj_Close'].pct_change().rolling(window=20).std()
    df['Returns_1d'] = df['Adj_Close'].pct_change()
    df['Returns_5d'] = df['Adj_Close'].pct_change(5)
    df['Returns_10d'] = df['Adj_Close'].pct_change(10)
    df['Returns_20d'] = df['Adj_Close'].pct_change(20)

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

    return df.dropna()
