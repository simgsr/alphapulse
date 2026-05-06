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

    return df.dropna()
