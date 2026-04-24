import yfinance as yf
import pandas as pd
import numpy as np

def fetch_latest_data(ticker, period="60d"):
    """
    Fetch the latest historical data for a ticker.
    Returns a cleaned DataFrame suitable for feature engineering.
    """
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return None
        
        # Handle MultiIndex columns from yfinance 1.3.0+
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Standardize column names
        data = data[['Close', 'Volume']].copy()
        data.columns = ['Adj_Close', 'Adj_Volume']
        
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def calculate_technical_indicators(df):
    """
    Calculate the same features used during model training.
    """
    df = df.copy()
    
    # RSI Calculation
    window = 14
    delta = df['Adj_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA_5'] = df['Adj_Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Adj_Close'].rolling(window=20).mean()
    df['SMA_5_ratio'] = df['Adj_Close'] / df['SMA_5']
    df['SMA_20_ratio'] = df['Adj_Close'] / df['SMA_20']
    
    # Volatility and Returns
    df['Volatility_20'] = df['Adj_Close'].pct_change().rolling(window=20).std()
    df['Returns_1d'] = df['Adj_Close'].pct_change()
    df['Returns_5d'] = df['Adj_Close'].pct_change(5)
    
    return df.dropna()
