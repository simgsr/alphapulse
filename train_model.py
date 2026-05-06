import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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
