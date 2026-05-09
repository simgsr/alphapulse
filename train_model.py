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
    'Stoch_K', 'Stoch_D',
    'ATR_ratio',
    'ADX_14',
    'OBV_ratio',
    'CCI_20',
    'CMF_20',
]

RANK_FEATURE_NAMES = [f'{f}_rank' for f in FEATURE_NAMES]
ALL_FEATURE_NAMES = FEATURE_NAMES + RANK_FEATURE_NAMES


def _ticker_exchange(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith('.HK'):
        return 'HK'
    if t.endswith('.SI'):
        return 'SGX'
    return 'ALL'


def binarize_return(r: float, thresh: float = 0.03) -> int:
    """Map a forward return to a binary label.

    Returns 1 if r > thresh (UP), 0 otherwise (NOT-UP).
    """
    return 1 if r > thresh else 0


def load_tickers(csv_path: str) -> list:
    """Return ticker symbols from a single-column CSV (no header required)."""
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].dropna().str.strip().tolist()


def build_ticker_dataset(
    ticker: str,
    period: str = '5y',
    horizon: int = 7,
    up_thresh: float = 0.03,
    clip: float = 0.30,
) -> Optional[pd.DataFrame]:
    """Fetch, engineer features, and label one ticker.

    Returns None if data is unavailable or fewer than 252 labeled rows remain.
    """
    raw = fetch_latest_data(ticker, period=period)
    if raw is None:
        return None
    df = calculate_technical_indicators(raw)
    df = df.copy()
    df['exchange'] = _ticker_exchange(ticker)
    df['forward_return'] = df['Adj_Close'].shift(-horizon) / df['Adj_Close'] - 1
    df = df.dropna(subset=['forward_return'])
    df['forward_return'] = df['forward_return'].clip(-clip, clip)
    df['target'] = df['forward_return'].apply(
        lambda r: binarize_return(r, thresh=up_thresh)
    )
    if len(df) < 252:
        return None
    return df


def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    extra_csv_paths: list = None,
) -> Tuple[Tuple, Tuple]:
    """Fetch all tickers, combine, and time-split 80/20.

    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    tickers = load_tickers(csv_path)
    if extra_csv_paths:
        for extra in extra_csv_paths:
            tickers += load_tickers(extra)
        tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    frames = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] Processing tickers...", flush=True)
        df = build_ticker_dataset(ticker, horizon=horizon, up_thresh=up_thresh)
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


def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build the unfitted sklearn Pipeline.

    RobustScaler is resistant to HK small-cap outliers.
    scale_pos_weight = count(NOT-UP) / count(UP) balances the binary imbalance.
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LGBMClassifier(
            n_estimators=500,
            num_leaves=95,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
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


def train_and_save(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    clip: float = 0.30,
    output_path: str = 'stock_model.joblib',
    extra_csv_paths: list = None,
) -> None:
    """Full training run: load data, fit pipeline, evaluate, save model."""
    from sklearn.metrics import precision_score, recall_score
    print(f"\n=== AlphaPulse Model Training — {horizon}-day horizon ===")
    print(f"Ticker source : {csv_path}")
    print(f"UP threshold  : > {up_thresh*100:.0f}%  (binary: NOT-UP otherwise)")
    print(f"Clip range    : ±{clip*100:.0f}%")
    print(f"Output path   : {output_path}\n")

    (X_train, y_train), (X_test, y_test) = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        extra_csv_paths=extra_csv_paths,
    )

    print(f"\nTraining rows : {len(X_train):,}")
    print(f"Test rows     : {len(X_test):,}")
    print("\nClass distribution (train set):")
    print(y_train.value_counts().sort_index().to_string())

    spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"\nscale_pos_weight : {spw:.2f}  (NOT-UP / UP ratio)")
    pipeline = build_pipeline(scale_pos_weight=spw)
    print("Fitting pipeline — this may take a few minutes...")
    pipeline.fit(X_train, y_train)
    print("Fitting complete.")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print("\n=== Held-out Test Set Evaluation ===")
    label_names = ["NOT-UP", f"UP>{up_thresh*100:.0f}%"]
    print(classification_report(y_test, y_pred, target_names=label_names))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log-loss : {log_loss(y_test, y_proba):.4f}")

    classes_list = list(pipeline.classes_)
    up_idx = classes_list.index(1)
    y_proba_up = y_proba[:, up_idx]
    y_test_bin = y_test  # already binary
    thresholds = np.arange(0.50, 0.81, 0.05)
    print(f"\n=== UP class Precision / Recall sweep ===")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Signals':>10}")
    for t in thresholds:
        y_pred_t = (y_proba_up >= t).astype(int)
        n_sig = int(y_pred_t.sum())
        prec = precision_score(y_test_bin, y_pred_t, zero_division=0)
        rec = recall_score(y_test_bin, y_pred_t, zero_division=0)
        print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {n_sig:>10}")

    model_data = {
        'model': pipeline,
        'features': FEATURE_NAMES,
        'horizon': horizon,
        'up_thresh': up_thresh,
        'binary': True,
        'description': (
            f'Binary stock predictor. '
            f'Classes: 0=NOT-UP, 1=UP>{up_thresh*100:.0f}%. '
            f'Horizon: {horizon} trading days.'
        ),
    }
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == '__main__':
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _hk_csv = find_ticker_csv(_data_dir)
    _sgx_csv = os.path.join(_data_dir, 'sgx_tickers.csv')
    _extra = [_sgx_csv] if os.path.exists(_sgx_csv) else None
    if _extra:
        print(f"SGX tickers found: {_sgx_csv}")

    train_and_save(
        _hk_csv,
        horizon=5,
        up_thresh=0.03,
        clip=0.20,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_model_5d.joblib'),
        extra_csv_paths=_extra,
    )
    train_and_save(
        _hk_csv,
        horizon=14,
        up_thresh=0.03,
        clip=0.30,
        output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_model_14d.joblib'),
        extra_csv_paths=_extra,
    )
