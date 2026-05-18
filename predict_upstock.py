"""
Predict top 5 UP stocks.

Loads stock_model_7d.joblib and stock_model_14d.joblib, fetches live data for
every ticker in the supplied CSV, and ranks them by P(UP > 3%) confidence.

Usage:
    python predict_upstock.py <ticker_csv>
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

from get_price_data import fetch_latest_data, calculate_technical_indicators, fetch_regime_data
from train_model import FEATURE_NAMES, REGIME_FEATURE_NAMES, ALL_FEATURE_NAMES, _ticker_exchange

# Minimum P(UP>3%) to show a ticker as a signal (classifier mode only).
SIGNAL_THRESHOLD = 0.33


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {path}')
    return joblib.load(path)


def fetch_snapshots(tickers: list[str]) -> tuple[list, list]:
    records, failed = [], []
    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            print(f'  [{i}/{len(tickers)}] fetching data...', flush=True)
        raw = fetch_latest_data(ticker, period='120d')
        if raw is None or raw.empty:
            failed.append(ticker)
            continue
        df = calculate_technical_indicators(raw)
        if df.empty:
            failed.append(ticker)
            continue
        last = df.iloc[[-1]].copy()
        last['ticker'] = ticker
        last['exchange'] = _ticker_exchange(ticker)
        records.append(last)
    return records, failed


def build_feature_matrix(snapshot: pd.DataFrame, regime_snapshot: dict) -> pd.DataFrame:
    for feat in FEATURE_NAMES:
        rank_col = f'{feat}_rank'
        snapshot[rank_col] = snapshot.groupby('exchange')[feat].rank(pct=True)
        nan_mask = snapshot[rank_col].isna()
        if nan_mask.any():
            snapshot.loc[nan_mask, rank_col] = snapshot.loc[nan_mask, feat].rank(pct=True)

    # Merge today's market regime values by exchange
    for feat in REGIME_FEATURE_NAMES:
        snapshot[feat] = np.nan
    for exch in snapshot['exchange'].unique():
        row = regime_snapshot.get(exch) or regime_snapshot.get('ALL')
        if row is not None:
            mask = snapshot['exchange'] == exch
            for feat in REGIME_FEATURE_NAMES:
                if feat in row.index:
                    snapshot.loc[mask, feat] = row[feat]

    return snapshot[ALL_FEATURE_NAMES]


def top5_table(model_data: dict, X: pd.DataFrame, ticker_list: list[str], label: str) -> pd.DataFrame:
    model_type = model_data.get('model_type', 'classifier')

    if model_type == 'ranker':
        scaler = model_data['scaler']
        ranker = model_data['ranker']
        X_sc = np.asarray(scaler.transform(X))
        scores = ranker.predict(X_sc)
        df = pd.DataFrame({'Ticker': ticker_list, 'Score': scores})
        df = df.sort_values('Score', ascending=False).head(5).reset_index(drop=True)
        df.index = range(1, len(df) + 1)
        df.index.name = 'Rank'
        df['Score'] = df['Score'].map('{:.4f}'.format)
        score_col = 'Score'
    else:
        pipeline = model_data['model']
        up_idx = list(pipeline.classes_).index(1)
        proba = pipeline.predict_proba(X)[:, up_idx]
        df = pd.DataFrame({'Ticker': ticker_list, 'P(UP>3%)': proba})
        df = df[df['P(UP>3%)'] >= SIGNAL_THRESHOLD].sort_values('P(UP>3%)', ascending=False).head(5).reset_index(drop=True)
        df.index = range(1, len(df) + 1)
        df.index.name = 'Rank'
        df['P(UP>3%)'] = df['P(UP>3%)'].map('{:.1%}'.format)
        score_col = 'P(UP>3%)'

    print(f'\n{"="*40}')
    print(f'  TOP 5  —  {label}')
    print(f'{"="*40}')
    if df.empty:
        print(f'  No tickers above {SIGNAL_THRESHOLD:.0%} confidence threshold.')
    else:
        print(df.to_string())
    return df


def main():
    parser = argparse.ArgumentParser(description='Predict top 5 UP stocks.')
    parser.add_argument('csv', help='Path to single-column ticker CSV file')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent

    model_7d = load_model(project_dir / 'stock_model_7d.joblib')
    model_14d = load_model(project_dir / 'stock_model_14d.joblib')
    print(f'7-day  model : {model_7d["description"]}')
    print(f'14-day model : {model_14d["description"]}')

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f'File not found: {csv_path}')

    tickers = pd.read_csv(csv_path, header=None).iloc[:, 0].dropna().str.strip().tolist()
    print(f'Loaded {len(tickers)} tickers from {csv_path}')

    print('Fetching market regime data...')
    regime_data = fetch_regime_data(period='1y')
    regime_snapshot = {exch: df.iloc[-1] for exch, df in regime_data.items() if not df.empty}
    print(f'  Regime exchanges available: {list(regime_snapshot.keys())}')

    records, failed = fetch_snapshots(tickers)
    print(f'\nSuccessful: {len(records)}  |  Failed: {len(failed)}')
    if failed:
        shown = failed[:15]
        more = f' ... and {len(failed) - 15} more' if len(failed) > 15 else ''
        print(f'Failed: {", ".join(shown)}{more}')

    if not records:
        sys.exit('No valid ticker data fetched. Check your CSV and network connection.')

    snapshot = pd.concat(records, ignore_index=True)
    X = build_feature_matrix(snapshot, regime_snapshot)
    print(f'Feature matrix: {X.shape}  ({X.shape[0]} tickers, {X.shape[1]} features)')

    ticker_list = snapshot['ticker'].tolist()
    top5_table(model_7d,  X, ticker_list, '7-Day Horizon  (P(UP>3%))')
    top5_table(model_14d, X, ticker_list, '14-Day Horizon (P(UP>3%))')


if __name__ == '__main__':
    main()
