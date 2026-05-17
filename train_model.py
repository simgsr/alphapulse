import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, log_loss, accuracy_score,
    average_precision_score, precision_score, recall_score,
)
from lightgbm import (
    LGBMClassifier,
    early_stopping as lgb_early_stopping,
    log_evaluation as lgb_log_evaluation,
)
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
    raw_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Fetch, engineer features, and label one ticker.

    Returns None if data is unavailable or fewer than 252 labeled rows remain.
    Pass raw_df to skip the fetch+indicator step (reuse cached data).
    """
    if raw_df is None:
        raw = fetch_latest_data(ticker, period=period)
        if raw is None:
            return None
        raw_df = calculate_technical_indicators(raw)
    df = raw_df.copy()
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


def _build_quantile_table(train_df: pd.DataFrame) -> dict:
    """Build a quantile lookup table from training rows, keyed by exchange."""
    table: dict = {}
    for exch in train_df['exchange'].unique():
        exch_rows = train_df[train_df['exchange'] == exch]
        table[exch] = {
            feat: np.sort(exch_rows[feat].dropna().values)
            for feat in FEATURE_NAMES
        }
    table['ALL'] = {
        feat: np.sort(train_df[feat].dropna().values)
        for feat in FEATURE_NAMES
    }
    return table


def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    extra_csv_paths: list = None,
    raw_cache: Optional[dict] = None,
) -> Tuple[Tuple, Tuple, dict]:
    """Fetch all tickers, combine, compute cross-sectional ranks, and time-split 80/20.

    Returns:
        ((X_train, y_train), (X_test, y_test), quantile_table)
    """
    tickers = load_tickers(csv_path)
    if extra_csv_paths:
        for extra in extra_csv_paths:
            tickers += load_tickers(extra)
        tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    frames = []
    used_tickers = []
    skipped = 0
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] Processing tickers...", flush=True)
        cached = raw_cache.get(ticker) if raw_cache else None
        df = build_ticker_dataset(ticker, horizon=horizon, up_thresh=up_thresh, raw_df=cached)
        if df is not None:
            frames.append(df)
            used_tickers.append(ticker)
        else:
            skipped += 1

    print(f"\nUsed {len(frames)} tickers, skipped {skipped}")
    combined = pd.concat(frames).sort_index()

    print("\nClass distribution (full dataset):")
    print(combined['target'].value_counts().sort_index().to_string())

    print("Computing cross-sectional rank features...", flush=True)
    date_key = combined.index.normalize()
    for feat in FEATURE_NAMES:
        combined[f'{feat}_rank'] = (
            combined.groupby([date_key, combined['exchange']])[feat].rank(pct=True)
        )

    split = int(len(combined) * 0.8)
    train_slice = combined.iloc[:split]
    quantile_table = _build_quantile_table(train_slice)

    X = combined[ALL_FEATURE_NAMES]
    y = combined['target']
    return (X.iloc[:split], y.iloc[:split]), (X.iloc[split:], y.iloc[split:]), quantile_table, used_tickers


def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build the unfitted sklearn Pipeline.

    RobustScaler is resistant to HK small-cap outliers.
    scale_pos_weight = count(NOT-UP) / count(UP) balances the binary imbalance.
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LGBMClassifier(
            n_estimators=1000,
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
    optuna_tune: bool = False,
    save_tickers: bool = True,
    raw_cache: Optional[dict] = None,
) -> None:
    """Full training run: load data, fit pipeline, evaluate, save model."""
    print(f"\n=== AlphaPulse Model Training — {horizon}-day horizon ===")
    print(f"Ticker source : {csv_path}")
    print(f"UP threshold  : > {up_thresh*100:.0f}%  (binary: NOT-UP otherwise)")
    print(f"Clip range    : ±{clip*100:.0f}%")
    print(f"Output path   : {output_path}\n")

    (X_train, y_train), (X_test, y_test), quantile_table, used_tickers = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        extra_csv_paths=extra_csv_paths,
        raw_cache=raw_cache,
    )

    print(f"\nTraining rows : {len(X_train):,}")
    print(f"Test rows     : {len(X_test):,}")
    print("\nClass distribution (train set):")
    print(y_train.value_counts().sort_index().to_string())

    spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"\nscale_pos_weight : {spw:.2f}  (NOT-UP / UP ratio)")

    if optuna_tune:
        import optuna
        from sklearn.model_selection import train_test_split
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("Running Optuna hyperparameter search (50 trials)...")
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=0.55, stratify=y_train, random_state=42
        )
        X_tv, X_val, y_tv, y_val = train_test_split(
            X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42
        )

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
            _sc = RobustScaler()
            X_tv_sc = _sc.fit_transform(X_tv)
            X_val_sc = _sc.transform(X_val)
            _clf = LGBMClassifier(
                scale_pos_weight=spw, n_jobs=2, random_state=42, verbose=-1, **params,
            )
            _clf.fit(
                X_tv_sc, y_tv,
                eval_set=[(X_val_sc, y_val)],
                callbacks=[lgb_early_stopping(50, verbose=False)],
            )
            y_proba_val = _clf.predict_proba(X_val_sc)[:, 1]
            return average_precision_score(y_val, y_proba_val)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        print(f"Best Optuna params : {study.best_params}")
        print(f"Best AUC-PR        : {study.best_value:.4f}")
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', LGBMClassifier(
                scale_pos_weight=spw, n_jobs=2, random_state=42, verbose=-1,
                **study.best_params,
            ))
        ])
    else:
        pipeline = build_pipeline(scale_pos_weight=spw)

    print("Fitting pipeline with early stopping — this may take a few minutes...")
    _val_n = max(int(len(X_train) * 0.15), 500)
    X_tr_es = X_train.iloc[:-_val_n]
    X_val_es = X_train.iloc[-_val_n:]
    y_tr_es = y_train.iloc[:-_val_n]
    y_val_es = y_train.iloc[-_val_n:]
    _scaler = pipeline.named_steps['scaler']
    _clf = pipeline.named_steps['clf']
    X_tr_sc = _scaler.fit_transform(X_tr_es)
    X_val_sc = _scaler.transform(X_val_es)
    _clf.fit(
        X_tr_sc, y_tr_es,
        eval_set=[(X_val_sc, y_val_es)],
        callbacks=[lgb_early_stopping(50, verbose=False), lgb_log_evaluation(100)],
    )
    print(f"Best iteration: {_clf.best_iteration_}  (max: {_clf.get_params()['n_estimators']})")
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
        'features': ALL_FEATURE_NAMES,
        'quantile_table': quantile_table,
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

    if save_tickers:
        tickers_csv = output_path.replace('.joblib', '_tickers.csv')
        pd.Series(used_tickers, name='ticker').to_csv(tickers_csv, index=False)
        print(f"Valid tickers  : {tickers_csv}  ({len(used_tickers)} tickers)")


if __name__ == '__main__':
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _all_csvs = sorted(
        os.path.join(_data_dir, f) for f in os.listdir(_data_dir) if f.endswith('.csv')
    )
    if not _all_csvs:
        raise FileNotFoundError(f"No CSV files found in {_data_dir}")
    print(f"Ticker CSVs found: {[os.path.basename(p) for p in _all_csvs]}")

    # Pre-fetch raw data once; reuse across all horizon runs to avoid duplicate downloads.
    _tickers = load_tickers(_all_csvs[0])
    for _extra in (_all_csvs[1:] or []):
        _tickers += load_tickers(_extra)
    _tickers = list(dict.fromkeys(_tickers))
    print(f"\nPre-fetching raw data for {len(_tickers)} tickers...")
    _raw_cache: dict = {}
    for _i, _t in enumerate(_tickers):
        if _i % 100 == 0:
            print(f"  [{_i}/{len(_tickers)}] Fetching...", flush=True)
        _raw = fetch_latest_data(_t)
        if _raw is not None:
            _raw_cache[_t] = calculate_technical_indicators(_raw)
    print(f"Cached {len(_raw_cache)} tickers.\n")

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_save(
        _all_csvs[0],
        horizon=5,
        up_thresh=0.03,
        clip=0.20,
        output_path=os.path.join(_base_dir, 'stock_model_5d.joblib'),
        extra_csv_paths=_all_csvs[1:] or None,
        raw_cache=_raw_cache,
        optuna_tune=True,
    )
    train_and_save(
        _all_csvs[0],
        horizon=14,
        up_thresh=0.03,
        clip=0.30,
        output_path=os.path.join(_base_dir, 'stock_model_14d.joblib'),
        extra_csv_paths=_all_csvs[1:] or None,
        raw_cache=_raw_cache,
        optuna_tune=True,
        save_tickers=False,
    )
