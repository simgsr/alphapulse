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
    LGBMRanker,
    early_stopping as lgb_early_stopping,
    log_evaluation as lgb_log_evaluation,
)
from get_price_data import fetch_latest_data, calculate_technical_indicators, fetch_regime_data

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
REGIME_FEATURE_NAMES = ['mkt_ret_20d', 'mkt_sma200_ratio', 'vix_level', 'vix_chg_5d']
ALL_FEATURE_NAMES = FEATURE_NAMES + RANK_FEATURE_NAMES + REGIME_FEATURE_NAMES


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
    min_price: float = 1.0,
    min_avg_volume: int = 500_000,
) -> Optional[pd.DataFrame]:
    """Fetch, engineer features, and label one ticker.

    Returns None if data is unavailable, fails price/volume filters, or fewer
    than 252 labeled rows remain.
    Pass raw_df to skip the fetch+indicator step (reuse cached data).
    """
    if raw_df is None:
        raw = fetch_latest_data(ticker, period=period)
        if raw is None:
            return None
        raw_df = calculate_technical_indicators(raw)
    df = raw_df.copy()
    if df['Adj_Close'].median() < min_price:
        return None
    if df['Adj_Volume'].median() < min_avg_volume:
        return None
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


def _add_regime_features(combined: pd.DataFrame, regime_data: dict) -> pd.DataFrame:
    """Merge market regime features into the combined stock DataFrame by date + exchange."""
    date_key = combined.index.normalize()
    for feat in REGIME_FEATURE_NAMES:
        combined[feat] = np.nan
    for exch in combined['exchange'].unique():
        df_reg = regime_data.get(exch)
        if df_reg is None:
            df_reg = regime_data.get('ALL')
        if df_reg is None:
            continue
        mask = combined['exchange'] == exch
        aligned = df_reg[REGIME_FEATURE_NAMES].reindex(date_key[mask], method='ffill')
        for feat in REGIME_FEATURE_NAMES:
            combined.loc[mask, feat] = aligned[feat].values
    # Forward-fill any remaining gaps (exchange holidays, missing market data)
    combined[REGIME_FEATURE_NAMES] = combined[REGIME_FEATURE_NAMES].ffill()
    return combined


def _group_sizes_by_date(index: pd.DatetimeIndex) -> np.ndarray:
    """Count of samples per trading date, in sorted order, for LambdaRank group param."""
    _, counts = np.unique(index.normalize(), return_counts=True)
    return counts


def _precision_at_k(
    y_true: pd.Series, scores: np.ndarray, date_index: pd.DatetimeIndex, k: int = 10
) -> float:
    """Mean precision@K across all dates — the natural trading evaluation metric."""
    dates = date_index.normalize()
    results = []
    for d in np.unique(dates):
        mask = dates == d
        y_d = y_true.values[mask]
        s_d = scores[mask]
        if y_d.sum() == 0 or len(y_d) < k:
            continue
        top_idx = np.argsort(s_d)[-k:]
        results.append(float(y_d[top_idx].mean()))
    return float(np.mean(results)) if results else 0.0


def build_full_dataset(
    csv_path: str,
    horizon: int = 7,
    up_thresh: float = 0.03,
    extra_csv_paths: list = None,
    raw_cache: Optional[dict] = None,
    min_price: float = 1.0,
    min_avg_volume: int = 500_000,
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
        df = build_ticker_dataset(
            ticker, horizon=horizon, up_thresh=up_thresh, raw_df=cached,
            min_price=min_price, min_avg_volume=min_avg_volume,
        )
        if df is not None:
            frames.append(df)
            used_tickers.append(ticker)
        else:
            skipped += 1

    print(f"\nUsed {len(frames)} tickers, skipped {skipped}")
    combined = pd.concat(frames).sort_index()

    print("Fetching market regime data (SPY, ^VIX, ^HSI, ^STI)...", flush=True)
    regime_data = fetch_regime_data(period='5y')
    combined = _add_regime_features(combined, regime_data)
    before = len(combined)
    combined = combined.dropna(subset=REGIME_FEATURE_NAMES)
    dropped = before - len(combined)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing regime data (early history).")

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
    scale_pos_weight=n_neg/n_pos reweights the loss without resampling, which
    tends to spread the probability distribution wider than is_unbalance=True.
    Pass scale_pos_weight=1.0 (default) to disable class weighting.
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


def build_ranker() -> tuple:
    """Return (scaler, ranker) for the LambdaRank training path.

    LambdaRank optimises NDCG@5, directly targeting the top-K trading use case.
    Scale_pos_weight is irrelevant here — label_gain handles class imbalance.
    """
    scaler = RobustScaler()
    ranker = LGBMRanker(
        objective='lambdarank',
        lambdarank_truncation_level=5,
        n_estimators=1000,
        num_leaves=95,
        learning_rate=0.01,
        min_child_samples=20,
        n_jobs=2,
        random_state=42,
        verbose=-1,
    )
    return scaler, ranker


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
    use_ranking: bool = False,
    save_tickers: bool = True,
    raw_cache: Optional[dict] = None,
    min_price: float = 1.0,
    min_avg_volume: int = 500_000,
) -> None:
    """Full training run: load data, fit pipeline, evaluate, save model."""
    print(f"\n=== AlphaPulse Model Training — {horizon}-day horizon ===")
    print(f"Ticker source : {csv_path}")
    print(f"UP threshold  : > {up_thresh*100:.0f}%  (binary: NOT-UP otherwise)")
    print(f"Clip range    : ±{clip*100:.0f}%")
    print(f"Min price     : {min_price}")
    print(f"Min avg vol   : {min_avg_volume:,}")
    print(f"Output path   : {output_path}")
    print(f"Model type    : {'LambdaRank' if use_ranking else 'Classifier'}\n")

    (X_train, y_train), (X_test, y_test), quantile_table, used_tickers = build_full_dataset(
        csv_path,
        horizon=horizon,
        up_thresh=up_thresh,
        extra_csv_paths=extra_csv_paths,
        raw_cache=raw_cache,
        min_price=min_price,
        min_avg_volume=min_avg_volume,
    )

    print(f"\nTraining rows : {len(X_train):,}")
    print(f"Test rows     : {len(X_test):,}")
    print("\nClass distribution (train set):")
    print(y_train.value_counts().sort_index().to_string())

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"scale_pos_weight : {spw:.3f}  (n_neg={n_neg:,}  n_pos={n_pos:,})")

    _val_n = max(int(len(X_train) * 0.15), 500)
    X_tr_es = X_train.iloc[:-_val_n]
    X_val_es = X_train.iloc[-_val_n:]
    y_tr_es = y_train.iloc[:-_val_n]
    y_val_es = y_train.iloc[-_val_n:]

    if use_ranking:
        # ── LambdaRank path ────────────────────────────────────────────────

        # Diagnostic: group size distribution (stocks per trading date)
        _gs_tr_all = _group_sizes_by_date(X_tr_es.index)
        print(f"\nGroup size diagnostics (train, before filter):")
        print(f"  min={_gs_tr_all.min()}  median={int(np.median(_gs_tr_all))}  "
              f"max={_gs_tr_all.max()}  total_groups={len(_gs_tr_all)}")
        print(f"  Groups with <10 stocks: {((_gs_tr_all < 10).sum())} "
              f"({100*(_gs_tr_all < 10).mean():.1f}%)")

        # Filter out dates with fewer than 10 stocks — too noisy for NDCG
        _min_group = 10
        _date_key_tr = X_tr_es.index.normalize()
        _date_counts_tr = _date_key_tr.value_counts()
        _valid_dates_tr = _date_counts_tr[_date_counts_tr >= _min_group].index
        _mask_tr = _date_key_tr.isin(_valid_dates_tr)
        X_tr_es_f = X_tr_es[_mask_tr]
        y_tr_es_f = y_tr_es[_mask_tr]

        _date_key_val = X_val_es.index.normalize()
        _date_counts_val = _date_key_val.value_counts()
        _valid_dates_val = _date_counts_val[_date_counts_val >= _min_group].index
        _mask_val = _date_key_val.isin(_valid_dates_val)
        X_val_es_f = X_val_es[_mask_val]
        y_val_es_f = y_val_es[_mask_val]

        _gs_tr = _group_sizes_by_date(X_tr_es_f.index)
        _gs_val = _group_sizes_by_date(X_val_es_f.index)
        print(f"After filter  (>={_min_group} stocks/date): "
              f"train groups={len(_gs_tr)}, val groups={len(_gs_val)}")
        print(f"  train rows={len(X_tr_es_f):,}  val rows={len(X_val_es_f):,}\n")

        print("Fitting LambdaRank model with early stopping...")
        _scaler, _ranker = build_ranker()
        X_tr_sc = np.asarray(_scaler.fit_transform(X_tr_es_f))
        X_val_sc = np.asarray(_scaler.transform(X_val_es_f))
        _ranker.fit(
            X_tr_sc, y_tr_es_f,
            group=_gs_tr,
            eval_set=[(X_val_sc, y_val_es_f)],
            eval_group=[_gs_val],
            eval_metric='ndcg',
            callbacks=[lgb_early_stopping(150, verbose=False), lgb_log_evaluation(100)],
        )
        print(f"Best iteration: {_ranker.best_iteration_}  (max: {_ranker.get_params()['n_estimators']})")
        print("Fitting complete.")

        X_test_sc = np.asarray(_scaler.transform(X_test))
        scores = _ranker.predict(X_test_sc)

        print("\n=== Held-out Test Set Evaluation (LambdaRank) ===")
        base_rate = float(y_test.mean())
        print(f"UP base rate  : {base_rate:.3f}  (random precision@K baseline)")
        for k in [5, 10, 20, 50]:
            p_at_k = _precision_at_k(y_test, scores, X_test.index, k=k)
            print(f"Precision@{k:<3} : {p_at_k:.4f}  (lift: {p_at_k/base_rate:.2f}x)")

        model_data = {
            'model_type': 'ranker',
            'scaler': _scaler,
            'ranker': _ranker,
            'features': ALL_FEATURE_NAMES,
            'quantile_table': quantile_table,
            'horizon': horizon,
            'up_thresh': up_thresh,
            'description': (
                f'LambdaRank stock ranker. '
                f'Target: UP>{up_thresh*100:.0f}% in {horizon} trading days.'
            ),
        }

    else:
        # ── Classifier path ────────────────────────────────────────────────
        if optuna_tune:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            print("Running Optuna hyperparameter search (50 trials)...")
            _n_sub = int(len(X_train) * 0.55)
            X_sub = X_train.iloc[:_n_sub]
            y_sub = y_train.iloc[:_n_sub]
            _n_tv = int(len(X_sub) * 0.80)
            X_tv, X_val = X_sub.iloc[:_n_tv], X_sub.iloc[_n_tv:]
            y_tv, y_val = y_sub.iloc[:_n_tv], y_sub.iloc[_n_tv:]

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
                X_tv_sc = np.asarray(_sc.fit_transform(X_tv))
                X_val_sc = np.asarray(_sc.transform(X_val))
                _clf = LGBMClassifier(
                    scale_pos_weight=spw, n_jobs=2, random_state=42, verbose=-1, **params,
                )
                _clf.fit(
                    X_tv_sc, y_tv,
                    eval_set=[(X_val_sc, y_val)],
                    eval_metric='average_precision',
                    callbacks=[lgb_early_stopping(100, verbose=False)],
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
        _scaler = pipeline.named_steps['scaler']
        _clf = pipeline.named_steps['clf']
        X_tr_sc = np.asarray(_scaler.fit_transform(X_tr_es))
        X_val_sc = np.asarray(_scaler.transform(X_val_es))
        _clf.fit(
            X_tr_sc, y_tr_es,
            eval_set=[(X_val_sc, y_val_es)],
            eval_metric='average_precision',
            callbacks=[lgb_early_stopping(100, verbose=False), lgb_log_evaluation(100)],
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
        thresholds = np.arange(0.50, 0.81, 0.05)
        print(f"\n=== UP class Precision / Recall sweep ===")
        print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Signals':>10}")
        for t in thresholds:
            y_pred_t = (y_proba_up >= t).astype(int)
            n_sig = int(y_pred_t.sum())
            prec = precision_score(y_test, y_pred_t, zero_division=0)
            rec = recall_score(y_test, y_pred_t, zero_division=0)
            print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {n_sig:>10}")

        # Cross-sectional precision@K — same metric as ranker evaluation
        print(f"\n=== Cross-sectional Precision@K (classifier scores) ===")
        base_rate = float(y_test.mean())
        print(f"UP base rate  : {base_rate:.3f}  (random precision@K baseline)")
        for k in [5, 10, 20, 50]:
            p_at_k = _precision_at_k(y_test, y_proba_up, X_test.index, k=k)
            print(f"Precision@{k:<3} : {p_at_k:.4f}  (lift: {p_at_k/base_rate:.2f}x)")

        # Feature importances
        _clf_fitted = pipeline.named_steps['clf']
        _imp = pd.Series(_clf_fitted.feature_importances_, index=ALL_FEATURE_NAMES).sort_values(ascending=False)
        print(f"\n=== Top 20 Feature Importances ===")
        print(_imp.head(20).to_string())
        _rank_weight = _imp[RANK_FEATURE_NAMES].sum() / _imp.sum()
        print(f"\nRank-feature share of total importance : {_rank_weight:.1%}")
        _regime_weight = _imp[REGIME_FEATURE_NAMES].sum() / _imp.sum()
        print(f"Regime-feature share of total importance: {_regime_weight:.1%}")

        model_data = {
            'model_type': 'classifier',
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


def diagnose_saved_model(
    model_path: str,
    csv_path: str,
    n_tickers: int = 60,
    extra_csv_paths: list = None,
) -> None:
    """Load a saved model and run probability/feature diagnostics on a ticker sample.

    Avoids a full retrain — useful for quickly locating the operating threshold
    and checking whether features carry weight.
    """
    data = joblib.load(model_path)
    if data.get('model_type') == 'ranker':
        print("diagnose_saved_model does not support ranker models — use predict_upstock.py instead.")
        return
    pipeline = data['model']
    horizon = data.get('horizon', 7)
    up_thresh = data.get('up_thresh', 0.03)

    tickers = load_tickers(csv_path)
    if extra_csv_paths:
        for p in extra_csv_paths:
            tickers += load_tickers(p)
        tickers = list(dict.fromkeys(tickers))
    sample = tickers[:n_tickers]

    print(f"\n=== Diagnosing {model_path} — sampling {n_tickers} tickers ===")
    frames = []
    for i, ticker in enumerate(sample):
        if i % 20 == 0:
            print(f"  [{i}/{len(sample)}] Fetching...", flush=True)
        raw = fetch_latest_data(ticker, period='5y')
        if raw is None:
            continue
        df_raw = calculate_technical_indicators(raw)
        df = build_ticker_dataset(ticker, horizon=horizon, up_thresh=up_thresh, raw_df=df_raw)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No usable data found in sample.")
        return

    combined = pd.concat(frames).sort_index()
    date_key = combined.index.normalize()
    for feat in FEATURE_NAMES:
        combined[f'{feat}_rank'] = (
            combined.groupby([date_key, combined['exchange']])[feat].rank(pct=True)
        )

    X = combined[ALL_FEATURE_NAMES]
    y = combined['target']
    y_proba_up = pipeline.predict_proba(X)[:, list(pipeline.classes_).index(1)]

    # --- Probability distribution ---
    print(f"\n=== P(UP) Distribution  (rows={len(y):,}, UP base rate={y.mean():.3f}) ===")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:02d}: {np.percentile(y_proba_up, p):.4f}")

    # --- Extended threshold sweep ---
    print(f"\n=== Extended Threshold Sweep (step 0.02) ===")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Signals':>10}")
    for t in np.arange(0.30, 0.81, 0.02):
        mask = (y_proba_up >= t).astype(int)
        n_sig = int(mask.sum())
        prec = precision_score(y, mask, zero_division=0)
        rec = recall_score(y, mask, zero_division=0)
        print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {n_sig:>10}")

    # --- Feature importances ---
    clf = pipeline.named_steps['clf']
    imp = pd.Series(clf.feature_importances_, index=ALL_FEATURE_NAMES).sort_values(ascending=False)
    print(f"\n=== Top 20 Feature Importances ===")
    print(imp.head(20).to_string())
    rank_weight = imp[RANK_FEATURE_NAMES].sum() / imp.sum()
    print(f"\nRank-feature share of total importance: {rank_weight:.1%}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == '--diagnose':
        # Usage: python train_model.py --diagnose <model.joblib> [n_tickers]
        _model_path = sys.argv[2] if len(sys.argv) > 2 else 'stock_model_5d.joblib'
        _n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        _all_csvs = sorted(
            os.path.join(_data_dir, f) for f in os.listdir(_data_dir) if f.endswith('.csv')
        )
        diagnose_saved_model(_model_path, _all_csvs[0], n_tickers=_n,
                             extra_csv_paths=_all_csvs[1:] or None)
        sys.exit(0)

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
        _raw = fetch_latest_data(_t, period='5y')
        if _raw is not None:
            _raw_cache[_t] = calculate_technical_indicators(_raw)
    print(f"Cached {len(_raw_cache)} tickers.\n")

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_save(
        _all_csvs[0],
        horizon=7,
        up_thresh=0.03,
        clip=0.20,
        output_path=os.path.join(_base_dir, 'stock_model_7d.joblib'),
        extra_csv_paths=_all_csvs[1:] or None,
        raw_cache=_raw_cache,
        use_ranking=False,
    )
    train_and_save(
        _all_csvs[0],
        horizon=14,
        up_thresh=0.03,
        clip=0.30,
        output_path=os.path.join(_base_dir, 'stock_model_14d.joblib'),
        extra_csv_paths=_all_csvs[1:] or None,
        raw_cache=_raw_cache,
        use_ranking=False,
        save_tickers=False,
    )
