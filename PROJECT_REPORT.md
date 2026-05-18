# AlphaPulse — Project Report

**Project:** yf_price_prediction  
**Period:** Initial commit → May 2026  
**Status:** Concluded — no exploitable cross-sectional alpha found with current feature set

---

## 1. Project Goal

Build a machine learning model to predict which stocks will rise more than 3% within a short forward horizon (7 or 14 calendar days), across three exchanges: US equities, Hong Kong (HKEX), and Singapore (SGX). The intended output was a daily ranked list of top stock picks.

---

## 2. What Was Built

### 2.1 Data Pipeline (`get_price_data.py`)

- **Source:** Yahoo Finance via `yfinance`, 5-year history per ticker
- **Universe:** ~5,200 tickers across US, HK (`.HK`), and SGX (`.SI`)
- **Filters:** minimum price $1.00, minimum median daily volume 500,000
- **Market regime data:** SPY, ^VIX, ^HSI, ^STI fetched separately and merged by exchange and date

### 2.2 Feature Engineering

**21 technical indicators** (per-ticker, per-day):

| Category | Features |
|----------|----------|
| Trend | SMA_5_ratio, SMA_20_ratio, SMA_50_ratio |
| Momentum | RSI_14, RSI_7, MACD, MACD_hist, Stoch_K, Stoch_D |
| Volatility | Bollinger Band %B, Volatility_20, ATR_ratio, ADX_14 |
| Returns | Returns_1d, Returns_5d, Returns_10d, Returns_20d |
| Volume/Flow | Volume_ratio_20, OBV_ratio, CCI_20, CMF_20 |

**21 cross-sectional rank features** — each indicator ranked percentile within its exchange group per day (e.g. `RSI_14_rank`)

**4 market regime features** — applied per exchange:
- `mkt_ret_20d` — 20-day return of the local benchmark index
- `mkt_sma200_ratio` — benchmark vs its 200-day SMA (bull/bear regime)
- `vix_level` — CBOE VIX close
- `vix_chg_5d` — 5-day VIX change

**Total: 46 features**

### 2.3 Labels

- Binary: `1` if forward return > 3% in the horizon, `0` otherwise
- Clipped at ±20–30% to limit outlier influence
- Class imbalance handled via `scale_pos_weight = n_neg / n_pos`

### 2.4 Models Tried (chronological)

| Phase | Model | Notes |
|-------|-------|-------|
| v1 | Random Forest | Replaced — too slow, no probability calibration |
| v2 | LGBMClassifier, 5-class | DOWN>3%, DOWN<3%, STABLE, UP<3%, UP>3% |
| v3 | LGBMClassifier, 3-class | Simplified to DOWN / STABLE / UP |
| v4 | LGBMClassifier, binary | Replaced multiclass — cleaner loss, better calibration |
| v5 | LGBMClassifier + cross-sectional ranks | Added percentile ranks per day/exchange |
| v6 | LGBMClassifier + regime features | Added SPY, VIX, HSI, STI macro context |
| v7 | LGBMRanker (LambdaRank) | Optimised NDCG directly for top-K ranking |

### 2.5 Training Setup

- **Train/test split:** 80/20 time-based (no look-ahead leakage)
- **Early stopping:** held-out last 15% of training set as validation
- **Evaluation:** cross-sectional Precision@K (mean precision of top-K picks per day), threshold sweep, feature importances
- **Horizons trained:** 7-day and 14-day simultaneously, sharing a pre-fetched data cache

### 2.6 UI and Inference (`app.py`, `predict_upstock.py`)

- Gradio web interface for single-ticker analysis and universe-wide scan
- `predict_upstock.py` CLI loads both models, fetches live snapshots, and ranks by P(UP>3%) or ranker score

---

## 3. Results

### 3.1 Final Classifier Run (definitive sanity check)

| | 7-day model | 14-day model |
|--|-------------|--------------|
| Training rows | 4,093,582 | 4,063,804 |
| Test rows | 1,023,396 | 1,015,951 |
| UP base rate | 29.2% | 35.4% |
| Best iteration (of 1,000) | **6** | **17** |
| Precision@5 (cross-sectional) | 0.2905 — **1.00x** | 0.2995 — **0.85x** |
| Precision@10 | 0.2910 — **1.00x** | 0.3325 — **0.94x** |
| Precision@50 | 0.2920 — **1.00x** | 0.3312 — **0.94x** |
| Best threshold precision (absolute) | 0.00 (no signals) | 0.45 @ threshold 0.65 |

### 3.2 Feature Importance Breakdown

Both models converged on the same pattern:

| Feature group | 7-day share | 14-day share |
|---------------|-------------|--------------|
| Regime features (VIX, SPY, HSI/STI) | **81.4%** | **83.1%** |
| Technical indicators (RSI, SMA, MACD, etc.) | 13.5% | 13.2% |
| Cross-sectional rank features | **5.1%** | **3.7%** |

Top 4 features in both models:
1. `mkt_sma200_ratio` (SPY vs 200-day SMA)
2. `vix_level`
3. `mkt_ret_20d`
4. `vix_chg_5d`

### 3.3 LambdaRank Run

Switching to LGBMRanker (LambdaRank, NDCG objective) produced the same outcome: best iteration 3–125 depending on hyperparameters, Precision@K consistently at or below baseline.

---

## 4. Diagnosis — Why the Model Has No Alpha

### Finding 1: The model is a market timer, not a stock picker

83% of feature importance is in four macro/regime features that apply identically to all stocks on a given day. The model learned: *"when the market is in a bull regime (SPY above 200-SMA, VIX low), predict UP for most stocks."* This is not stock selection — it is market timing, and a mediocre one at that.

### Finding 2: Technical indicators do not carry cross-sectional alpha

Cross-sectional Precision@K at or below 1.00x means the model cannot distinguish which stocks within the same day's universe will outperform. RSI, MACD, Bollinger Bands, OBV, CCI, CMF, Stochastic — none added meaningful signal over the regime baseline. Cross-sectional rank versions of these indicators contributed only 3.7–5.1% of importance.

This is consistent with the academic and practitioner literature: standard technical indicators are widely known, easily computable, and largely priced in by the time a retail system can act on them.

### Finding 3: 7-day horizon is too noisy to learn from

Best iteration of 6 out of 1,000 means the gradient boosting trees found no stable signal to exploit at a 7-day horizon. The noise-to-signal ratio is simply too high for these features to overcome.

### Finding 4: 14-day absolute precision is a regime artefact, not alpha

At threshold 0.65, the 14-day model achieves precision of 0.45 (vs base rate 0.35). This sounds useful — but it fires only in sustained bull markets (when SPY is well above its 200-SMA and VIX is low). In those regimes, close to half of all stocks are already going up. The model is identifying market conditions, not individual stock opportunity.

---

## 5. What Would Be Needed to Build a Viable Model

The core problem is that the feature set describes *how the market is behaving*, not *why a specific stock should outperform its peers*. To generate genuine cross-sectional alpha, the following data categories are typically required:

| Data type | Signal | Source examples |
|-----------|--------|-----------------|
| **Earnings revisions** | Analysts raising estimates → price follows | IBES, Bloomberg |
| **Earnings surprise** | Beat vs consensus → drift in following days | Compustat, Refinitiv |
| **Fundamental ratios** | Value, quality, profitability factors | WRDS, Compustat |
| **Short interest** | High short → squeeze candidate; low short → cleaner trend | FINRA, S3 Partners |
| **News / sentiment** | Stock-specific positive/negative flow | NewsAPI, RavenPack, FinBERT |
| **Options flow** | Unusual call buying → informed buying signal | CBOE, unusual_whales |
| **Insider transactions** | Cluster buying by management | SEC Form 4, OpenInsider |
| **Analyst ratings changes** | Upgrades/downgrades | Bloomberg, TipRanks |

A longer horizon (30–60 days) would also reduce label noise enough for these signals to be learnable.

---

## 6. What Was Learned

### Technical

- **yfinance** is a viable free data source for prototyping but lacks fundamentals, short interest, and sentiment.
- **LightGBM** with time-based train/test split and early stopping is a sound evaluation framework — it gave us honest results.
- **LambdaRank / NDCG** is the correct objective for a top-K stock picking task, but it requires genuinely discriminative features to work.
- **Cross-sectional percentile ranks** are a good normalisation technique across markets with different price scales. They just didn't help here because the underlying indicators aren't predictive to begin with.
- **Regime features** are useful for a market timing overlay, but should not be the primary signal in a stock-selection model.
- **Early stopping iteration number is a diagnostic tool**: iteration 6 out of 1,000 means the model is learning almost nothing from the data.

### Process

- Iterating from a 5-class problem down to binary simplified training significantly and improved interpretability without losing evaluation power.
- Adding the cross-sectional precision@K metric was the right call — it exposed that the model was not doing stock selection at all.
- Running the classifier as a "sanity check" before continuing to tune the ranker saved time and gave a clear answer.

### Financial

- Standard retail technical analysis indicators (RSI, MACD, Bollinger Bands, etc.) do not provide exploitable edge in a machine learning framework at short horizons when applied cross-sectionally to a large multi-exchange universe.
- The signal that was found (regime timing) is real but not actionable for stock picking — it is equivalent to "buy the index in a bull market."

---

## 7. Conclusion

The project successfully built a complete end-to-end ML stock prediction pipeline: data ingestion, feature engineering, multi-exchange support, model training with proper time-series evaluation, cross-sectional ranking, and a live inference CLI. The infrastructure is sound.

However, the feature set does not contain cross-sectional stock-picking signal. The model's best learned behaviour is a market timing heuristic (buy when VIX is low and SPY is above its 200-day SMA), which is not the objective.

**Recommended next step if continuing:** Acquire at least one of the following before rebuilding — earnings revision data, short interest, or stock-specific news sentiment. These are the categories most likely to provide the discriminative cross-sectional signal that technical indicators cannot.

**Decision to close the project in its current form is well-founded.**

---

*Report generated: May 2026*
