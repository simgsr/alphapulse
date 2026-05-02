import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, log_loss
from get_price_data import fetch_latest_data, calculate_technical_indicators

FEATURE_NAMES = [
    'SMA_5_ratio', 'SMA_20_ratio', 'RSI_14',
    'Volatility_20', 'Returns_1d', 'Returns_5d'
]


def discretize_return(r: float) -> int:
    """Map a 7-day forward return to a 5-class label.

    Classes:
        2  — UP > 5%
        1  — UP 3–5%
        0  — STABLE (within ±3%)
       -1  — DOWN 3–5%
       -2  — DOWN > 5%
    """
    if r > 0.05:
        return 2
    elif r > 0.03:
        return 1
    elif r >= -0.03:
        return 0
    elif r >= -0.05:
        return -1
    else:
        return -2
