import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_mock_model(prediction: int = 2):
    """Return a mock Pipeline whose predict/predict_proba mimic 5-class output."""
    m = MagicMock()
    m.predict.return_value = np.array([prediction])
    m.predict_proba.return_value = np.array([[0.05, 0.10, 0.23, 0.31, 0.31]])
    m.classes_ = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return m


def _make_mock_data():
    import pandas as pd
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    return pd.DataFrame(
        {"Adj_Close": [57.35] * 30, "Adj_Volume": [500_000.0] * 30},
        index=idx,
    )


class TestBuildPredictionResponse:
    def test_confidence_up_3pct_is_sum_of_class_1_and_2(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=2)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        # proba: [-2]=0.05, [-1]=0.10, [0]=0.23, [1]=0.31, [2]=0.31
        # confidence_up_3pct = P(1) + P(2) = 0.31 + 0.31 = 0.62
        assert abs(result["confidence_up_3pct"] - 0.62) < 1e-4

    def test_confidence_up_5pct_is_class_2_only(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=2)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert abs(result["confidence_up_5pct"] - 0.31) < 1e-4

    def test_confidence_5pct_never_exceeds_3pct(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert result["confidence_up_5pct"] <= result["confidence_up_3pct"]

    def test_signal_map_all_classes(self):
        from app import SIGNAL_MAP
        assert SIGNAL_MAP[2] == "UP > 5%"
        assert SIGNAL_MAP[1] == "UP 3-5%"
        assert SIGNAL_MAP[0] == "STABLE"
        assert SIGNAL_MAP[-1] == "DOWN 3-5%"
        assert SIGNAL_MAP[-2] == "DOWN > 5%"

    def test_response_contains_required_fields(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        for key in ("ticker", "prediction", "signal",
                    "confidence_up_3pct", "confidence_up_5pct",
                    "probabilities", "current_price", "last_updated"):
            assert key in result, f"Missing key: {key}"

    def test_probabilities_keyed_by_string_int(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert set(result["probabilities"].keys()) == {"-2", "-1", "0", "1", "2"}
