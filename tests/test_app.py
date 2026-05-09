import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_mock_model(prediction: int = 1):
    """Return a mock Pipeline whose predict/predict_proba mimic binary output."""
    m = MagicMock()
    m.predict.return_value = np.array([prediction])
    m.predict_proba.return_value = np.array([[0.38, 0.62]])
    m.classes_ = np.array([0.0, 1.0])
    return m


def _make_mock_data():
    import pandas as pd
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    return pd.DataFrame(
        {"Adj_Close": [57.35] * 30, "Adj_Volume": [500_000.0] * 30},
        index=idx,
    )


class TestBuildPredictionResponse:
    def test_confidence_up_3pct_is_class_1(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        # proba: [0]=0.38, [1]=0.62 → confidence_up_3pct = P(1) = 0.62
        assert abs(result["confidence_up_3pct"] - 0.62) < 1e-4

    def test_signal_is_up_for_prediction_one(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert result["signal"] == "UP >3%"

    def test_signal_is_no_signal_for_prediction_zero(self):
        from app import build_prediction_response
        m = MagicMock()
        m.predict.return_value = np.array([0])
        m.predict_proba.return_value = np.array([[0.75, 0.25]])
        m.classes_ = np.array([0.0, 1.0])
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", m, features, mock_data)

        assert result["signal"] == "NO SIGNAL"

    def test_signal_map_is_binary(self):
        from app import SIGNAL_MAP
        assert SIGNAL_MAP[1] == "UP >3%"
        assert SIGNAL_MAP[0] == "NO SIGNAL"
        assert len(SIGNAL_MAP) == 2

    def test_response_contains_required_fields(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        for key in ("ticker", "prediction", "signal",
                    "confidence_up_3pct", "edge_ratio",
                    "probabilities", "current_price", "last_updated"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_down_not_in_response(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert "confidence_down_3pct" not in result

    def test_probabilities_keyed_by_binary_classes(self):
        from app import build_prediction_response
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        assert set(result["probabilities"].keys()) == {"0", "1"}

    def test_edge_ratio_is_up_over_threshold(self):
        from app import build_prediction_response
        mock_model = _make_mock_model(prediction=1)
        mock_data = _make_mock_data()
        features = np.array([[1.0, 1.0, 50.0, 0.01, 0.005, 0.01]])

        result = build_prediction_response("0001.HK", mock_model, features, mock_data)

        # P(UP) = 0.62, UP_THRESHOLD = 0.5 → edge_ratio = 0.62 / 0.5 = 1.24
        assert abs(result["edge_ratio"] - round(0.62 / 0.5, 2)) < 1e-4


class TestRankScanResults:
    def _make_result(self, ticker, edge_ratio):
        return {"ticker": ticker, "edge_ratio": edge_ratio, "signal": "STABLE", "confidence_up_3pct": 0.5}

    def test_returns_top_n_by_edge_ratio_descending(self):
        from app import rank_scan_results
        results = [
            self._make_result("A.HK", 1.2),
            self._make_result("B.HK", 4.5),
            self._make_result("C.HK", 0.8),
            self._make_result("D.HK", 3.1),
            self._make_result("E.HK", 6.0),
            self._make_result("F.HK", 2.3),
        ]
        top = rank_scan_results(results, n=3)
        assert [r["ticker"] for r in top] == ["E.HK", "B.HK", "D.HK"]

    def test_returns_all_when_fewer_than_n(self):
        from app import rank_scan_results
        results = [
            self._make_result("A.HK", 2.0),
            self._make_result("B.HK", 5.0),
        ]
        top = rank_scan_results(results, n=5)
        assert len(top) == 2

    def test_returns_empty_list_for_empty_input(self):
        from app import rank_scan_results
        assert rank_scan_results([], n=5) == []
