import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


def _make_feature_df(n=100):
    np.random.seed(2)
    from features.technical import FEATURE_COLS, add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


class _MockModel:
    def __init__(self, prob):
        self._prob = prob

    def predict(self, X):
        return np.full(len(X), self._prob)


def test_predict_returns_required_keys():
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.7)):
        result = predict(df, FEATURE_COLS)
    for key in ["rise_prob", "fall_prob", "confidence", "signal"]:
        assert key in result


def test_predict_probs_sum_to_one():
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.65)):
        result = predict(df, FEATURE_COLS)
    assert abs(result["rise_prob"] + result["fall_prob"] - 1.0) < 1e-6


def test_predict_signal_buy_when_high_prob(monkeypatch):
    """旧 P0 语义:绝对阈值切信号(buy_top_pct=None 模式)。
    隔离 P1 calibrator 影响,确保测试只验证 prob_raw → signal 的旧逻辑。"""
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    monkeypatch.setattr("models.predictor._apply_calibration",
                        lambda p, **_: (float(p), 0.0))  # 恒等映射 + abs_threshold=0
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.75)):
        result = predict(df, FEATURE_COLS)
    assert result["signal"] == "买入"
    assert result["confidence"] == "高"


def test_predict_signal_avoid_when_low_prob(monkeypatch):
    """旧 P0 语义:绝对阈值低分回避。"""
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    monkeypatch.setattr("models.predictor._apply_calibration",
                        lambda p, **_: (float(p), 0.0))
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.25)):
        result = predict(df, FEATURE_COLS)
    assert result["signal"] == "回避"
