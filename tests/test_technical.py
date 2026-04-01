import pandas as pd
import numpy as np
import pytest


def _make_ohlcv(n=120):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - np.random.uniform(0, 1, n),
        "high": close + np.random.uniform(0, 2, n),
        "low": close - np.random.uniform(0, 2, n),
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })


def test_add_indicators_returns_ma_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["ma5", "ma10", "ma20", "ma60"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_macd_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["macd_dif", "macd_dea", "macd_hist"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_rsi_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["rsi6", "rsi12", "rsi24"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_bollinger_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["bb_upper", "bb_mid", "bb_lower", "bb_width"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_volume_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["vol_ratio", "obv", "atr"]:
        assert col in result.columns, f"Missing {col}"


def test_feature_cols_all_present_after_indicators():
    from features.technical import add_indicators, FEATURE_COLS
    df = _make_ohlcv()
    result = add_indicators(df)
    missing = [c for c in FEATURE_COLS if c not in result.columns]
    assert not missing, f"FEATURE_COLS missing from output: {missing}"
