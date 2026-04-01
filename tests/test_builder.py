import pandas as pd
import numpy as np


def _make_ohlcv(n=120):
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })


def test_build_features_has_no_leading_nan_rows():
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    df = _make_ohlcv(120)
    result = build_features(df)
    for col in FEATURE_COLS:
        assert col in result.columns
    # 最后几行不应全是 NaN
    last = result.tail(10)[FEATURE_COLS]
    assert not last.isnull().all(axis=None)


def test_build_features_preserves_ohlcv():
    from features.builder import build_features
    df = _make_ohlcv(120)
    result = build_features(df)
    for col in ["date", "open", "high", "low", "close", "volume"]:
        assert col in result.columns
