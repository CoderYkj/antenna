import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def _make_training_df(n=500):
    np.random.seed(1)
    from features.technical import FEATURE_COLS, add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


def test_build_labels_correct_shape():
    from models.trainer import build_labels
    df = _make_training_df()
    labels = build_labels(df, target_days=5, threshold=0.02)
    assert len(labels) == len(df)
    assert labels.isin([0, 1]).all()


def test_train_returns_booster(tmp_path):
    import lightgbm as lgb
    from models.trainer import train, build_labels
    from features.technical import FEATURE_COLS
    df = _make_training_df(n=500)
    df["code"] = "600519"
    # label 必须在合并前按股票单独计算，否则 shift 会跨越股票边界
    df["label"] = build_labels(df, target_days=5, threshold=0.02)
    model = train(df, feature_cols=FEATURE_COLS)
    assert hasattr(model, "predict")


def test_save_and_load_model(tmp_path):
    import lightgbm as lgb
    from models.trainer import train, build_labels, save_model, load_latest_model
    from features.technical import FEATURE_COLS
    df = _make_training_df(n=500)
    df["code"] = "600519"
    df["label"] = build_labels(df, target_days=5, threshold=0.02)
    model = train(df, feature_cols=FEATURE_COLS)
    path = save_model(model, saved_dir=str(tmp_path))
    loaded = load_latest_model(saved_dir=str(tmp_path))
    assert hasattr(loaded, "predict")
