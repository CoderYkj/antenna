"""tests/test_feature_learner.py — feature_learner + get_active_feature_cols 单测。"""
import json
from pathlib import Path

import pytest


# ── get_active_feature_cols ─────────────────────────────────

def test_get_active_feature_cols_no_file(tmp_path, monkeypatch):
    """文件缺失时回退完整 FEATURE_COLS。"""
    monkeypatch.chdir(tmp_path)
    from features.technical import get_active_feature_cols, FEATURE_COLS
    assert get_active_feature_cols() == FEATURE_COLS


def test_get_active_feature_cols_reads_active(tmp_path, monkeypatch):
    """feature_weights.json 存在时返回 active 列表。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"active": ["ma5", "rsi6"]}), encoding="utf-8"
    )
    from importlib import reload
    import features.technical as t
    reload(t)
    assert t.get_active_feature_cols() == ["ma5", "rsi6"]


def test_get_active_feature_cols_empty_active_fallback(tmp_path, monkeypatch):
    """active 列表为空时回退 FEATURE_COLS。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"active": []}), encoding="utf-8"
    )
    from importlib import reload
    import features.technical as t
    reload(t)
    from features.technical import FEATURE_COLS
    assert t.get_active_feature_cols() == FEATURE_COLS


def test_predictor_fills_missing_feature_with_zero(tmp_path, monkeypatch):
    """predict() 对 df 中不存在的特征自动补 0，而非抛 KeyError。"""
    import pandas as pd
    import numpy as np
    monkeypatch.chdir(tmp_path)

    class _FakeModel:
        def predict(self, X):
            return np.array([0.5] * len(X))

    df = pd.DataFrame({"close": [10.0], "ma5": [9.8]})
    from models.predictor import predict
    # ma10 等大量列缺失，predict 应补 0 而不抛异常
    result = predict(df, feature_cols=["ma5", "ma10"], model=_FakeModel(), buy_top_pct=None)
    assert "rise_prob" in result


def test_build_features_appends_alt_to_last_row():
    """build_features(df, alt={...}) 把 alt 值追加到 df 最后一行。"""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "open": [10.0] * 5, "high": [11.0] * 5,
        "low":  [9.0]  * 5, "close": [10.5] * 5,
        "volume": [1e6] * 5,
    })
    from features.builder import build_features
    result = build_features(df, alt={"main_net_in_1d": 0.3, "sector_heat_rank": -0.5})
    assert result["main_net_in_1d"].iloc[-1] == pytest.approx(0.3)
    assert result["sector_heat_rank"].iloc[-1] == pytest.approx(-0.5)
    # 非最后行保持 NaN
    assert pd.isna(result["main_net_in_1d"].iloc[0])


def test_build_features_no_alt_unchanged():
    """不传 alt 时行为与原版一致。"""
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "open": [10.0] * 5, "high": [11.0] * 5,
        "low":  [9.0]  * 5, "close": [10.5] * 5,
        "volume": [1e6] * 5,
    })
    from features.builder import build_features
    result = build_features(df)
    assert "main_net_in_1d" not in result.columns
