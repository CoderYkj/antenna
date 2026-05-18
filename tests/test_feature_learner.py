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
