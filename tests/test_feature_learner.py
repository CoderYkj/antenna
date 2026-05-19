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


# ── feature_learner config ──────────────────────────────────

def test_load_config_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    from learning.feature_learner import load_config
    with pytest.raises(ValueError, match="feature_learner.yaml"):
        load_config()


def test_load_config_reads_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    cfg_path = tmp_path / "learning" / "feature_learner.yaml"
    cfg_path.write_text(
        "permutation:\n  n_repeats: 5\n  lookback_days: 60\n"
        "drop:\n  importance_threshold: 0.05\n  candidate_weeks: 4\n  drop_weeks: 8\n"
        "alt_data:\n  ic_threshold: 0.02\n  ic_lookback_days: 90\n",
        encoding="utf-8",
    )
    from importlib import reload
    import learning.feature_learner as m; reload(m)
    cfg = m.load_config(cfg_path)
    assert cfg.n_repeats == 5
    assert cfg.candidate_weeks == 4


# ── drop tracker ────────────────────────────────────────────

def test_update_drop_tracker_normal_feature_stays_active():
    """重要性 >= threshold 的特征保持 active。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {}, "dropped": []}
    importances = {"ma5": 0.10, "cci": 0.08}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert "ma5" not in new["candidate_drop"]
    assert "ma5" not in new["dropped"]


def test_update_drop_tracker_accumulates_candidate():
    """低重要性特征连续累积 weeks_below 计数。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {"cci": {"weeks_below": 3}}, "dropped": []}
    importances = {"cci": 0.01}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert new["candidate_drop"]["cci"]["weeks_below"] == 4


def test_update_drop_tracker_promotes_to_dropped():
    """连续 drop_weeks 后移入 dropped。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {"cci": {"weeks_below": 7}}, "dropped": []}
    importances = {"cci": 0.01}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert "cci" in new["dropped"]
    assert "cci" not in new["candidate_drop"]


def test_update_drop_tracker_resets_on_recovery():
    """特征重要性恢复后，candidate_drop 计数清零。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {"cci": {"weeks_below": 3}}, "dropped": []}
    importances = {"cci": 0.10}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert "cci" not in new["candidate_drop"]


# ── weekly skip ─────────────────────────────────────────────

def test_run_skips_on_non_sunday(tmp_path, monkeypatch):
    """非周日调用 run() 返回 skipped。"""
    monkeypatch.chdir(tmp_path)
    from learning.feature_learner import run
    # 2026-05-18 is Monday
    result = run("2026-05-18")
    assert result["status"] == "skipped"


def test_run_cold_start_no_model(tmp_path, monkeypatch):
    """无模型时返回 cold_start 或 no_model（不抛异常）。2026-05-17 是周日。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "data").mkdir()
    cfg_path = tmp_path / "learning" / "feature_learner.yaml"
    cfg_path.write_text(
        "permutation:\n  n_repeats: 3\n  lookback_days: 90\n"
        "drop:\n  importance_threshold: 0.05\n  candidate_weeks: 4\n  drop_weeks: 8\n"
        "alt_data:\n  ic_threshold: 0.02\n  ic_lookback_days: 90\n",
        encoding="utf-8",
    )
    (tmp_path / "models" / "saved").mkdir(parents=True)
    from importlib import reload
    import learning.feature_learner as m; reload(m)
    result = m.run("2026-05-17")  # Sunday
    assert result["status"] in ("cold_start", "no_model")
