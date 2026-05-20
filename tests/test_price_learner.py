"""tests/test_price_learner.py — price_learner 单元测试。"""
import json
from pathlib import Path

import pandas as pd
import pytest


_YAML_CFG = (
    "lookback_days: 90\nmin_samples: 30\n"
    "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
    "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
    "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
    "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
    "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
    "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n"
)


@pytest.fixture()
def price_learner_yaml(tmp_path):
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "price_learner.yaml").write_text(_YAML_CFG, encoding="utf-8")
    return tmp_path


# ── load_price_params ────────────────────────────────────────

def test_load_price_params_missing_file_returns_defaults(price_learner_yaml, monkeypatch):
    """price_params.json 缺失时返回 yaml 中的默认值，不抛异常。"""
    monkeypatch.chdir(price_learner_yaml)
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("range")
    assert params["short_atr_mult"] == pytest.approx(1.5)
    assert params["long_amp_mult"] == pytest.approx(3.0)


def test_load_price_params_reads_json(price_learner_yaml, monkeypatch):
    """price_params.json 存在时读取对应 state 的参数。"""
    monkeypatch.chdir(price_learner_yaml)
    (price_learner_yaml / "learning" / "price_params.json").write_text(
        json.dumps({"bull": {"short_atr_mult": 1.8, "short_gain_mult": 2.5,
                             "long_amp_mult": 3.2, "long_ma60_buffer": 0.98}}),
        encoding="utf-8",
    )
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("bull")
    assert params["short_atr_mult"] == pytest.approx(1.8)


def test_load_price_params_corrupt_json_falls_back_to_defaults(price_learner_yaml, monkeypatch):
    """price_params.json 损坏时静默回退到默认值。"""
    monkeypatch.chdir(price_learner_yaml)
    (price_learner_yaml / "learning" / "price_params.json").write_text("{broken", encoding="utf-8")
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("bull")
    assert "short_atr_mult" in params


# ── _score_params ─────────────────────────────────────────────

def test_score_params_win_triggers():
    """hit_5d >= sell threshold 计为 win，score > 0。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.06,
        "max_drawdown_5d": -0.01,
        "atr_pct": 0.015,
        "up_ratio": 0.02,
    }])
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score > 0.0


def test_score_params_loss_triggers():
    """max_drawdown_5d <= -stop threshold 计为 loss，score < 0。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.01,
        "max_drawdown_5d": -0.04,
        "atr_pct": 0.015,
        "up_ratio": 0.02,
    }])
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score < 0.0


def test_score_params_boundary_win():
    """hit_5d 恰好等于 sell threshold 时计为 win (>= 语义)。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.04,          # exactly up_ratio * short_gain_mult = 0.02 * 2.0
        "max_drawdown_5d": -0.01,
        "atr_pct": 0.015,
        "up_ratio": 0.02,
    }])
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score > 0.0


# ── fit_price_params cold start ──────────────────────────────

def test_fit_price_params_cold_start_uses_defaults(price_learner_yaml, monkeypatch):
    """样本 < min_samples 时各桶都回退默认值，status == 'cold_start'。"""
    monkeypatch.chdir(price_learner_yaml)
    (price_learner_yaml / "learning" / "data").mkdir(parents=True, exist_ok=True)
    from unittest.mock import patch
    from importlib import reload
    import learning.price_learner as m; reload(m)
    with patch("learning.tracker.load_predictions_by_scene", return_value=[]):
        cfg = m.load_config()
        result = m.fit_price_params("2026-05-17", cfg)
    assert result["status"] == "cold_start"
    assert set(result["cold_states"]) == {"bull", "bear", "range"}


# ── weekly skip ─────────────────────────────────────────────

def test_run_skips_on_non_sunday():
    from learning.price_learner import run
    result = run("2026-05-19")  # Tuesday
    assert result["status"] == "skipped"
    assert "weekly only" in result.get("reason", "")


# ── load_config ──────────────────────────────────────────────

def test_load_config_missing_raises_value_error(tmp_path, monkeypatch):
    """price_learner.yaml 缺失时 load_config() 抛 ValueError。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    from importlib import reload
    import learning.price_learner as m; reload(m)
    with pytest.raises(ValueError, match="price_learner.yaml"):
        m.load_config()
