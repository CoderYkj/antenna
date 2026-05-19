"""tests/test_price_learner.py — price_learner 单元测试。"""
import json
from pathlib import Path

import pandas as pd
import pytest


# ── load_price_params ────────────────────────────────────────

def test_load_price_params_missing_file_returns_defaults(tmp_path, monkeypatch):
    """price_params.json 缺失时返回 yaml 中的默认值，不抛异常。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    cfg_path = tmp_path / "learning" / "price_learner.yaml"
    cfg_path.write_text(
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n",
        encoding="utf-8",
    )
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("range")
    assert params["short_atr_mult"] == pytest.approx(1.5)
    assert params["long_amp_mult"] == pytest.approx(3.0)


def test_load_price_params_reads_json(tmp_path, monkeypatch):
    """price_params.json 存在时读取对应 state 的参数。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "price_learner.yaml").write_text(
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n",
        encoding="utf-8",
    )
    (tmp_path / "learning" / "price_params.json").write_text(
        json.dumps({"bull": {"short_atr_mult": 1.8, "short_gain_mult": 2.5,
                             "long_amp_mult": 3.2, "long_ma60_buffer": 0.98}}),
        encoding="utf-8",
    )
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("bull")
    assert params["short_atr_mult"] == pytest.approx(1.8)


# ── _score_params ─────────────────────────────────────────────

def test_score_params_win_triggers():
    """hit_5d >= sell threshold 计为 win，score > 0。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.06,          # 6% return
        "max_drawdown_5d": -0.01,
        "atr_pct": 0.015,
        "up_ratio": 0.02,        # 2% daily move estimate
    }])
    # short_gain_mult=2.0 → sell_thresh = 0.02 * 2.0 = 4% → win (hit_5d=6%)
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score > 0.0


def test_score_params_loss_triggers():
    """max_drawdown_5d <= -stop threshold 计为 loss，score < 0。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.01,
        "max_drawdown_5d": -0.04,  # -4% drawdown
        "atr_pct": 0.015,
        "up_ratio": 0.02,
    }])
    # short_atr_mult=1.5 → stop_thresh = 1.5 * 0.015 = 2.25% → loss (-4% < -2.25%)
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score < 0.0


# ── fit_price_params cold start ──────────────────────────────

def test_fit_price_params_cold_start_uses_defaults(tmp_path, monkeypatch):
    """样本 < min_samples 时各桶都回退默认值。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning" / "data").mkdir(parents=True)
    cfg_yaml = (
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n"
    )
    (tmp_path / "learning" / "price_learner.yaml").write_text(cfg_yaml, encoding="utf-8")
    from importlib import reload
    import learning.price_learner as m; reload(m)
    cfg = m.load_config()
    result = m.fit_price_params("2026-05-17", cfg)
    assert result["status"] in ("cold_start", "ok")


# ── weekly skip ─────────────────────────────────────────────

def test_run_skips_on_non_sunday(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from learning.price_learner import run
    result = run("2026-05-18")  # Monday
    assert result["status"] == "skipped"
