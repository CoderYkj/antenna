"""tests/test_tactic_learner_cold_start.py — 冷启动 + 编排器入口集成。

覆盖:
  - evaluate_tactic 在无 pred/outcome 数据时返回 None
  - fit_tactic_params 在无任何历史样本时:全部 status='insufficient_samples',
    阈值维持 defaults(bear 用 override)
  - load_resonance_weights 文件缺失/损坏时回退 weight_min
  - run() 编排器入口在数据空白下仍能写出合法 JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_cfg(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "tactic_learner.yaml"
    yaml_path.write_text(
        """
defaults:
  value:
    roe_min:        {value:  8.0, direction: tighten_up}
    debt_ratio_max: {value: 50.0, direction: tighten_down}
  growth:
    rev_growth_min: {value: 15.0, direction: tighten_up}
    roe_min:        {value: 12.0, direction: tighten_up}
  leader:
    roe_min:        {value: 15.0, direction: tighten_up}
    gross_margin_min: {value: 30.0, direction: tighten_up}
  contra:
    drawdown_max:  {value: -0.15, direction: tighten_down}
    roe_min:       {value:  3.0,  direction: tighten_up}
defaults_bear_override:
  value:
    roe_min: 12.0
bounds:
  value:
    roe_min: [5.0, 15.0]
    debt_ratio_max: [40.0, 60.0]
  growth:
    rev_growth_min: [10.0, 25.0]
    roe_min: [8.0, 18.0]
  leader:
    roe_min: [12.0, 20.0]
    gross_margin_min: [25.0, 40.0]
  contra:
    drawdown_max: [-0.25, -0.10]
    roe_min: [2.0, 8.0]
step:
  default: 1.0
  drawdown: 0.01
evaluation:
  lookback_days: 90
  min_samples:   30
  acc_low:       0.30
  acc_high:      0.60
weights:
  enable_resonance_rank_boost: true
  rank_boost_per_tactic: 0.02
  rank_boost_max:        0.06
  weight_min:            0.05
  weight_max:            0.40
""",
        encoding="utf-8",
    )
    return yaml_path


# ── evaluate_tactic 数据空白 ─────────────────────────────

class TestEvaluateTacticEmpty:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        from learning import tactic_learner, tracker, market_state
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir()
        # market_state.load_state_on_date 总返回 range
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "range")

    def test_no_data_returns_none_acc(self, tmp_path):
        from learning.tactic_learner import evaluate_tactic
        result = evaluate_tactic("range", "value", "2026-04-29", lookback_days=90)
        assert result["samples"] == 0
        assert result["acc_90d"] is None


# ── fit_tactic_params 在数据空白时 ──────────────────────

class TestFitTacticParamsEmpty:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        from learning import tactic_learner, tracker, market_state, feedback_io
        monkeypatch.setattr(tactic_learner, "STATE_FILE", tmp_path / "tp.json")
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
        # market_state 全程 range,bear/bull 桶必然 0 样本
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "range")
        monkeypatch.setattr(market_state, "load_current_state",
                            lambda: {"current": "range"})
        self.tmp_path = tmp_path

    def test_all_buckets_insufficient_when_no_data(self, tmp_path):
        from learning.tactic_learner import fit_tactic_params, load_config
        cfg = load_config(_write_cfg(tmp_path))
        result = fit_tactic_params("2026-04-29", cfg)
        # 12 个桶全 insufficient_samples
        for state in ("bull", "bear", "range"):
            for tactic in ("value", "growth", "leader", "contra"):
                entry = result[state][tactic]
                assert entry["status"] == "insufficient_samples"
                assert entry["samples"] == 0

    def test_bear_uses_override_defaults(self, tmp_path):
        from learning.tactic_learner import fit_tactic_params, load_config
        cfg = load_config(_write_cfg(tmp_path))
        result = fit_tactic_params("2026-04-29", cfg)
        # bear 桶维持 defaults_bear_override 的值
        assert result["bear"]["value"]["roe_min"] == 12.0  # override
        # bull/range 维持 plain defaults
        assert result["bull"]["value"]["roe_min"] == 8.0
        assert result["range"]["value"]["roe_min"] == 8.0

    def test_persists_state_file(self, tmp_path):
        from learning.tactic_learner import fit_tactic_params, load_config
        from learning import tactic_learner
        cfg = load_config(_write_cfg(tmp_path))
        fit_tactic_params("2026-04-29", cfg)
        assert tactic_learner.STATE_FILE.exists()
        data = json.loads(tactic_learner.STATE_FILE.read_text(encoding="utf-8"))
        assert "params" in data
        assert "history" in data
        assert data["version"] == 1


# ── load_resonance_weights 兜底 ──────────────────────────

class TestLoadResonanceWeights:
    def test_missing_file_returns_weight_min(self, tmp_path, monkeypatch):
        from learning import tactic_learner
        monkeypatch.setattr(tactic_learner, "STATE_FILE", tmp_path / "nope.json")
        monkeypatch.setattr(tactic_learner, "CONFIG_PATH", _write_cfg(tmp_path))
        weights = tactic_learner.load_resonance_weights("range")
        for tactic in ("value", "growth", "leader", "contra"):
            assert weights[tactic] == 0.05

    def test_corrupt_file_returns_weight_min(self, tmp_path, monkeypatch):
        from learning import tactic_learner
        bad = tmp_path / "bad.json"
        bad.write_text("not-json{", encoding="utf-8")
        monkeypatch.setattr(tactic_learner, "STATE_FILE", bad)
        monkeypatch.setattr(tactic_learner, "CONFIG_PATH", _write_cfg(tmp_path))
        weights = tactic_learner.load_resonance_weights("range")
        assert all(v == 0.05 for v in weights.values())

    def test_valid_file_returns_persisted_weights(self, tmp_path, monkeypatch):
        from learning import tactic_learner
        f = tmp_path / "tp.json"
        f.write_text(json.dumps({
            "params": {
                "range": {
                    "value":  {"roe_min": 8.0, "weight": 0.30},
                    "growth": {"roe_min": 12.0, "weight": 0.25},
                    "leader": {"roe_min": 15.0, "weight": 0.25},
                    "contra": {"roe_min": 3.0, "weight": 0.20},
                }
            }
        }), encoding="utf-8")
        monkeypatch.setattr(tactic_learner, "STATE_FILE", f)
        monkeypatch.setattr(tactic_learner, "CONFIG_PATH", _write_cfg(tmp_path))
        weights = tactic_learner.load_resonance_weights("range")
        assert weights["value"]  == 0.30
        assert weights["growth"] == 0.25
        assert weights["contra"] == 0.20


# ── run() 编排器入口 ─────────────────────────────────────

class TestRunEntry:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        from learning import tactic_learner, tracker, market_state, feedback_io
        monkeypatch.setattr(tactic_learner, "STATE_FILE", tmp_path / "tp.json")
        monkeypatch.setattr(tactic_learner, "CONFIG_PATH", _write_cfg(tmp_path))
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "range")
        monkeypatch.setattr(market_state, "load_current_state",
                            lambda: {"current": "range"})

    def test_run_with_explicit_date(self, tmp_path):
        from learning import tactic_learner
        result = tactic_learner.run("2026-04-29")
        assert "range" in result
        assert "value" in result["range"]

    def test_run_no_date_uses_today(self, tmp_path):
        """无参调用应自动用今日,不抛。"""
        from learning import tactic_learner
        result = tactic_learner.run()
        assert isinstance(result, dict)
        assert len(result) == 3  # bull/bear/range
