"""tests/test_p2_e2e.py — P2 端到端冒烟。

验证:
  - orchestrator.MODULES 含 tactic_learner 且 depends_on=["market_state"]
  - market_state 先于 tactic_learner 注册
  - run_all(dry_run=True) 对 tactic_learner 返回 status=dry_run
  - run_all 真实模式:market_state 失败 → tactic_learner 自动 skipped
  - run_all 真实模式:数据空白时 tactic_learner 不抛,产物落盘
  - orchestrator.check() 含 tactic_params.json 校验
  - 同一次 run_all 产出 model_learner.json + tactic_params.json 共存
"""
from __future__ import annotations

import json

import pytest


# ── MODULES 注册 ───────────────────────────────────────────

class TestModulesRegistry:
    def test_tactic_learner_registered(self):
        from learning.orchestrator import MODULES
        names = [m["name"] for m in MODULES]
        assert "tactic_learner" in names

    def test_tactic_learner_depends_on_market_state(self):
        from learning.orchestrator import MODULES
        m = next(m for m in MODULES if m["name"] == "tactic_learner")
        assert "market_state" in m["depends_on"]

    def test_market_state_before_tactic_learner(self):
        from learning.orchestrator import MODULES
        names = [m["name"] for m in MODULES]
        assert names.index("market_state") < names.index("tactic_learner")


# ── dry_run ────────────────────────────────────────────────

class TestRunAllDryRun:
    def test_all_modules_dry_run(self):
        from learning.orchestrator import run_all
        results = run_all(date_str="2026-04-29", dry_run=True)
        assert results["market_state"]["status"] == "dry_run"
        assert results["model_learner"]["status"] == "dry_run"
        assert results["tactic_learner"]["status"] == "dry_run"


# ── 依赖失败传播 ─────────────────────────────────────────

class TestRunAllDependencyPropagation:
    def test_tactic_learner_skipped_when_market_state_fails(self, monkeypatch):
        from learning import orchestrator, alerts

        def boom(date_str=None):
            raise RuntimeError("fake")

        patched = []
        for m in orchestrator.MODULES:
            if m["name"] == "market_state":
                patched.append({**m, "run": boom})
            else:
                patched.append(m)
        monkeypatch.setattr(orchestrator, "MODULES", patched)
        monkeypatch.setattr(alerts, "send_alert", lambda *a, **k: None)

        results = orchestrator.run_all(date_str="2026-04-29")
        assert results["market_state"]["status"] == "failed"
        assert results["tactic_learner"]["status"] == "skipped"
        assert "market_state" in results["tactic_learner"]["reason"]


# ── 真实模式 冷启动 ────────────────────────────────────

class TestRunAllColdStart:
    def test_tactic_learner_persists_even_without_samples(self, tmp_path, monkeypatch):
        """数据空白时 tactic_learner 仍写出合法 tactic_params.json。"""
        from learning import orchestrator, tactic_learner, model_learner, feedback_io
        from learning import market_state, tracker

        # 隔离路径
        monkeypatch.setattr(model_learner, "STATE_FILE", tmp_path / "ml.json")
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path / "saved")
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(tactic_learner, "STATE_FILE", tmp_path / "tp.json")
        monkeypatch.setattr(tactic_learner, "CONFIG_PATH", tmp_path / "tl.yaml")
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")

        # market_state 打桩(必须同时 patch MODULES 里的 run 引用,orchestrator 导入时已绑定)
        fake_state = lambda date_str=None: {"current": "range"}
        monkeypatch.setattr(market_state, "run", fake_state)
        monkeypatch.setattr(market_state, "load_current_state",
                            lambda: {"current": "range"})
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "range")
        # 替换 MODULES 里 market_state 项的 run 引用
        patched = []
        for m in orchestrator.MODULES:
            if m["name"] == "market_state":
                patched.append({**m, "run": fake_state})
            else:
                patched.append(m)
        monkeypatch.setattr(orchestrator, "MODULES", patched)

        # 写 tactic yaml
        (tmp_path / "tl.yaml").write_text(
            """
defaults:
  value:  {roe_min: {value: 8.0, direction: tighten_up}}
  growth: {roe_min: {value: 12.0, direction: tighten_up}}
  leader: {roe_min: {value: 15.0, direction: tighten_up}}
  contra: {roe_min: {value: 3.0,  direction: tighten_up}}
bounds:
  value:  {roe_min: [5.0, 15.0]}
  growth: {roe_min: [8.0, 18.0]}
  leader: {roe_min: [12.0, 20.0]}
  contra: {roe_min: [2.0, 8.0]}
step: {default: 1.0, drawdown: 0.01}
evaluation: {lookback_days: 90, min_samples: 30, acc_low: 0.30, acc_high: 0.60}
weights:
  enable_resonance_rank_boost: true
  rank_boost_per_tactic: 0.02
  rank_boost_max: 0.06
  weight_min: 0.05
  weight_max: 0.40
""", encoding="utf-8")

        # model_learner yaml
        (tmp_path / "model_learner.yaml").write_text(
            """
sample_weights:
  buy_miss: 2.0
  buy_weak: 1.2
  buy_good: 1.0
  buy_great: 1.5
  non_buy: 0.8
  default: 1.0
calibration:
  lookback_days: 90
  min_samples_per_bucket: 50
  cold_start_fallback: global
absolute_threshold:
  initial: 0.30
  min: 0.30
  max: 0.60
  step: 0.02
  lookback_days: 30
  min_buy_signals: 10
""", encoding="utf-8")
        monkeypatch.setattr(model_learner, "CONFIG_PATH", tmp_path / "model_learner.yaml")

        results = orchestrator.run_all(date_str="2026-04-29")
        assert results["market_state"]["status"] == "ok"
        assert results["model_learner"]["status"] == "ok"
        assert results["tactic_learner"]["status"] == "ok"

        # 产物都落盘
        assert (tmp_path / "ml.json").exists()
        assert (tmp_path / "tp.json").exists()

        # tactic_params.json 结构合法
        tp_data = json.loads((tmp_path / "tp.json").read_text(encoding="utf-8"))
        assert tp_data.get("version") == 1
        assert "params" in tp_data
        # 冷启动 bear 桶用 override 兜底
        # (本测试 yaml 没写 defaults_bear_override,bear 值与 defaults 一致也可)


# ── orchestrator.check() ───────────────────────────────────

class TestCheckCommand:
    def test_check_passes_with_valid_p2_artifacts(self, tmp_path, monkeypatch):
        from learning import orchestrator
        monkeypatch.chdir(tmp_path)
        (tmp_path / "learning").mkdir()
        (tmp_path / "learning" / "tactic_params.json").write_text(
            json.dumps({"version": 1, "params": {}}), encoding="utf-8"
        )
        assert orchestrator.check() == 0

    def test_check_fails_with_corrupt_tactic_params(self, tmp_path, monkeypatch):
        from learning import orchestrator
        monkeypatch.chdir(tmp_path)
        (tmp_path / "learning").mkdir()
        (tmp_path / "learning" / "tactic_params.json").write_text(
            "not-json{", encoding="utf-8"
        )
        assert orchestrator.check() == 1
