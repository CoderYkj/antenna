"""tests/test_p1_e2e.py — P1 端到端冒烟。

验证:
  - orchestrator.MODULES 含 model_learner 且 depends_on=["market_state"]
  - run_all(dry_run=True) 对 model_learner 返回 status=dry_run
  - run_all 真实模式:market_state 失败时 model_learner 自动 skipped
  - run_all 真实模式:model 缺失时 model_learner 不抛,返回 no_model
  - orchestrator.check() 在正常 json 下返回 0
  - orchestrator.check() 在损坏 model_learner.json 下返回 1
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── 编排器 MODULES 注册 ─────────────────────────────────────

class TestModulesRegistry:
    def test_model_learner_registered(self):
        from learning.orchestrator import MODULES
        names = [m["name"] for m in MODULES]
        assert "model_learner" in names

    def test_model_learner_depends_on_market_state(self):
        from learning.orchestrator import MODULES
        ml = next(m for m in MODULES if m["name"] == "model_learner")
        assert "market_state" in ml["depends_on"]

    def test_market_state_runs_before_model_learner(self):
        """依赖顺序:market_state 注册位置必须在 model_learner 之前。"""
        from learning.orchestrator import MODULES
        names = [m["name"] for m in MODULES]
        assert names.index("market_state") < names.index("model_learner")


# ── run_all dry_run 场景 ───────────────────────────────────

class TestRunAllDryRun:
    def test_all_modules_dry_run(self):
        from learning.orchestrator import run_all
        results = run_all(date_str="2026-04-29", dry_run=True)
        assert results["market_state"]["status"] == "dry_run"
        assert results["model_learner"]["status"] == "dry_run"


# ── run_all 真实模式:依赖失败传播 ───────────────────────

class TestRunAllDependencyPropagation:
    def test_model_learner_skipped_when_market_state_fails(self, monkeypatch):
        """构造 market_state 模块的 run 抛异常,验证 model_learner 被标记 skipped。"""
        from learning import orchestrator, alerts
        # MODULES 在 import 时已绑定 market_state.run 引用,需要替换 MODULES 里那一项的 run
        def boom(date_str=None):
            raise RuntimeError("fake")

        patched = []
        for m in orchestrator.MODULES:
            if m["name"] == "market_state":
                patched.append({**m, "run": boom})
            else:
                patched.append(m)
        monkeypatch.setattr(orchestrator, "MODULES", patched)
        # 静默 alerts 推送
        monkeypatch.setattr(alerts, "send_alert", lambda *a, **k: None)

        results = orchestrator.run_all(date_str="2026-04-29")
        assert results["market_state"]["status"] == "failed"
        assert results["model_learner"]["status"] == "skipped"
        assert "market_state" in results["model_learner"]["reason"]


# ── run_all 真实模式:无模型时 model_learner 优雅降级 ─────

class TestRunAllNoModelGraceful:
    def test_model_learner_returns_no_model_when_pkl_missing(self, tmp_path, monkeypatch):
        """models/saved/ 无 model_*.pkl → 所有桶 status=no_model,run_all 不抛。"""
        from learning import orchestrator, model_learner, market_state, feedback_io

        # 隔离学习产物路径
        monkeypatch.setattr(model_learner, "STATE_FILE", tmp_path / "ml.json")
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path / "saved")
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")
        (tmp_path / "data").mkdir()
        # 让 market_state.run 返回 ok 不碰网络
        monkeypatch.setattr(market_state, "run",
                            lambda date_str=None: {"current": "range"})
        # yaml 用默认路径,确保存在
        yaml_path = tmp_path / "model_learner.yaml"
        yaml_path.write_text(
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
  initial: 0.45
  min: 0.30
  max: 0.60
  step: 0.02
  lookback_days: 30
  min_buy_signals: 10
""", encoding="utf-8")
        monkeypatch.setattr(model_learner, "CONFIG_PATH", yaml_path)

        results = orchestrator.run_all(date_str="2026-04-29")
        assert results["market_state"]["status"] == "ok"
        assert results["model_learner"]["status"] == "ok"
        # result 里每个 state 桶 status=no_model
        r = results["model_learner"]["result"]
        for state in ("bull", "bear", "range"):
            assert r[state]["status"] == "no_model"
        # 状态文件已落盘
        assert (tmp_path / "ml.json").exists()
        data = json.loads((tmp_path / "ml.json").read_text(encoding="utf-8"))
        assert "abs_threshold" in data


# ── orchestrator.check() ───────────────────────────────────

class TestCheckCommand:
    def test_check_passes_with_valid_json(self, tmp_path, monkeypatch, capsys):
        """注入合法 model_learner.json,check() 应返回 0。"""
        from learning import orchestrator
        # 切到 tmp_path 让 check 的 Path(...) 找不到文件(相对路径)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "learning").mkdir()
        (tmp_path / "learning" / "model_learner.json").write_text(
            json.dumps({"abs_threshold": 0.45}), encoding="utf-8")
        assert orchestrator.check() == 0

    def test_check_fails_with_corrupt_json(self, tmp_path, monkeypatch, capsys):
        from learning import orchestrator
        monkeypatch.chdir(tmp_path)
        (tmp_path / "learning").mkdir()
        (tmp_path / "learning" / "model_learner.json").write_text("not-json{", encoding="utf-8")
        assert orchestrator.check() == 1

    def test_check_allows_missing_files(self, tmp_path, monkeypatch):
        """所有产物文件都不存在时,check() 返回 0(首次启动场景)。"""
        from learning import orchestrator
        monkeypatch.chdir(tmp_path)
        assert orchestrator.check() == 0
