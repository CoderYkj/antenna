from pathlib import Path
import pytest
from learning import orchestrator


def test_registry_has_market_state():
    names = [m["name"] for m in orchestrator.MODULES]
    assert "market_state" in names


def test_module_failure_does_not_block_others(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    calls = []

    def ok_run(date_str=None):
        calls.append("ok")
        return {"state": "range"}

    def boom_run(date_str=None):
        calls.append("boom")
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "boom",        "run": boom_run,  "depends_on": []},
        {"name": "after_boom",  "run": ok_run,    "depends_on": []},
    ])
    # 让 send_alert 静默,避免往本机真飞书发
    monkeypatch.setattr(orchestrator.alerts, "_get_webhook", lambda: "")
    monkeypatch.setattr(orchestrator.alerts, "ALERTS_FILE", tmp_path / "alerts.jsonl")

    result = orchestrator.run_all(date_str="2026-04-27")
    assert "boom" in calls
    assert "ok" in calls
    assert result["boom"]["status"] == "failed"
    assert result["after_boom"]["status"] == "ok"


def test_module_with_unmet_dependency_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    def boom_run(date_str=None):
        raise RuntimeError("boom")

    def after_run(date_str=None):
        return {"ok": True}

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "boom",  "run": boom_run,  "depends_on": []},
        {"name": "after", "run": after_run, "depends_on": ["boom"]},
    ])
    monkeypatch.setattr(orchestrator.alerts, "_get_webhook", lambda: "")
    monkeypatch.setattr(orchestrator.alerts, "ALERTS_FILE", tmp_path / "alerts.jsonl")

    result = orchestrator.run_all(date_str="2026-04-27")
    assert result["after"]["status"] == "skipped"
    assert result["boom"]["status"] == "failed"


def test_dry_run_does_not_execute_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    called = []

    def must_not_run(date_str=None):
        called.append(1)

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "m", "run": must_not_run, "depends_on": []},
    ])

    result = orchestrator.run_all(date_str="2026-04-27", dry_run=True)
    assert called == []
    assert result["m"]["status"] == "dry_run"


def test_run_all_writes_feedback_for_successful_module(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    def ok_run(date_str=None):
        return {"state": "bull", "samples_used": 100}

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "m", "run": ok_run, "depends_on": []},
    ])

    orchestrator.run_all(date_str="2026-04-27")
    feedback_path = tmp_path / "feedback" / "m_2026-04-27.json"
    assert feedback_path.exists()


def test_check_returns_zero_when_all_files_valid(tmp_path, monkeypatch):
    import json
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "market_state.json").write_text(
        json.dumps({"current": "range", "history": []}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code == 0


def test_check_returns_nonzero_on_corrupt_json(tmp_path, monkeypatch):
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "market_state.json").write_text("{BROKEN", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code != 0


# ── P3/P4 新模块注册测试 ──────────────────────────────────

def test_registry_has_feature_learner():
    names = [m["name"] for m in orchestrator.MODULES]
    assert "feature_learner" in names


def test_registry_has_price_learner():
    names = [m["name"] for m in orchestrator.MODULES]
    assert "price_learner" in names


def test_check_passes_when_only_feature_weights_present(tmp_path, monkeypatch):
    import json
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"weights": {}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code == 0


def test_check_passes_when_only_price_params_present(tmp_path, monkeypatch):
    import json
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "price_params.json").write_text(
        json.dumps({"atr_mult": 1.5}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code == 0


def test_check_returns_nonzero_on_corrupt_feature_weights(tmp_path, monkeypatch):
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text("{BROKEN", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code != 0


def test_check_returns_nonzero_on_corrupt_price_params(tmp_path, monkeypatch):
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "price_params.json").write_text("{BROKEN", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code != 0


# ── feature_weights.json active 语义校验 ───────────────────────

def test_check_rejects_unknown_active_col(tmp_path, monkeypatch):
    """active 列表含非法列名 → check() 返回非零。"""
    import json
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"active": ["ma5", "rsi6", "__totally_unknown_col__"]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code != 0


def test_check_accepts_valid_active_cols(tmp_path, monkeypatch):
    """active 列表只含合法列名 → check() 返回 0。"""
    import json
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"active": ["ma5", "rsi6", "main_net_in_1d"]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code == 0
