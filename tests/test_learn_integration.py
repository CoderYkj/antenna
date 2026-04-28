"""
集成测试:走一遍完整的 learn 管线,确保 P0 基础设施衔接正确。
"""
import json
from pathlib import Path
import pytest


def test_cli_learn_dry_run(tmp_path, monkeypatch):
    """模拟 cli.py learn --dry-run 不触发实际网络/磁盘操作。"""
    from learning import orchestrator
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    results = orchestrator.run_all(date_str="2026-04-27", dry_run=True)
    # P0 默认只挂了 market_state
    assert "market_state" in results
    assert results["market_state"]["status"] == "dry_run"


def test_tracker_and_bucket_pipeline(tmp_path, monkeypatch):
    """tracker 写入 → scene_bucket 读取的端到端。"""
    from learning import tracker
    from learning.scene_bucket import bucket_by_scene

    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)

    tracker.log_predictions("2026-04-27", [
        {"code": "A", "scene": "scan", "signal": "买入", "rise_prob": 0.7},
        {"code": "A", "scene": "tactic:value", "signal": "买入", "rise_prob": 0.72},
        {"code": "B", "scene": "scan", "signal": "观望", "rise_prob": 0.55},
    ])

    records = tracker.load_predictions("2026-04-27")
    buckets = bucket_by_scene(records)
    assert len(buckets["scan"]) == 2
    assert len(buckets["tactic:value"]) == 1


def test_outcome_metrics_end_to_end(tmp_path, monkeypatch):
    """写入扩展 outcome → load 验证字段存在。"""
    from learning import tracker
    from learning.outcome_metrics import compute_hit_tier

    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)

    outcomes = {
        "600519": {
            "actual_open": 1800, "actual_close": 1854, "actual_high": 1860, "actual_low": 1790,
            "actual_pct": 3.0,
            "hit_tier": compute_hit_tier(3.0),
            "hit_5d": 7.0,
            "max_drawdown_5d": -1.5,
        }
    }
    tracker.log_outcomes("2026-04-27", outcomes)
    loaded = tracker.load_outcomes("2026-04-27")
    assert loaded["600519"]["hit_tier"] == "good"
    assert loaded["600519"]["hit_5d"] == 7.0


def test_orchestrator_runs_market_state_with_real_data(tmp_path, monkeypatch):
    """验证编排器真能调用 market_state.run(含防御 — 若拿不到数据,fail 但不崩)。"""
    from learning import orchestrator, alerts

    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.jsonl")
    # 不允许真调飞书
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "")

    results = orchestrator.run_all(date_str="2026-04-27")
    # 不 assert ok(取决于环境网络),只要 orchestrator 不崩即可
    assert "market_state" in results
    assert results["market_state"]["status"] in ("ok", "failed")


def test_replay_evaluator_mirrors_optimizer_accuracy(tmp_path, monkeypatch):
    """
    回放脚本的 _evaluate_single 与 optimizer.evaluate_day 应当算出相同的精准率,
    这是 P0 验收的关键:否则回放基线与线上统计不一致,后续 learner 验收都失真。
    """
    from scripts.replay_learn import _evaluate_single
    from learning import tracker

    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    # 5 条 buy 信号,3 条涨幅 >= 1%(命中)
    tracker.log_predictions("2026-04-20", [
        {"code": "A1", "signal": "买入", "rise_prob": 0.7, "scene": "scan"},
        {"code": "A2", "signal": "买入", "rise_prob": 0.7, "scene": "scan"},
        {"code": "A3", "signal": "买入", "rise_prob": 0.7, "scene": "scan"},
        {"code": "A4", "signal": "买入", "rise_prob": 0.7, "scene": "scan"},
        {"code": "A5", "signal": "买入", "rise_prob": 0.7, "scene": "scan"},
        {"code": "W1", "signal": "观望", "rise_prob": 0.4, "scene": "scan"},
    ])
    tracker.log_outcomes("2026-04-20", {
        "A1": {"actual_pct": 2.5},
        "A2": {"actual_pct": 1.2},
        "A3": {"actual_pct": 3.8},
        "A4": {"actual_pct": 0.3},
        "A5": {"actual_pct": -1.5},
        "W1": {"actual_pct": 0.1},
    })

    preds = tracker.load_predictions("2026-04-20")
    outcomes = tracker.load_outcomes("2026-04-20")
    result = _evaluate_single(preds, outcomes, scene_filter=None)

    scan = result["scan"]
    assert scan["buys"] == 5      # 5 买入信号
    assert scan["scored"] == 5    # 全部都有 outcome
    assert scan["hits"] == 3      # 2.5 / 1.2 / 3.8 命中
    assert scan["accuracy"] == 0.6


def test_fill_5d_metrics_on_insufficient_data(tmp_path, monkeypatch):
    """对没 5 天后续数据的日期,fill_5d 应静默跳过,不崩。"""
    from scripts import task_fill_5d_metrics
    from learning import tracker

    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    tracker.log_outcomes("2026-04-25", {
        "X": {"actual_pct": 1.5},
    })

    # 用未来日期(肯定没 5 天数据),不应抛异常
    result = task_fill_5d_metrics.fill_5d_metrics_for_date("2026-04-25")
    assert "filled" in result
    assert "skipped" in result
