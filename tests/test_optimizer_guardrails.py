"""tests/test_optimizer_guardrails.py — 策略风险护栏与默认字段兼容测试。"""

import json
from pathlib import Path


def test_load_strategy_merges_new_defaults(tmp_path, monkeypatch):
    from learning import optimizer

    strategy_file = tmp_path / "strategy.json"
    strategy_file.write_text(json.dumps({"buy_top_pct": 0.11}), encoding="utf-8")
    monkeypatch.setattr(optimizer, "STRATEGY_FILE", Path(strategy_file))

    loaded = optimizer.load_strategy()
    assert loaded["buy_top_pct"] == 0.11
    assert "risk_guardrails" in loaded
    assert "positioning" in loaded
    assert loaded["risk_guardrails"]["by_state_bounds"]["bear"]["max"] == 0.14


def test_resolve_buy_top_bounds_applies_low_accuracy_cap(monkeypatch):
    from learning import optimizer

    monkeypatch.setattr(optimizer, "load_current_state", lambda: {"current": "bear"})
    strategy = optimizer._default_strategy()
    min_pct, max_pct, state = optimizer._resolve_buy_top_bounds(strategy, acc_30d=0.35)

    assert state == "bear"
    assert min_pct == 0.06
    assert max_pct == 0.12  # bear 上限 0.14，再被低准确率 cap 到 0.12


def test_optimize_clamps_by_guardrail(monkeypatch):
    from learning import optimizer

    strategy = optimizer._default_strategy()
    strategy["buy_top_pct"] = 0.13

    monkeypatch.setattr(optimizer, "load_strategy", lambda: strategy.copy())
    monkeypatch.setattr(optimizer, "save_strategy", lambda _s: None)
    monkeypatch.setattr(optimizer, "load_current_state", lambda: {"current": "bear"})

    calls = {"n": 0}

    def _rolling(window_days):
        calls["n"] += 1
        if window_days == 7:
            return 0.70, 7, 10
        return 0.35, 35, 100

    monkeypatch.setattr(optimizer, "rolling_accuracy", _rolling)
    updated, change = optimizer.optimize({"date": "2026-01-01"})

    assert updated["buy_top_pct"] == 0.12
    assert "13%" in change and "12%" in change
    trace = updated.get("last_guardrail_trace", {})
    assert trace.get("state") == "bear"
    assert trace.get("triggered") is True
    assert "low_acc_30d_cap" in (trace.get("reasons") or [])
    assert "30日低精准率上限收敛" in trace.get("reason_text", "")
    assert "护栏追踪" in trace.get("summary", "")
    hist = updated.get("history", [])
    assert hist
    latest = hist[-1]
    assert latest.get("guardrail_triggered") is True
    assert latest.get("guardrail_reason_text") == trace.get("reason_text")


def test_format_guardrail_reasons_maps_labels():
    from learning.optimizer import format_guardrail_reasons

    assert format_guardrail_reasons(["state_bound_clamp", "floor_reset"]) == "状态区间限幅、下限连续触发重置"
    assert format_guardrail_reasons([]) == "无"


def test_summarize_guardrail_history_counts_reasons():
    from learning.optimizer import summarize_guardrail_history

    history = [
        {"date": "2026-01-01", "guardrail_triggered": False},
        {"date": "2026-01-02", "guardrail_triggered": True, "guardrail_reasons": ["low_acc_30d_cap"], "guardrail_reason_text": "30日低精准率上限收敛"},
        {"date": "2026-01-03", "guardrail_triggered": True, "guardrail_reasons": ["state_bound_clamp", "low_acc_30d_cap"], "guardrail_reason_text": "状态区间限幅、30日低精准率上限收敛"},
    ]
    s = summarize_guardrail_history(history, 30)
    assert s["trigger_days"] == 2
    assert s["total_days"] == 3
    assert s["top_reasons"][0]["code"] == "low_acc_30d_cap"
    assert s["top_reasons"][0]["count"] == 2


def test_build_monitor_dashboard_metrics(monkeypatch):
    from learning import optimizer

    strategy = optimizer._default_strategy()
    strategy["rise_target_pct"] = 1.0
    strategy["history"] = [
        {"date": "2026-01-01", "guardrail_triggered": True, "guardrail_reasons": ["low_acc_30d_cap"], "change": "a"},
        {"date": "2026-01-02", "guardrail_triggered": False, "change": ""},
    ]
    monkeypatch.setattr(optimizer, "load_strategy", lambda: strategy)
    monkeypatch.setattr(optimizer, "list_prediction_dates", lambda: ["2026-01-01", "2026-01-02"])

    preds = {
        "2026-01-01": [
            {"code": "000001", "signal": "买入", "rise_prob": 0.66, "confidence": "高", "global_rank_pct": 0.01},
            {"code": "000002", "signal": "买入", "rise_prob": 0.61, "confidence": "中", "global_rank_pct": 0.02},
            {"code": "000003", "signal": "观望", "rise_prob": 0.58, "global_rank_pct": 0.03},
        ],
        "2026-01-02": [
            {"code": "000004", "signal": "买入", "rise_prob": 0.59, "confidence": "低", "global_rank_pct": 0.01},
        ],
    }
    outs = {
        "2026-01-01": {"000001": {"actual_pct": 2.0}, "000002": {"actual_pct": -1.0}, "000003": {"actual_pct": 0.5}},
        "2026-01-02": {"000004": {"actual_pct": 1.2}},
    }
    monkeypatch.setattr(optimizer, "load_predictions", lambda d: preds.get(d, []))
    monkeypatch.setattr(optimizer, "load_outcomes", lambda d: outs.get(d, {}))

    payload = optimizer.build_monitor_dashboard_metrics()
    w7 = payload["windows"]["7d"]
    assert w7["samples"] == 3
    assert w7["hits"] == 2
    assert w7["hit_rate"] == 0.6667
    assert w7["topn_hit_rate"]["top3"] == 0.6667
    assert w7["confidence_hit_rate"]["高"] == 1.0
    assert payload["parameter_changes"]["7d"] == 1
    assert payload["alerts"] == []


def test_build_monitor_dashboard_metrics_over_demotion_alert(monkeypatch):
    from learning import optimizer

    strategy = optimizer._default_strategy()
    strategy["rise_target_pct"] = 1.0
    strategy["history"] = [
        {"date": "2026-01-14", "guardrail_triggered": True, "guardrail_reasons": ["low_acc_30d_cap"], "change": "a"},
        {"date": "2026-01-15", "guardrail_triggered": True, "guardrail_reasons": ["state_bound_clamp"], "change": "b"},
        {"date": "2026-01-16", "guardrail_triggered": True, "guardrail_reasons": ["low_acc_30d_cap"], "change": ""},
        {"date": "2026-01-17", "guardrail_triggered": True, "guardrail_reasons": ["state_bound_clamp"], "change": ""},
        {"date": "2026-01-18", "guardrail_triggered": False, "change": ""},
        {"date": "2026-01-19", "guardrail_triggered": False, "change": ""},
        {"date": "2026-01-20", "guardrail_triggered": False, "change": ""},
    ]
    monkeypatch.setattr(optimizer, "load_strategy", lambda: strategy)
    dates = [f"2026-01-{i:02d}" for i in range(1, 21)]
    monkeypatch.setattr(optimizer, "list_prediction_dates", lambda: dates)

    def _preds(d):
        if d in {"2026-01-14", "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19", "2026-01-20"}:
            if d in {"2026-01-18", "2026-01-19", "2026-01-20"}:
                return [{"code": d[-2:] + "0001", "signal": "买入", "rise_prob": 0.61, "confidence": "中", "global_rank_pct": 0.02}]
            return [{"code": d[-2:] + "0002", "signal": "观望", "rise_prob": 0.55, "confidence": "低", "global_rank_pct": 0.20}]
        else:
            return [{"code": d[-2:] + "0001", "signal": "买入", "rise_prob": 0.61, "confidence": "中", "global_rank_pct": 0.02}]

    def _outs(d):
        rows = _preds(d)
        out = {}
        for r in rows:
            out[r["code"]] = {"actual_pct": 1.0}
        return out

    monkeypatch.setattr(optimizer, "load_predictions", _preds)
    monkeypatch.setattr(optimizer, "load_outcomes", _outs)

    payload = optimizer.build_monitor_dashboard_metrics()
    alerts = payload["alerts"]
    assert alerts
    assert alerts[0]["code"] == "over_demotion_risk"
