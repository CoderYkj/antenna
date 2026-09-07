"""tests/test_predict_cmd_guardrails.py — 推荐卡片风险护栏文案测试。"""


def test_strategy_guardrail_snapshot_triggers_cap():
    from server.predict_cmd import _strategy_guardrail_snapshot

    strategy = {
        "accuracy_30d": 0.35,
        "risk_guardrails": {
            "by_state_bounds": {"bear": {"min": 0.06, "max": 0.14}},
            "low_accuracy_cap": {"acc_30d_threshold": 0.40, "max_buy_top_pct": 0.12},
        },
        "positioning": {"bear": 0.3},
    }
    snap = _strategy_guardrail_snapshot(strategy, "bear")
    assert snap["cap_triggered"] is True
    assert snap["min_pct"] == 0.06
    assert snap["max_pct"] == 0.12
    assert snap["position_pct"] == 0.3


def test_strategy_guardrail_line_includes_deviation_hint():
    from server.predict_cmd import _strategy_guardrail_line

    strategy = {
        "accuracy_30d": 0.50,
        "risk_guardrails": {
            "by_state_bounds": {"range": {"min": 0.08, "max": 0.20}},
            "low_accuracy_cap": {"acc_30d_threshold": 0.40, "max_buy_top_pct": 0.12},
        },
        "positioning": {"range": 0.6},
    }
    line = _strategy_guardrail_line(strategy, current_state="range", buy_top_pct=0.25)
    assert "护栏" in line
    assert "建议仓位" in line
    assert "偏离护栏区间" in line
