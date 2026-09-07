"""tests/test_review_guardrail_reporting.py — 复盘护栏追踪展示测试。"""

from unittest.mock import patch


def _sample_report():
    return {
        "date": "2026-01-01",
        "day_result": {
            "details": [
                {"code": "000001", "name": "平安银行", "is_buy": True, "rise_prob": 0.61, "actual_pct": 1.2, "hit": True},
            ],
            "hits": 1,
            "total": 1,
            "accuracy": 1.0,
        },
        "acc_7d": 0.6,
        "acc_30d": 0.35,
        "samples_7": 10,
        "samples_30": 40,
        "change_desc": "测试调整",
        "strategy": {"buy_top_pct": 0.12, "target_accuracy": 0.55},
        "guardrail_trace": {
            "state": "bear",
            "min_pct": 0.06,
            "max_pct": 0.12,
            "buy_top_pct_before": 0.13,
            "buy_top_pct_after": 0.12,
            "reasons": ["state_bound_clamp", "low_acc_30d_cap"],
        },
        "guardrail_summary": {
            "window_days": 30,
            "total_days": 30,
            "trigger_days": 5,
            "trigger_rate": 0.1667,
            "top_reasons": [{"code": "low_acc_30d_cap", "label": "30日低精准率上限收敛", "count": 4}],
            "recent_events": [],
        },
        "dashboard_metrics": {
            "windows": {
                "7d": {"hit_rate": 0.6, "samples": 10, "avg_return": 0.012, "max_drawdown": -0.03, "topn_hit_rate": {"top5": 0.7}},
                "30d": {"hit_rate": 0.35, "samples": 40, "avg_return": 0.003, "max_drawdown": -0.11, "topn_hit_rate": {"top5": 0.42}},
            },
            "alerts": [
                {"level": "warning", "code": "over_demotion_risk", "message": "存在过度降级风险，建议回调 weak 模式闸门强度。"},
            ],
        },
    }


def test_send_review_report_contains_guardrail_trace():
    from notify.feishu import send_review_report

    with patch("notify.feishu._post", return_value=True) as mock_post:
        ok = send_review_report("http://hook", _sample_report())

    assert ok is True
    payload = mock_post.call_args[0][1]
    text = " ".join(e.get("content", "") for e in payload["card"]["elements"] if e.get("tag") == "markdown")
    assert "护栏追踪" in text
    assert "13%" in text and "12%" in text
    assert "状态区间限幅" in text
    assert "30日护栏" in text
    assert "7天/30天监控看板" in text
    assert "提精风险告警" in text


def test_send_review_report_no_webhook_skips():
    from notify.feishu import send_review_report

    with patch("notify.feishu._post") as mock_post:
        ok = send_review_report("", _sample_report())
    assert ok is False
    mock_post.assert_not_called()
