"""tests/test_cmd_strategy_learning.py — 策略/战法指令学习面板单元测试。"""
from unittest.mock import patch, MagicMock
import json


def _make_learning_files():
    ml = {"current_state": "range", "abs_threshold": 0.42, "last_calibrated": "2026-05-18T00:00:00"}
    tp = {"current_state": "range", "params": {"range": {
        "value":  {"precision": 0.61, "samples": 10},
        "growth": {"precision": 0.55, "samples": 8},
        "leader": {"precision": 0.58, "samples": 9},
        "contra": {"precision": 0.49, "samples": 6},
    }}}
    fw = {"active": list(range(24)), "last_updated": "2026-05-18"}
    pp = {"range": {"short_atr_mult": 1.5, "short_gain_mult": 2.0}}
    return ml, tp, fw, pp


def _rt_factory(ml, tp, fw, pp):
    def _rt(self, **kw):
        name = str(self)
        if "model_learner" in name:  return json.dumps(ml)
        if "tactic_params" in name:  return json.dumps(tp)
        if "feature_weights" in name: return json.dumps(fw)
        if "price_params" in name:   return json.dumps(pp)
        raise FileNotFoundError(name)
    return _rt


def test_cmd_strategy_contains_learning_panel():
    ml, tp, fw, pp = _make_learning_files()
    with patch("pathlib.Path.read_text", _rt_factory(ml, tp, fw, pp)), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("learning.optimizer.load_strategy", return_value={"buy_top_pct": 0.10}), \
         patch("learning.optimizer.rolling_accuracy", return_value=(0.60, 6, 10)), \
         patch("learning.tracker.list_prediction_dates", return_value=[]):
        from server.predict_cmd import cmd_strategy
        result = cmd_strategy()

    assert result.get("msg_type") == "interactive"
    card = json.loads(result["content"])
    all_text = " ".join(
        e.get("content", "") for e in card["elements"] if e.get("tag") == "markdown"
    )
    assert "P1 校准" in all_text
    assert "0.42" in all_text
    assert "P2 战法" in all_text
    assert "61%" in all_text
    assert "P3 特征" in all_text
    assert "24" in all_text
    assert "P4 价位" in all_text
    assert "1.5" in all_text


def test_cmd_strategy_learning_panel_degrades_gracefully():
    """学习文件全部不存在时，策略指令仍正常返回（无 learning panel 但不崩溃）。"""
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("learning.optimizer.load_strategy", return_value={"buy_top_pct": 0.10}), \
         patch("learning.optimizer.rolling_accuracy", return_value=(0.50, 5, 10)), \
         patch("learning.tracker.list_prediction_dates", return_value=[]):
        from server.predict_cmd import cmd_strategy
        result = cmd_strategy()

    assert result.get("msg_type") == "interactive"
    card = json.loads(result["content"])
    assert len(card["elements"]) > 0


def test_cmd_strategy_contains_guardrail_and_positioning_lines():
    strategy = {
        "buy_top_pct": 0.10,
        "history": [
            {
                "date": "2026-01-02",
                "buy_top_pct": 0.12,
                "acc_7d": 0.51,
                "change": "测试",
                "guardrail_triggered": True,
                "guardrail_reason_text": "状态区间限幅",
            }
        ],
        "risk_guardrails": {
            "by_state_bounds": {"range": {"min": 0.08, "max": 0.20}},
            "low_accuracy_cap": {"acc_30d_threshold": 0.40, "max_buy_top_pct": 0.12},
        },
        "positioning": {"bull": 1.0, "range": 0.6, "bear": 0.3},
    }
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("learning.optimizer.load_strategy", return_value=strategy), \
         patch("learning.optimizer.rolling_accuracy", return_value=(0.56, 6, 10)), \
         patch("learning.tracker.list_prediction_dates", return_value=[]):
        from server.predict_cmd import cmd_strategy
        result = cmd_strategy()

    card = json.loads(result["content"])
    all_text = " ".join(e.get("content", "") for e in card["elements"] if e.get("tag") == "markdown")
    assert "风险护栏" in all_text
    assert "仓位建议" in all_text
    assert "30日精准率 < 40%" in all_text
    assert "近期护栏触发" in all_text
    assert "状态区间限幅" in all_text
    assert "30日护栏统计" in all_text


def test_cmd_strategy_contains_precision_alerts():
    strategy = {"buy_top_pct": 0.10, "history": []}
    metrics = {
        "windows": {
            "7d": {"hit_rate": 0.52, "samples": 12, "avg_return": 0.01, "max_drawdown": -0.04, "topn_hit_rate": {"top5": 0.6}, "topn_samples": {"top5": 12}},
            "30d": {"hit_rate": 0.58, "samples": 40, "avg_return": 0.015, "max_drawdown": -0.07, "topn_hit_rate": {"top5": 0.62}, "topn_samples": {"top5": 40}},
        },
        "alerts": [
            {"level": "warning", "code": "hit_rate_drift", "message": "近7日命中率低于30日，短期质量走弱。"},
        ],
    }
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("learning.optimizer.load_strategy", return_value=strategy), \
         patch("learning.optimizer.rolling_accuracy", return_value=(0.52, 6, 12)), \
         patch("learning.optimizer.build_monitor_dashboard_metrics", return_value=metrics), \
         patch("learning.tracker.list_prediction_dates", return_value=[]):
        from server.predict_cmd import cmd_strategy
        result = cmd_strategy()

    card = json.loads(result["content"])
    all_text = " ".join(e.get("content", "") for e in card["elements"] if e.get("tag") == "markdown")
    assert "提精风险告警" in all_text
    assert "短期质量走弱" in all_text


def test_tactic_precision_line_returns_text():
    tp_data = {"params": {"bull": {"value": {"precision": 0.65, "samples": 12}}}}
    with patch("learning.market_state.load_current_state", return_value={"current": "bull"}), \
         patch("pathlib.Path.read_text", return_value=json.dumps(tp_data)):
        from server.predict_cmd import _tactic_precision_line
        line = _tactic_precision_line("value")

    assert "65%" in line
    assert "bull" in line
    assert "12" in line


def test_tactic_precision_line_graceful_on_missing_key():
    """战法 key 不在数据中时返回空串。"""
    tp_data = {"params": {"range": {}}}
    with patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("pathlib.Path.read_text", return_value=json.dumps(tp_data)):
        from server.predict_cmd import _tactic_precision_line
        line = _tactic_precision_line("contra")

    assert line == ""


def test_tactic_precision_line_graceful_on_file_missing():
    """文件不存在时返回空串，不抛异常。"""
    with patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        from server.predict_cmd import _tactic_precision_line
        line = _tactic_precision_line("value")

    assert line == ""
