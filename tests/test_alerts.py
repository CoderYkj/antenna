import json
from pathlib import Path
import pytest
from learning import alerts


@pytest.fixture
def alerts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.jsonl")
    return tmp_path


def test_record_alert_appends_jsonl_line(alerts_dir):
    alerts.record_alert("market_state", "data fetch failed", severity="error")
    lines = (alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["module"] == "market_state"
    assert data["severity"] == "error"


def test_multiple_alerts_accumulate(alerts_dir):
    alerts.record_alert("m1", "err1", "warning")
    alerts.record_alert("m2", "err2", "error")
    lines = (alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_severity_defaults_to_error(alerts_dir):
    alerts.record_alert("m", "msg")
    data = json.loads((alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip())
    assert data["severity"] == "error"


def test_push_to_feishu_not_called_when_no_webhook(alerts_dir, monkeypatch):
    called = []

    def fake_push(hook, msg):
        called.append(msg)

    monkeypatch.setattr(alerts, "_push_feishu", fake_push)
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "")
    alerts.send_alert("market_state", "failed")
    assert called == []


def test_send_alert_calls_push_when_webhook_present(alerts_dir, monkeypatch):
    pushed = []
    monkeypatch.setattr(alerts, "_push_feishu", lambda hook, msg: pushed.append((hook, msg)))
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "https://hook.example")
    alerts.send_alert("market_state", "failed: connection refused")
    assert len(pushed) == 1
    assert "market_state" in pushed[0][1]


def test_send_alert_writes_jsonl_even_without_webhook(alerts_dir, monkeypatch):
    """无 webhook 时仍应写入 alerts.jsonl。"""
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "")
    alerts.send_alert("m", "boom")
    assert (alerts_dir / "alerts.jsonl").exists()


def test_send_alert_with_traceback_includes_format_exc(alerts_dir, monkeypatch):
    """traceback=True 时应在 message 中包含 traceback 文本(模拟异常栈)。"""
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "")
    try:
        raise ValueError("synthetic")
    except ValueError:
        alerts.send_alert("m", "failed", traceback=True)
    line = (alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert "ValueError" in data["message"] or "synthetic" in data["message"]
