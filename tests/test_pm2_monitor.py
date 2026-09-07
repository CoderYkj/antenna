"""tests/test_pm2_monitor.py - unit tests for server/pm2_monitor.py."""
import json
from unittest.mock import patch

import pytest


# ── helpers ─────────────────────────────────────────────────────────────

def _make_proc(name="antenna-bot", status="online", pid=1234,
               cpu=1, mem=50 * 1024 * 1024, restarts=0, uptime_ms=None):
    return {
        "name": name,
        "pid":  pid,
        "pm2_env": {
            "status":       status,
            "restart_time": restarts,
            "pm_uptime":    uptime_ms,
        },
        "monit": {"cpu": cpu, "memory": mem},
    }


@pytest.fixture()
def client(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "pm2_monitor:\n  username: admin\n  password: secret\n  port: 9615\n",
        encoding="utf-8",
    )
    from server import pm2_monitor
    pm2_monitor.CONFIG_PATH = cfg_path
    pm2_monitor.app.config["TESTING"] = True
    with pm2_monitor.app.test_client() as c:
        yield c


def _auth(client, path, method="get", **kwargs):
    fn = getattr(client, method)
    return fn(path, headers={"Authorization": "Basic YWRtaW46c2VjcmV0"}, **kwargs)


# ── _format_uptime ──────────────────────────────────────────────────────

def test_format_uptime_none_returns_dash():
    from server.pm2_monitor import _format_uptime
    assert _format_uptime(None) == "—"
    assert _format_uptime(0) == "—"


def test_format_uptime_seconds():
    import time
    from server.pm2_monitor import _format_uptime
    now_ms = int(time.time() * 1000)
    assert "s" in _format_uptime(now_ms - 45_000)


def test_format_uptime_days():
    import time
    from server.pm2_monitor import _format_uptime
    now_ms = int(time.time() * 1000)
    assert "d" in _format_uptime(now_ms - 2 * 86400 * 1000)


# ── _format_mem ─────────────────────────────────────────────────────────

def test_format_mem_none_returns_dash():
    from server.pm2_monitor import _format_mem
    assert _format_mem(None) == "—"
    assert _format_mem(0) == "—"


def test_format_mem_megabytes():
    from server.pm2_monitor import _format_mem
    assert _format_mem(100 * 1024 * 1024) == "100.0 MB"


# ── _pm2_jlist ──────────────────────────────────────────────────────────

def test_pm2_jlist_returns_empty_on_error():
    from server import pm2_monitor
    with patch.object(pm2_monitor, "_pm2", return_value=(1, "", "not found")):
        assert pm2_monitor._pm2_jlist() == []


def test_pm2_jlist_returns_empty_on_bad_json():
    from server import pm2_monitor
    with patch.object(pm2_monitor, "_pm2", return_value=(0, "{BROKEN", "")):
        assert pm2_monitor._pm2_jlist() == []


def test_pm2_jlist_parses_valid_json():
    from server import pm2_monitor
    procs = [_make_proc()]
    with patch.object(pm2_monitor, "_pm2", return_value=(0, json.dumps(procs), "")):
        result = pm2_monitor._pm2_jlist()
    assert len(result) == 1
    assert result[0]["name"] == "antenna-bot"


# ── HTTP routes ─────────────────────────────────────────────────────────

def test_healthz_no_auth(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert b"ok" in resp.data


def test_index_requires_auth(client):
    resp = client.get("/")
    assert resp.status_code == 401


def test_index_shows_processes(client):
    from server import pm2_monitor
    procs = [_make_proc("antenna-bot", status="online")]
    with patch.object(pm2_monitor, "_pm2_jlist", return_value=procs):
        resp = _auth(client, "/")
    assert resp.status_code == 200
    assert b"antenna-bot" in resp.data
    assert b"online" in resp.data


def test_index_empty_when_no_procs(client):
    from server import pm2_monitor
    with patch.object(pm2_monitor, "_pm2_jlist", return_value=[]):
        resp = _auth(client, "/")
    assert resp.status_code == 200
    assert "没有运行的 PM2 进程".encode() in resp.data


def test_action_restart_redirects(client):
    from server import pm2_monitor
    with patch.object(pm2_monitor, "_pm2", return_value=(0, "ok", "")) as m:
        resp = _auth(client, "/action/restart/antenna-bot", method="post")
    assert resp.status_code in (302, 303)
    m.assert_called_once_with("restart", "antenna-bot")


def test_action_invalid_name_rejected(client):
    resp = _auth(client, "/action/restart/bad/name", method="post")
    assert resp.status_code == 404  # Flask path-sep rule catches this


def test_action_invalid_action_rejected(client):
    resp = _auth(client, "/action/delete/antenna-bot", method="post")
    assert resp.status_code == 400


def test_logs_returns_html(client):
    from server import pm2_monitor
    with patch.object(pm2_monitor, "_pm2", return_value=(0, "log line 1\nlog line 2", "")):
        resp = _auth(client, "/logs/antenna-bot")
    assert resp.status_code == 200
    assert b"log line 1" in resp.data
    assert b"/api/logs/" in resp.data


def test_api_logs_returns_json(client):
    from server import pm2_monitor
    with patch.object(pm2_monitor, "_pm2", return_value=(0, "line x", "")):
        resp = _auth(client, "/api/logs/antenna-bot")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["name"] == "antenna-bot"
    assert data["text"] == "line x"


def test_api_processes_returns_json(client):
    from server import pm2_monitor
    procs = [_make_proc()]
    with patch.object(pm2_monitor, "_pm2_jlist", return_value=procs):
        resp = _auth(client, "/api/processes")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data[0]["name"] == "antenna-bot"
