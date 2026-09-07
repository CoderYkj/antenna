"""Unit tests for notify/dispatcher.py"""
from unittest.mock import patch, MagicMock
import pytest


_FEISHU_ONLY = {"feishu": True, "wecom": False}
_WECOM_ONLY = {"feishu": False, "wecom": True}
_BOTH = {"feishu": True, "wecom": True}
_NONE = {"feishu": False, "wecom": False}
_WEBHOOK = "https://example.com/hook"


def _cfg(channels: dict, webhook: str = _WEBHOOK):
    return (channels, webhook)


# ── send_scan_result ───────────────────────────────────────────────────────────

def test_only_feishu_enabled_calls_feishu_only():
    from notify import dispatcher
    scan = {"top": [], "total": 0, "skipped": 0, "thresh_strong": 0.6, "thresh_watch": 0.5}

    with patch.object(dispatcher, "_load_config", return_value=_cfg(_FEISHU_ONLY)):
        with patch("notify.feishu.send_scan_result", return_value=True) as fs:
            with patch("notify.wecom.send_scan_result", return_value=True) as ws:
                result = dispatcher.send_scan_result(scan)

    fs.assert_called_once_with(_WEBHOOK, scan)
    ws.assert_not_called()
    assert result == {"feishu": True}


def test_only_wecom_enabled_calls_wecom_only():
    from notify import dispatcher
    scan = {"top": [], "total": 0, "skipped": 0, "thresh_strong": 0.6, "thresh_watch": 0.5}

    with patch.object(dispatcher, "_load_config", return_value=_cfg(_WECOM_ONLY)):
        with patch("notify.feishu.send_scan_result", return_value=True) as fs:
            with patch("notify.wecom.send_scan_result", return_value=True) as ws:
                result = dispatcher.send_scan_result(scan)

    fs.assert_not_called()
    ws.assert_called_once_with(scan)
    assert result == {"wecom": True}


def test_both_enabled_calls_both():
    from notify import dispatcher
    scan = {"top": [], "total": 0, "skipped": 0, "thresh_strong": 0.6, "thresh_watch": 0.5}

    with patch.object(dispatcher, "_load_config", return_value=_cfg(_BOTH)):
        with patch("notify.feishu.send_scan_result", return_value=True):
            with patch("notify.wecom.send_scan_result", return_value=True):
                result = dispatcher.send_scan_result(scan)

    assert "feishu" in result
    assert "wecom" in result


def test_no_channels_returns_empty():
    from notify import dispatcher
    scan = {"top": [], "total": 0, "skipped": 0, "thresh_strong": 0.6, "thresh_watch": 0.5}

    with patch.object(dispatcher, "_load_config", return_value=_cfg(_NONE)):
        result = dispatcher.send_scan_result(scan)

    assert result == {}


# ── 失败隔离 ───────────────────────────────────────────────────────────────────

def test_feishu_failure_does_not_block_wecom():
    from notify import dispatcher
    scan = {"top": [], "total": 0, "skipped": 0, "thresh_strong": 0.6, "thresh_watch": 0.5}

    with patch.object(dispatcher, "_load_config", return_value=_cfg(_BOTH)):
        with patch("notify.feishu.send_scan_result", side_effect=RuntimeError("network error")):
            with patch("notify.wecom.send_scan_result", return_value=True):
                result = dispatcher.send_scan_result(scan)

    assert result["feishu"] is False
    assert result["wecom"] is True


def test_wecom_failure_does_not_block_feishu():
    from notify import dispatcher
    scan = {"top": [], "total": 0, "skipped": 0, "thresh_strong": 0.6, "thresh_watch": 0.5}

    with patch.object(dispatcher, "_load_config", return_value=_cfg(_BOTH)):
        with patch("notify.feishu.send_scan_result", return_value=True):
            with patch("notify.wecom.send_scan_result", side_effect=RuntimeError("wecom down")):
                result = dispatcher.send_scan_result(scan)

    assert result["feishu"] is True
    assert result["wecom"] is False


# ── send_text ──────────────────────────────────────────────────────────────────

def test_send_text_both_channels():
    from notify import dispatcher
    with patch.object(dispatcher, "_load_config", return_value=_cfg(_BOTH)):
        with patch("notify.feishu.send_text", return_value=True) as fs:
            with patch("notify.wecom.send_text", return_value=True) as ws:
                result = dispatcher.send_text("告警")

    fs.assert_called_once_with(_WEBHOOK, "告警")
    ws.assert_called_once_with("告警")
    assert result == {"feishu": True, "wecom": True}


# ── send_train_complete alias ──────────────────────────────────────────────────

def test_send_train_done_is_alias():
    from notify import dispatcher
    assert dispatcher.send_train_done is dispatcher.send_train_complete
