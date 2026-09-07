"""Unit tests for notify/wecom.py"""
from unittest.mock import MagicMock, patch
import json
import pytest


# ── _get_access_token ─────────────────────────────────────────────────────────

def test_get_access_token_cached(monkeypatch):
    """缓存命中时不发请求。"""
    import notify.wecom as wm
    import time
    wm._TOKEN_CACHE["token"] = "cached_token"
    wm._TOKEN_CACHE["expires_at"] = time.time() + 3600
    with patch("urllib.request.urlopen") as mock_url:
        token = wm._get_access_token("corp", "secret")
    assert token == "cached_token"
    mock_url.assert_not_called()


def test_get_access_token_fetches_on_miss(monkeypatch):
    """缓存过期时重新请求。"""
    import notify.wecom as wm
    wm._TOKEN_CACHE["token"] = ""
    wm._TOKEN_CACHE["expires_at"] = 0.0

    resp_body = json.dumps({"errcode": 0, "access_token": "new_token", "expires_in": 7200}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = resp_body

    with patch("urllib.request.urlopen", return_value=mock_resp):
        token = wm._get_access_token("corp", "secret")

    assert token == "new_token"
    assert wm._TOKEN_CACHE["token"] == "new_token"


def test_get_access_token_returns_empty_on_api_error():
    import notify.wecom as wm
    wm._TOKEN_CACHE["token"] = ""
    wm._TOKEN_CACHE["expires_at"] = 0.0

    resp_body = json.dumps({"errcode": 40013, "errmsg": "invalid corpid"}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = resp_body

    with patch("urllib.request.urlopen", return_value=mock_resp):
        token = wm._get_access_token("bad_corp", "secret")

    assert token == ""


# ── _post 配置检查 ─────────────────────────────────────────────────────────────

def test_post_skips_when_disabled(monkeypatch):
    """channels.wecom=False 时 _post 不发请求。"""
    import notify.wecom as wm
    with patch.object(wm, "_load_cfg", return_value=(False, {})):
        with patch("urllib.request.urlopen") as mock_url:
            result = wm._post({"msgtype": "text", "text": {"content": "test"}})
    assert result is False
    mock_url.assert_not_called()


def test_post_skips_when_credentials_missing(monkeypatch):
    import notify.wecom as wm
    with patch.object(wm, "_load_cfg", return_value=(True, {"corp_id": "", "secret": "", "agent_id": 0})):
        with patch("urllib.request.urlopen") as mock_url:
            result = wm._post({"msgtype": "text", "text": {"content": "test"}})
    assert result is False
    mock_url.assert_not_called()


def test_post_retries_on_token_expired(monkeypatch):
    """errcode=40014 时清缓存并重试一次。"""
    import notify.wecom as wm

    cfg = {"corp_id": "corpid", "secret": "secret", "agent_id": 1}
    with patch.object(wm, "_load_cfg", return_value=(True, cfg)):
        with patch.object(wm, "_get_access_token", return_value="tok"):
            # 第一次返回 40014，第二次返回 0
            responses = [
                json.dumps({"errcode": 40014, "errmsg": "token expired"}).encode(),
                json.dumps({"errcode": 0, "errmsg": "ok", "msgid": "1"}).encode(),
            ]
            call_count = [0]

            def fake_urlopen(req, timeout=10):
                m = MagicMock()
                m.__enter__ = lambda s: s
                m.__exit__ = MagicMock(return_value=False)
                m.read.return_value = responses[call_count[0]]
                call_count[0] += 1
                return m

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                result = wm._post({"msgtype": "text", "text": {"content": "hi"}})

    assert result is True
    assert call_count[0] == 2


# ── send_text ──────────────────────────────────────────────────────────────────

def test_send_text_calls_post():
    import notify.wecom as wm
    with patch.object(wm, "_post", return_value=True) as mock_post:
        result = wm.send_text("hello")
    assert result is True
    payload = mock_post.call_args[0][0]
    assert payload["msgtype"] == "text"
    assert payload["text"]["content"] == "hello"


# ── send_scan_result ───────────────────────────────────────────────────────────

def test_send_scan_result_markdown_contains_key_fields():
    import notify.wecom as wm
    scan_result = {
        "top": [{"code": "600519", "name": "贵州茅台", "signal": "强势",
                 "rise_prob": 0.72, "confidence": "高", "trade": {}}],
        "total": 100,
        "skipped": 5,
        "thresh_strong": 0.65,
        "thresh_watch": 0.55,
    }
    with patch.object(wm, "_post", return_value=True) as mock_post:
        wm.send_scan_result(scan_result)
    payload = mock_post.call_args[0][0]
    content = payload["markdown"]["content"]
    assert "贵州茅台" in content
    assert "600519" in content
    assert "72.0%" in content


# ── send_train_complete ────────────────────────────────────────────────────────

def test_send_train_complete_is_text():
    import notify.wecom as wm
    with patch.object(wm, "_post", return_value=True) as mock_post:
        wm.send_train_complete(elapsed_seconds=120)
    payload = mock_post.call_args[0][0]
    assert payload["msgtype"] == "text"
    assert "120s" in payload["text"]["content"]


# ── send_learn_complete ────────────────────────────────────────────────────────

def test_send_learn_complete_shows_failed():
    import notify.wecom as wm
    results = {
        "trainer": {"status": "ok", "error": None},
        "optimizer": {"status": "failed", "error": "timeout"},
    }
    with patch.object(wm, "_post", return_value=True) as mock_post:
        wm.send_learn_complete(results, elapsed_seconds=60)
    content = mock_post.call_args[0][0]["text"]["content"]
    assert "❌" in content
    assert "timeout" in content
