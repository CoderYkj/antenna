"""tests/test_learn_notify.py — send_learn_complete 推送函数单元测试。"""
from unittest.mock import patch, MagicMock


def _make_results(ok=6, failed=0):
    statuses = ["ok"] * ok + ["failed"] * failed
    names = ["market_state", "model_learner", "tactic_learner",
             "blacklist", "feature_learner", "price_learner"][:ok + failed]
    results = {}
    for i, name in enumerate(names):
        status = statuses[i]
        results[name] = {"status": status, "error": "boom" if status == "failed" else None}
    return results


def test_send_learn_complete_all_ok():
    from notify.feishu import send_learn_complete
    with patch("notify.feishu._post", return_value=True) as mock_post:
        send_learn_complete("http://hook", _make_results(ok=6), elapsed_seconds=42)
    payload = mock_post.call_args[0][1]
    text = payload["content"]["text"]
    assert "✅" in text
    assert "ok=6" in text
    assert "42s" in text
    assert "market_state" in text


def test_send_learn_complete_with_failure():
    from notify.feishu import send_learn_complete
    with patch("notify.feishu._post", return_value=True) as mock_post:
        send_learn_complete("http://hook", _make_results(ok=5, failed=1), elapsed_seconds=10)
    text = mock_post.call_args[0][1]["content"]["text"]
    assert "❌" in text
    assert "failed=1" in text
    assert "boom" in text


def test_send_learn_complete_no_webhook_skips():
    from notify.feishu import send_learn_complete
    with patch("notify.feishu._post") as mock_post:
        result = send_learn_complete("", _make_results())
    assert result is False
    mock_post.assert_not_called()


def test_send_learn_complete_dry_run_not_called(tmp_path, monkeypatch):
    """cli.py learn --dry-run 不应触发飞书通知。"""
    import types, sys

    # 构造最小 args namespace
    args = types.SimpleNamespace(
        check=False,
        date=None,
        dry_run=True,
    )
    config = {"feishu": {"webhook_url": "http://hook"}}

    fake_results = {n: {"status": "dry_run", "error": None}
                    for n in ["market_state", "model_learner"]}

    with patch("learning.orchestrator.run_all", return_value=fake_results), \
         patch("notify.feishu.send_learn_complete") as mock_notify, \
         patch("sys.exit"):
        # import cli module fresh
        import importlib
        import cli as cli_mod
        importlib.reload(cli_mod)
        cli_mod.cmd_learn(args, config)

    mock_notify.assert_not_called()
