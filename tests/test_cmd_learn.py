"""tests/test_cmd_learn.py — 飞书"学习"指令测试。

cmd_learn 职责:
  - 解析参数: None / "dry-run" / "YYYY-MM-DD" / 非法
  - 返回"开始"同步消息(纯文本)
  - 后台线程跑 orchestrator.run_all,完成后推送"完成/失败/异常"消息

测试策略:
  - _run_learn_async 是同步函数,单独测
  - cmd_learn 仅测参数解析 + 返回值 + 线程启动(不测线程调度)
"""
from unittest.mock import patch, MagicMock

import pytest


# ── _run_learn_async 同步行为 ───────────────────────────────

def test_run_async_ok_pushes_success_card():
    """全部模块 ok → 推送 ✅ 开头、含耗时、含 ok=1 failed=0。"""
    from server.predict_cmd import _run_learn_async

    fake_results = {"market_state": {"status": "ok", "result": {"current": "range"}}}
    with patch("server.predict_cmd.run_all", return_value=fake_results) as m_run, \
         patch("server.predict_cmd.push") as m_push:
        _run_learn_async(date_str="2026-04-29", dry_run=False,
                         mode_text="真实执行", date_display="2026-04-29",
                         chat_id="oc_xxx")

    m_run.assert_called_once_with(date_str="2026-04-29", dry_run=False)
    m_push.assert_called_once()
    msg, kwargs = m_push.call_args[0][0], m_push.call_args[1]
    assert isinstance(msg, str)
    assert msg.startswith("✅")
    assert "学习真实执行完成" in msg
    assert "ok=1" in msg
    assert "failed=0" in msg
    assert "✅ market_state" in msg
    assert kwargs == {"chat_ids": ["oc_xxx"]}


def test_run_async_failure_pushes_warning_card():
    """有模块 failed → 推送 ⚠️ + 错误信息。"""
    from server.predict_cmd import _run_learn_async

    fake_results = {"market_state": {"status": "failed", "error": "KeyError('date')"}}
    with patch("server.predict_cmd.run_all", return_value=fake_results), \
         patch("server.predict_cmd.push") as m_push:
        _run_learn_async(date_str=None, dry_run=False,
                         mode_text="真实执行", date_display="2026-04-29",
                         chat_id="oc_xxx")

    msg = m_push.call_args[0][0]
    assert msg.startswith("⚠️")
    assert "ok=0" in msg
    assert "failed=1" in msg
    assert "❌ market_state" in msg
    assert "KeyError" in msg


def test_run_async_exception_pushes_error_card():
    """orchestrator 整体抛异常 → 推送 ❌ 学习异常。"""
    from server.predict_cmd import _run_learn_async

    with patch("server.predict_cmd.run_all", side_effect=ImportError("boom")), \
         patch("server.predict_cmd.push") as m_push:
        _run_learn_async(date_str=None, dry_run=False,
                         mode_text="真实执行", date_display="2026-04-29",
                         chat_id="oc_xxx")

    msg = m_push.call_args[0][0]
    assert msg.startswith("❌")
    assert "学习异常" in msg
    assert "ImportError" in msg
    assert "boom" in msg


def test_run_async_dry_run_marks_items():
    """dry-run → 所有模块用 📝 标记,mode_text 含'演练'。"""
    from server.predict_cmd import _run_learn_async

    fake_results = {"market_state": {"status": "dry_run"}}
    with patch("server.predict_cmd.run_all", return_value=fake_results), \
         patch("server.predict_cmd.push") as m_push:
        _run_learn_async(date_str=None, dry_run=True,
                         mode_text="演练", date_display="2026-04-29",
                         chat_id="oc_xxx")

    msg = m_push.call_args[0][0]
    assert "学习演练完成" in msg
    assert "📝 market_state" in msg


def test_run_async_no_chat_id_skips_push():
    """chat_id 为空 → 不推送(防止误发给 config 所有群)。"""
    from server.predict_cmd import _run_learn_async

    with patch("server.predict_cmd.run_all", return_value={}), \
         patch("server.predict_cmd.push") as m_push:
        _run_learn_async(date_str=None, dry_run=False,
                         mode_text="真实执行", date_display="2026-04-29",
                         chat_id="")

    m_push.assert_not_called()


# ── cmd_learn 参数解析与异步启动 ────────────────────────────

def test_cmd_learn_no_arg_returns_start_message_and_starts_thread():
    """无参 → 返回"开始"消息,启动线程。"""
    from server.predict_cmd import cmd_learn

    fake_thread = MagicMock()
    with patch("server.predict_cmd.threading.Thread", return_value=fake_thread) as m_thread:
        reply = cmd_learn(None, chat_id="oc_xxx")

    assert isinstance(reply, str)
    assert reply.startswith("🤖")
    assert "开始学习" in reply
    assert "真实执行" in reply
    m_thread.assert_called_once()
    fake_thread.start.assert_called_once()


def test_cmd_learn_dry_run_arg():
    """dry-run 参数 → mode_text='演练' + dry_run=True 传入 async。"""
    from server.predict_cmd import cmd_learn

    with patch("server.predict_cmd.threading.Thread") as m_thread:
        reply = cmd_learn("dry-run", chat_id="oc_xxx")

    assert "演练" in reply
    # 检查 Thread 的 kwargs
    kwargs = m_thread.call_args.kwargs
    assert kwargs["kwargs"]["dry_run"] is True


def test_cmd_learn_date_arg():
    """传入 YYYY-MM-DD → date_str 透传给 async。"""
    from server.predict_cmd import cmd_learn

    with patch("server.predict_cmd.threading.Thread") as m_thread:
        reply = cmd_learn("2026-04-28", chat_id="oc_xxx")

    assert "2026-04-28" in reply
    kwargs = m_thread.call_args.kwargs
    assert kwargs["kwargs"]["date_str"] == "2026-04-28"


def test_cmd_learn_invalid_arg_returns_usage_hint():
    """非法参数 → 返回用法提示,不启动线程。"""
    from server.predict_cmd import cmd_learn

    with patch("server.predict_cmd.threading.Thread") as m_thread:
        reply = cmd_learn("xyz", chat_id="oc_xxx")

    assert "用法" in reply
    assert "xyz" in reply
    m_thread.assert_not_called()


def test_cmd_learn_accepts_chinese_dry_run():
    """中文'演练' 也算 dry-run。"""
    from server.predict_cmd import cmd_learn

    with patch("server.predict_cmd.threading.Thread") as m_thread:
        reply = cmd_learn("演练", chat_id="oc_xxx")

    assert "演练" in reply
    kwargs = m_thread.call_args.kwargs
    assert kwargs["kwargs"]["dry_run"] is True


# ── 指令路由集成 ───────────────────────────────────────────

@pytest.mark.parametrize("cmd", ["学习", "自学习", "learn", "l"])
def test_handle_command_routes_learn_words(cmd):
    """LEARN_WORDS 四种写法都路由到 cmd_learn。"""
    from server.commands import handle_command

    with patch("server.predict_cmd.cmd_learn", return_value="🤖 started") as m:
        reply = handle_command(f"{cmd}", chat_id="oc_xxx")

    assert reply == "🤖 started"
    m.assert_called_once()


def test_handle_command_learn_with_date():
    """'学习 2026-04-28' → cmd_learn('2026-04-28', chat_id=...)。"""
    from server.commands import handle_command

    with patch("server.predict_cmd.cmd_learn", return_value="🤖") as m:
        handle_command("学习 2026-04-28", chat_id="oc_xxx")

    m.assert_called_once_with("2026-04-28", "oc_xxx")


# ── 并发槽位注册 ───────────────────────────────────────────

@pytest.mark.parametrize("word", ["学习", "自学习", "learn", "l"])
def test_learn_is_registered_as_heavy(word):
    """学习指令必须加入 _ALL_HEAVY,受并发槽位限制。"""
    from server.feishu_poll import _ALL_HEAVY

    assert word in _ALL_HEAVY
