"""Unit tests for notify/render.py"""
import pytest
from notify.render import feishu_to_wecom, _truncate


# ── feishu_to_wecom ───────────────────────────────────────────────────────────

def test_plain_string_passthrough():
    assert feishu_to_wecom("hello world") == "hello world"


def test_text_msg_type():
    payload = {"msg_type": "text", "content": {"text": "告警内容"}}
    assert feishu_to_wecom(payload) == "告警内容"


def test_interactive_card_title_and_markdown():
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "📡 扫描报告"}},
            "elements": [
                {"tag": "markdown", "content": "**茅台（600519）** | 强势"},
            ],
        },
    }
    result = feishu_to_wecom(payload)
    assert "**📡 扫描报告**" in result
    assert "**茅台（600519）**" in result


def test_interactive_card_hr_separator():
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "报告"}},
            "elements": [
                {"tag": "markdown", "content": "第一段"},
                {"tag": "hr"},
                {"tag": "markdown", "content": "第二段"},
            ],
        },
    }
    result = feishu_to_wecom(payload)
    assert "---" in result
    assert "第一段" in result
    assert "第二段" in result


def test_interactive_card_column_set():
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "标题"}},
            "elements": [
                {
                    "tag": "column_set",
                    "columns": [
                        {"elements": [{"tag": "markdown", "content": "左列"}]},
                        {"elements": [{"tag": "markdown", "content": "右列"}]},
                    ],
                }
            ],
        },
    }
    result = feishu_to_wecom(payload)
    assert "左列" in result
    assert "右列" in result
    assert " | " in result


def test_interactive_card_skips_img_elements():
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "标题"}},
            "elements": [
                {"tag": "img", "img_key": "img_xxx"},
                {"tag": "markdown", "content": "文字保留"},
            ],
        },
    }
    result = feishu_to_wecom(payload)
    assert "img_xxx" not in result
    assert "文字保留" in result


def test_unknown_msg_type_returns_string():
    payload = {"msg_type": "post", "content": {"zh_cn": {"title": "x"}}}
    result = feishu_to_wecom(payload)
    assert isinstance(result, str)
    assert len(result) > 0


# ── _truncate ─────────────────────────────────────────────────────────────────

def test_truncate_short_text_unchanged():
    text = "短文本"
    assert _truncate(text) == text


def test_truncate_long_text_ends_with_ellipsis():
    # 生成超过 4096 字节的文本
    long_text = "A" * 5000
    result = _truncate(long_text)
    assert result.endswith("…")
    assert len(result.encode("utf-8")) <= 4096


def test_truncate_multibyte_no_broken_encoding():
    # 确保截断时不产生乱码（中文每字 3 字节）
    long_text = "测试文本" * 400  # ~4800 字节
    result = _truncate(long_text)
    result.encode("utf-8")  # 不应抛出 UnicodeEncodeError
    assert result.endswith("…")
