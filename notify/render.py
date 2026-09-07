"""Feishu 消息载荷 → 企业微信 Markdown 字符串转换层。

Phase 3 的 bot 回复（server/commands.py 返回 Feishu interactive dict）
也经此模块转换后推送到企业微信。
"""

_WECOM_MAX_BYTES = 4096


def feishu_to_wecom(payload: dict | str) -> str:
    """将 Feishu 消息载荷转为企业微信 Markdown 字符串。

    支持：
    - str：原样返回（截断）
    - msg_type=text：取 content.text
    - msg_type=interactive：提取 card header + elements
    """
    if isinstance(payload, str):
        return _truncate(payload)

    msg_type = payload.get("msg_type", "")

    if msg_type == "text":
        text = payload.get("content", {}).get("text", "")
        return _truncate(text)

    if msg_type == "interactive":
        card = payload.get("card", {})
        parts: list[str] = []

        title = card.get("header", {}).get("title", {}).get("content", "")
        if title:
            parts.append(f"**{title}**")

        for el in card.get("elements", []):
            tag = el.get("tag", "")
            if tag == "markdown":
                content = el.get("content", "").strip()
                if content:
                    parts.append(content)
            elif tag == "hr":
                parts.append("---")
            elif tag == "column_set":
                col_texts = []
                for col in el.get("columns", []):
                    for cel in col.get("elements", []):
                        if cel.get("tag") == "markdown":
                            col_texts.append(cel.get("content", "").strip())
                if col_texts:
                    parts.append(" | ".join(col_texts))

        return _truncate("\n\n".join(p for p in parts if p))

    return _truncate(str(payload))


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= _WECOM_MAX_BYTES:
        return text
    return encoded[:_WECOM_MAX_BYTES - 3].decode("utf-8", errors="ignore") + "…"
