"""
watchlist.py - 自选股增删查，线程安全地读写 config.yaml
"""
import threading
import json
import yaml

_lock = threading.Lock()
CONFIG_PATH = "config.yaml"


def _load() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_watchlist() -> list[str]:
    return _load()["universe"]["watchlist"]


def _stock_label(code: str) -> str:
    """返回 '名称（代码）' 格式，名称查不到时降级为代码。"""
    try:
        from data.fetcher import _load_name_map
        name = _load_name_map().get(code, "")
    except Exception:
        name = ""
    return f"{name}（{code}）" if name else code


def add_code(code: str) -> tuple[bool, str]:
    """添加股票代码，返回 (成功, 消息)。"""
    code = code.strip().zfill(6)
    label = _stock_label(code)
    with _lock:
        config = _load()
        wl = config["universe"]["watchlist"]
        if code in wl:
            return False, f"{label} 已在关注列表中"
        wl.append(code)
        _save(config)
    return True, f"✅ 已添加 {label}，当前关注 {len(wl)} 只"


def remove_code(code: str) -> tuple[bool, str]:
    """删除股票代码，返回 (成功, 消息)。"""
    code = code.strip().zfill(6)
    label = _stock_label(code)
    with _lock:
        config = _load()
        wl = config["universe"]["watchlist"]
        if code not in wl:
            return False, f"{label} 不在关注列表中"
        wl.remove(code)
        _save(config)
    return True, f"✅ 已删除 {label}，当前关注 {len(wl)} 只"


def list_codes():
    """返回自选股列表，格式为飞书 interactive 卡片 payload（dict）。"""
    wl = get_watchlist()
    if not wl:
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "📋 自选股列表（0 只）"},
                "template": "grey",
            },
            "elements": [{"tag": "markdown", "content": "关注列表为空，使用「添加 <代码>」添加股票。"}],
        }
        return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}

    try:
        from data.fetcher import fetch_realtime_prices
        rt = fetch_realtime_prices(wl)
    except Exception:
        rt = {}

    def _col(content: str, weight: int = 1) -> dict:
        return {
            "tag": "column",
            "width": "weighted",
            "weight": weight,
            "vertical_align": "center",
            "elements": [{"tag": "markdown", "content": content}],
        }

    def _row(*cols) -> dict:
        return {"tag": "column_set", "flex_mode": "none", "background_style": "default", "columns": list(cols)}

    elements = []

    # 表头
    elements.append(_row(
        _col("**股票**", 4),
        _col("**现价**", 2),
        _col("**涨跌幅**", 3),
    ))
    elements.append({"tag": "hr"})

    for code in wl:
        info  = rt.get(code, {})
        name  = info.get("name") or ""
        name  = name.replace("*", "＊").replace("_", "\\_")
        label = f"{name}（{code}）" if name else code
        price = info.get("price")
        pct   = info.get("pct")

        price_str = f"{price:.2f}" if price is not None else "盘外"

        if price is not None and pct is not None:
            icon  = "🔴" if pct > 0 else ("🟢" if pct < 0 else "⚪")
            arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
            pct_str = f"{icon} {arrow}{abs(pct):.2f}%"
        else:
            pct_str = "—"

        elements.append(_row(
            _col(label, 4),
            _col(f"**{price_str}**", 2),
            _col(pct_str, 3),
        ))

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📋 自选股列表（{len(wl)} 只）"},
            "template": "blue",
        },
        "elements": elements,
    }

    return {
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False),
    }
