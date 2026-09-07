"""data/fin_cache.py — 财务数据轻量磁盘缓存，供扫描候选池预筛使用。

TTL: 30 天（财报按季度更新，30 天内不重复拉取）
存储: data/fin_cache.json
键: 股票代码(str)
值: {roe, debt_ratio, total_score, rev_growth, profit_growth, gross_margin, cached_at}
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

_PATH = Path("data/fin_cache.json")
_LOCK = threading.Lock()
_TTL_DAYS = 30

# 预筛权重：与战法阈值对齐
_ROE_GOOD   = 8.0
_DEBT_OK    = 50.0
_DEBT_LOOSE = 65.0
_SCORE_GOOD = 2.0
_SCORE_NEU  = 0.0
_SCORE_WEAK = -3.0
_REVGR_GOOD = 15.0


def _load() -> dict:
    if not _PATH.exists():
        return {}
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dump(data: dict) -> None:
    _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=None),
                     encoding="utf-8")


def get(code: str) -> dict | None:
    """返回有效缓存，过期或缺失返回 None。"""
    cache = _load()
    entry = cache.get(code)
    if not entry:
        return None
    try:
        age = datetime.now() - datetime.fromisoformat(entry.get("cached_at", ""))
        if age > timedelta(days=_TTL_DAYS):
            return None
    except Exception:
        return None
    return entry


def put_batch(updates: dict[str, dict]) -> None:
    """批量写入财务摘要（线程安全）。updates: {code: fin_res_dict}"""
    if not updates:
        return
    ts = datetime.now().isoformat(timespec="seconds")
    with _LOCK:
        cache = _load()
        for code, fin_res in updates.items():
            if not fin_res:
                continue
            cache[code] = {
                "roe":            fin_res.get("roe"),
                "debt_ratio":     fin_res.get("debt_ratio"),
                "total_score":    fin_res.get("total_score"),
                "rev_growth":     fin_res.get("rev_growth"),
                "profit_growth":  fin_res.get("profit_growth"),
                "gross_margin":   fin_res.get("gross_margin"),
                "cached_at":      ts,
            }
        _dump(cache)


def score(entry: dict) -> float:
    """根据缓存财务指标计算预筛分（越高越可能命中战法）。
    满分约 8.0；< 0 表示基本面较差，优先移出候选池。
    """
    roe   = entry.get("roe")   or 0.0
    debt  = entry.get("debt_ratio") or 100.0
    total = entry.get("total_score") or 0.0
    rev   = entry.get("rev_growth")  or 0.0

    s = 0.0
    # ROE 档位
    if roe > _ROE_GOOD:   s += 2.0
    elif roe > 0:         s += 1.0
    else:                 s -= 1.0     # 亏损扣分
    # 负债率档位
    if debt < _DEBT_OK:   s += 2.0
    elif debt < _DEBT_LOOSE: s += 1.0
    else:                 s -= 0.5
    # 财务总分档位
    if total >= _SCORE_GOOD:  s += 2.0
    elif total >= _SCORE_NEU: s += 1.0
    elif total >= _SCORE_WEAK: s += 0.0
    else:                      s -= 1.0  # 财务恶化
    # 营收增速
    if rev > _REVGR_GOOD: s += 1.5
    elif rev > 0:         s += 0.5

    return s
