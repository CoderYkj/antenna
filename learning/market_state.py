"""
market_state.py - 大盘状态打标(bull/bear/range),服务于所有 learner 的状态分桶。

算法(spec §7.1):
  bull   : close > ma60 且 60 日累计涨幅 > +5%,连续 3 日触发
  bear   : close < ma60 且 60 日累计跌幅 < -5%,连续 3 日触发
  range  : 其他;或 20 日 ATR/close > 2.5% 强制进入

产物:learning/market_state.json
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

STATE_FILE = Path("learning/market_state.json")
DEBOUNCE_DAYS = 3  # 连续 N 日触发才切换
DEFAULT_STATE = "range"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _classify(close: float, ma60: float, ret_60d: float, atr_pct: float) -> str:
    """纯分类函数,无防抖。"""
    if atr_pct > 0.025:
        return "range"
    if close > ma60 and ret_60d > 0.05:
        return "bull"
    if close < ma60 and ret_60d < -0.05:
        return "bear"
    return "range"


def compute_state(hs300_df: pd.DataFrame) -> dict:
    """
    根据沪深 300 DataFrame(含 close 列)计算当前原始状态(无防抖)。

    返回:
      {
        "state":   "bull"|"bear"|"range",
        "close":   float,
        "ma60":    float,
        "ret_60d": float,
        "atr_pct": float,
        "reason":  Optional[str],   # "insufficient_data" 等
      }
    """
    if hs300_df is None or len(hs300_df) < 60:
        return {"state": DEFAULT_STATE, "reason": "insufficient_data"}

    df = hs300_df.tail(90).copy().reset_index(drop=True)
    df["ma60"] = df["close"].rolling(60).mean()

    last_close = float(df["close"].iloc[-1])
    last_ma60 = float(df["ma60"].iloc[-1])
    if len(df) >= 60:
        price_60_ago = float(df["close"].iloc[-60])
    else:
        price_60_ago = float(df["close"].iloc[0])
    ret_60d = (last_close / price_60_ago - 1.0) if price_60_ago else 0.0

    hi = df["high"] if "high" in df.columns else df["close"]
    lo = df["low"] if "low" in df.columns else df["close"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (hi - lo).abs(),
        (hi - prev_close).abs(),
        (lo - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.tail(20).mean()
    atr_pct = (atr20 / last_close) if last_close else 0.0

    state = _classify(last_close, last_ma60, ret_60d, float(atr_pct))

    return {
        "state":   state,
        "close":   round(last_close, 2),
        "ma60":    round(last_ma60, 2),
        "ret_60d": round(float(ret_60d), 4),
        "atr_pct": round(float(atr_pct), 4),
    }


def load_current_state() -> dict:
    """加载当前持久化状态。不存在时返回默认。"""
    if not STATE_FILE.exists():
        return {"current": DEFAULT_STATE, "since": None, "pending": None, "history": []}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"current": DEFAULT_STATE, "since": None, "pending": None, "history": []}


def load_state_on_date(date_str: str) -> str:
    """
    根据 history 回查某日状态。若 date_str 之前有 history 记录则用最近的,否则返回 DEFAULT_STATE。
    用于回放测试:对每条 pred/outcome 查询当时市场状态。
    """
    data = load_current_state()
    history = data.get("history") or []
    matching = [h for h in history if h["date"] <= date_str]
    if not matching:
        return DEFAULT_STATE
    return matching[-1]["state"]


def update_state(hs300_df: pd.DataFrame, date_str: str) -> dict:
    """
    基于最新 hs300_df 更新状态(含防抖);追加 history 一条并落盘。

    防抖规则:
      - 若 raw_state == current:直接 append history,pending 清空
      - 若 raw_state != current 但 == pending:计数 +1;达到 DEBOUNCE_DAYS 切 current
      - 若 raw_state 不同于 current 也不同于 pending:pending = raw_state,计数重置
      - 首次(current 为 DEFAULT_STATE 且 since=None):允许立即设置
    """
    raw = compute_state(hs300_df)
    raw_state = raw["state"]
    data = load_current_state()

    current = data.get("current", DEFAULT_STATE)
    pending = data.get("pending")
    pending_days = data.get("pending_days", 0)

    first_time = data.get("since") is None

    if first_time:
        new_current = raw_state
        new_pending = None
        new_pending_days = 0
        new_since = date_str
    elif raw_state == current:
        new_current = current
        new_pending = None
        new_pending_days = 0
        new_since = data.get("since")
    elif raw_state == pending:
        new_pending_days = pending_days + 1
        if new_pending_days >= DEBOUNCE_DAYS:
            new_current = pending
            new_pending = None
            new_pending_days = 0
            new_since = date_str
        else:
            new_current = current
            new_pending = pending
            new_since = data.get("since")
    else:
        new_current = current
        new_pending = raw_state
        new_pending_days = 1
        new_since = data.get("since")

    history = data.get("history") or []
    history.append({"date": date_str, "state": new_current})
    history = history[-90:]  # 最近 90 日

    new_data = {
        "current":      new_current,
        "since":        new_since,
        "pending":      new_pending,
        "pending_days": new_pending_days,
        "hs300":        {k: v for k, v in raw.items() if k != "state"},
        "history":      history,
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write(STATE_FILE, new_data)
    return new_data


def run(date_str: Optional[str] = None) -> dict:
    """
    编排器入口:拉沪深 300 数据并更新状态。
    失败时抛异常,由编排器捕获。
    """
    from data.fetcher import fetch_stock_hist
    try:
        df = fetch_stock_hist("sh000300", days=120)
    except Exception:
        df = fetch_stock_hist("000300", days=120)

    if df is None or df.empty:
        raise RuntimeError("无法获取沪深300数据")

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    return update_state(df, date_str)
