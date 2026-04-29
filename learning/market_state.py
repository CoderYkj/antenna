"""
market_state.py - 大盘状态打标(bull/bear/range),服务于所有 learner 的状态分桶。

算法(spec §7.1):
  bull   : close > ma60 且 60 日累计涨幅 > +5%,连续 3 日触发
  bear   : close < ma60 且 60 日累计跌幅 < -5%,连续 3 日触发
  range  : 其他;或 20 日 ATR/close > 2.5% 强制进入

产物:learning/market_state.json
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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
        "reason":  str | None,   # "insufficient_data" 等
      }
    """
    if hs300_df is None or len(hs300_df) < 61:
        return {"state": DEFAULT_STATE, "reason": "insufficient_data"}

    df = hs300_df.tail(90).copy().reset_index(drop=True)
    df["ma60"] = df["close"].rolling(60).mean()

    last_close   = float(df["close"].iloc[-1])
    last_ma60    = float(df["ma60"].iloc[-1])
    price_60_ago = float(df["close"].iloc[-61])
    ret_60d      = (last_close / price_60_ago - 1.0) if price_60_ago else 0.0

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


def _apply_debounce(
    *,
    raw_state: str,
    current: str,
    pending: str | None,
    pending_days: int,
    since: str | None,
    date_str: str,
) -> tuple[str, str | None, int, str | None]:
    """
    应用 3 日防抖,返回 (new_current, new_pending, new_pending_days, new_since)。

    规则见 update_state docstring。
    """
    first_time = since is None

    if first_time:
        return raw_state, None, 0, date_str
    if raw_state == current:
        return current, None, 0, since
    if raw_state == pending:
        new_pending_days = pending_days + 1
        if new_pending_days >= DEBOUNCE_DAYS:
            return pending, None, 0, date_str
        return current, pending, new_pending_days, since
    # raw_state 既不是 current 也不是 pending → 重新开始计数
    return current, raw_state, 1, since


def update_state(hs300_df: pd.DataFrame, date_str: str) -> dict:
    """
    基于最新 hs300_df 更新状态(含防抖);追加 history 一条并落盘。

    防抖规则(连续 DEBOUNCE_DAYS 天看到同一新状态才切换):
      - 首次(since=None)→ 立即设置 current=raw_state
      - raw_state == current → 维持,清空 pending
      - raw_state == pending → pending_days += 1;达到 DEBOUNCE_DAYS 时切换 current = pending
      - raw_state != current 也 != pending → pending = raw_state, pending_days = 1(从今天起重新计数)

    示例(DEBOUNCE_DAYS=3):
      Day1 bull(current=bull)
      Day2 bear → pending=bear, pending_days=1
      Day3 bear → pending_days=2
      Day4 bear → pending_days=3,触发切换 current=bear
      即看到 3 个连续 bear 信号后切换。
    """
    raw = compute_state(hs300_df)
    raw_state = raw["state"]
    data = load_current_state()

    new_current, new_pending, new_pending_days, new_since = _apply_debounce(
        raw_state=raw_state,
        current=data.get("current", DEFAULT_STATE),
        pending=data.get("pending"),
        pending_days=data.get("pending_days", 0),
        since=data.get("since"),
        date_str=date_str,
    )

    history = [h for h in (data.get("history") or []) if h.get("date") != date_str]
    history.append({"date": date_str, "state": new_current})
    history = history[-90:]

    hs300_metrics = {
        "close":   raw.get("close"),
        "ma60":    raw.get("ma60"),
        "ret_60d": raw.get("ret_60d"),
        "atr_pct": raw.get("atr_pct"),
    }
    if raw.get("reason"):
        hs300_metrics["reason"] = raw["reason"]

    new_data = {
        "current":      new_current,
        "since":        new_since,
        "pending":      new_pending,
        "pending_days": new_pending_days,
        "hs300":        hs300_metrics,
        "history":      history,
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write(STATE_FILE, new_data)
    return new_data


def _fetch_hs300(days: int = 120) -> pd.DataFrame:
    """拉沪深 300 指数日线(akshare 新浪指数接口),返回尾部 days 根 K 线。"""
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol="sh000300")
    if df is None or df.empty:
        return df
    return df.tail(days).reset_index(drop=True)


def run(date_str: str | None = None) -> dict:
    """编排器入口:拉沪深 300 数据并更新状态。失败抛异常,由编排器捕获。"""
    df = _fetch_hs300(days=120)
    if df is None or df.empty:
        raise RuntimeError("无法获取沪深300数据")

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    return update_state(df, date_str)
