"""
outcome_metrics.py - outcome 扩展字段计算工具。

hit_tier: 按次日涨幅分档
  miss  : actual_pct < 1.0%      (未命中)
  weak  : 1.0% <= actual_pct < 2.0%
  good  : 2.0% <= actual_pct < 5.0%
  great : actual_pct >= 5.0%

5 日指标(第 6 个交易日回填):
  hit_5d:          5 日累计涨幅(%)
  max_drawdown_5d: 5 日内最大回撤(%),负值
"""
from typing import Sequence


def compute_hit_tier(actual_pct: float | None) -> str | None:
    if actual_pct is None:
        return None
    if actual_pct < 1.0:
        return "miss"
    if actual_pct < 2.0:
        return "weak"
    if actual_pct < 5.0:
        return "good"
    return "great"


def compute_5d_metrics(closes: Sequence[float]) -> dict | None:
    """
    closes: [day0_close, day1_close, ..., day4_close](pred 日为 day0,共 5 个交易日)

    返回:
      {"hit_5d": float, "max_drawdown_5d": float}
      数据不足(<5)时返回 None。
    """
    if not closes or len(closes) < 5:
        return None

    base = closes[0]
    if base == 0:
        return None

    # 累计涨幅
    cumulative = (closes[-1] / base - 1.0) * 100.0

    # 最大回撤:沿时间序列追踪 running peak,每点计算相对当前 peak 的回撤,取最小(最深)
    max_dd = 0.0
    running_peak = closes[0]
    for c in closes:
        if c > running_peak:
            running_peak = c
        dd = (c / running_peak - 1.0) * 100.0
        if dd < max_dd:
            max_dd = dd

    return {
        "hit_5d": round(cumulative, 2),
        "max_drawdown_5d": round(max_dd, 2),
    }
