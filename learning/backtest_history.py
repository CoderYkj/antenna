"""
backtest_history.py - 存储和读取历史回测摘要，供预测时作为参考依据。

文件：learning/backtest_results.json
结构：
  {
    "periods": [
      {
        "period":          "2026-01",        # YYYY-MM 或 YYYY
        "period_type":     "month",          # month / year
        "start":           "2026-01-02",
        "end":             "2026-01-31",
        "total_buy_sigs":  100,
        "total_hits":      72,
        "accuracy":        0.72,
        "trading_days":    21,
        "days_with_buys":  18,
        "best_day":        {"date": "...", "accuracy": 1.0, "hits": 5, "total": 5},
        "worst_day":       {"date": "...", "accuracy": 0.40, "hits": 2, "total": 5},
        "final_buy_top_pct": 0.08,
        "completed_at":    "2026-04-14T12:00:00"
      }
    ]
  }
"""
import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path("learning/backtest_results.json")


def load_backtest_history() -> list[dict]:
    """返回所有历史回测摘要列表（按时间升序）。"""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f).get("periods", [])
    except Exception:
        return []


def save_backtest_period(period: str, period_type: str, start: str, end: str,
                         total_buy_sigs: int, total_hits: int,
                         day_results: list[dict], final_buy_top_pct: float):
    """将一次回测结果追加到历史文件。"""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    periods = load_backtest_history()

    accuracy = round(total_hits / total_buy_sigs, 4) if total_buy_sigs else 0.0
    days_with_buys = len([d for d in day_results if d.get("total", 0) > 0])
    best  = max(day_results, key=lambda x: x["accuracy"]) if day_results else None
    worst = min(day_results, key=lambda x: x["accuracy"]) if day_results else None

    record = {
        "period":            period,
        "period_type":       period_type,
        "start":             start,
        "end":               end,
        "total_buy_sigs":    total_buy_sigs,
        "total_hits":        total_hits,
        "accuracy":          accuracy,
        "trading_days":      len(day_results) + 1,
        "days_with_buys":    days_with_buys,
        "best_day":          best,
        "worst_day":         worst,
        "final_buy_top_pct": final_buy_top_pct,
        "completed_at":      datetime.now().isoformat(),
    }

    # 同 period 覆盖旧记录
    periods = [p for p in periods if p.get("period") != period]
    periods.append(record)
    periods.sort(key=lambda x: x.get("period", ""))

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"periods": periods}, f, ensure_ascii=False, indent=2)

    return record


def get_backtest_context() -> dict:
    """
    返回供预测卡片使用的历史回测摘要：
      - 最近 3 个回测期的精准率
      - 整体趋势（上升/下降/平稳）
      - 最佳/最差期
    """
    periods = load_backtest_history()
    if not periods:
        return {}

    recent = periods[-3:]  # 最近 3 期

    # 趋势判断
    trend = "平稳"
    if len(recent) >= 2:
        delta = recent[-1]["accuracy"] - recent[0]["accuracy"]
        if delta > 0.05:
            trend = "上升 ↑"
        elif delta < -0.05:
            trend = "下降 ↓"

    best_period  = max(periods, key=lambda x: x["accuracy"])
    worst_period = min(periods, key=lambda x: x["accuracy"])

    return {
        "recent":       recent,
        "trend":        trend,
        "best_period":  best_period,
        "worst_period": worst_period,
        "total_periods": len(periods),
    }


def format_backtest_context_md() -> str:
    """
    生成适合嵌入飞书卡片的 markdown 文本，描述历史回测依据。
    空数据时返回空字符串。
    """
    ctx = get_backtest_context()
    if not ctx:
        return ""

    lines = ["**📜 历史回测依据**"]
    for p in ctx["recent"]:
        acc  = p["accuracy"]
        sigs = p["total_buy_sigs"]
        pct  = p.get("final_buy_top_pct", 0.10)
        icon = "✅" if acc >= 0.85 else ("⚠️" if acc >= 0.70 else "❌")
        lines.append(
            f"· **{p['period']}**　精准率 {acc:.1%} {icon}　"
            f"信号 {sigs} 条　收盘门槛 {pct:.0%}"
        )

    lines.append(
        f"近期趋势 **{ctx['trend']}**　"
        f"历史最佳 {ctx['best_period']['period']} **{ctx['best_period']['accuracy']:.1%}**"
    )
    return "\n".join(lines)
