"""
task_fill_5d_metrics.py - 每日 15:30 运行,回填 "今天往前推 5 交易日" 的 outcome 的
  hit_5d 和 max_drawdown_5d 字段。

触发时机:每日 task_daily_review.py 后调用一次。
"""
import sys
import io
import os
from datetime import datetime, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def fill_5d_metrics_for_date(pred_date_str: str) -> dict:
    """
    给定 pred 日期,回填 outcome 的 5 日指标。
    需要第 6 个交易日之后才能跑,否则数据不够。

    返回 {filled: int, skipped: int}。
    """
    import pandas as pd
    from learning.tracker import load_outcomes, log_outcomes
    from learning.outcome_metrics import compute_5d_metrics, compute_hit_tier
    from data.fetcher import fetch_stock_hist

    outcomes = load_outcomes(pred_date_str)
    if not outcomes:
        return {"filled": 0, "skipped": 0, "reason": "no outcomes"}

    pred_ts = pd.Timestamp(pred_date_str)
    filled = 0
    skipped = 0

    for code, r in outcomes.items():
        if r.get("hit_5d") is not None:
            skipped += 1
            continue

        try:
            df = fetch_stock_hist(code, days=30)
            if df is None or df.empty:
                skipped += 1
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= pred_ts].head(5)
            if len(df) < 5:
                skipped += 1
                continue
            closes = df["close"].tolist()
            metrics = compute_5d_metrics(closes)
            if metrics:
                r["hit_5d"] = metrics["hit_5d"]
                r["max_drawdown_5d"] = metrics["max_drawdown_5d"]
                # 顺便回填 hit_tier(若缺)
                if r.get("hit_tier") is None and r.get("actual_pct") is not None:
                    r["hit_tier"] = compute_hit_tier(r["actual_pct"])
                filled += 1
        except Exception:
            skipped += 1

    if filled:
        log_outcomes(pred_date_str, outcomes)

    return {"filled": filled, "skipped": skipped}


def run():
    # 取 7 日前的日期(给点 buffer),5 个交易日加上周末余量
    target = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"[fill_5d] 目标日期: {target}")
    result = fill_5d_metrics_for_date(target)
    print(f"[fill_5d] 回填 {result['filled']} 条,跳过 {result['skipped']} 条")


if __name__ == "__main__":
    run()
