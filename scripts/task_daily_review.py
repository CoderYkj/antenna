"""
task_daily_review.py - 每个交易日收盘后（15:30）自动执行：
  1. 拉取今日推荐股票的实际收盘数据
  2. 与盘前预测快照做命中率回测
  3. 调用优化器更新 buy_threshold
  4. 将回测报告推送至飞书

调度：Windows 任务计划程序，15:30 运行。
  python e:\\antenna\\scripts\\task_daily_review.py
"""
import sys
import io
import os
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml


def _load_cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_actual_outcomes(codes: list, date_str: str) -> dict:
    """
    拉取 codes 今日实际行情（开、收、高、低、涨幅）。
    优先用实时行情 API，收盘后价格即为收盘价。
    """
    from data.fetcher import fetch_realtime_prices, fetch_stock_hist
    import pandas as pd

    print(f"[review] 拉取 {len(codes)} 只实际收盘数据...")
    outcomes = {}

    # 先尝试实时行情（收盘后仍可获取当日数据）
    try:
        rt = fetch_realtime_prices(codes)
        for code, info in rt.items():
            outcomes[code] = {
                "actual_open":  info.get("open", 0),
                "actual_close": info["price"],
                "actual_high":  info.get("high", 0),
                "actual_low":   info.get("low", 0),
                "actual_pct":   info.get("pct", 0),
            }
        print(f"[review] 实时行情获取 {len(outcomes)} 只")
    except Exception as e:
        print(f"[review] 实时行情失败: {e}，降级用历史缓存")

    # 对拉不到的，从历史缓存补充今日数据
    missing = [c for c in codes if c not in outcomes]
    if missing:
        today = pd.Timestamp(date_str)
        for code in missing:
            try:
                df = fetch_stock_hist(code, days=5)
                row = df[df["date"] == today]
                if not row.empty:
                    r = row.iloc[0]
                    op = float(r["open"])
                    cl = float(r["close"])
                    outcomes[code] = {
                        "actual_open":  op,
                        "actual_close": cl,
                        "actual_high":  float(r["high"]),
                        "actual_low":   float(r["low"]),
                        "actual_pct":   round((cl / op - 1) * 100, 2) if op else 0,
                    }
            except Exception:
                pass

    return outcomes


def run():
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    cfg      = _load_cfg()

    from learning.tracker import load_predictions, log_outcomes
    from learning.optimizer import build_review_report

    # ── 检查今日是否有预测记录 ────────────────────────────
    preds = load_predictions(date_str)
    if not preds:
        print(f"[review] {date_str} 无预测记录，跳过回测。")
        return

    codes = list({p["code"] for p in preds})
    print(f"[review] {date_str} 共 {len(preds)} 条预测，{len(codes)} 只股票")

    # ── 拉取实际结果 ──────────────────────────────────────
    outcomes = fetch_actual_outcomes(codes, date_str)
    if not outcomes:
        print("[review] 无法获取实际收盘数据，跳过。")
        return

    # P0: 即时补 hit_tier(5 日指标由 task_fill_5d_metrics 延迟回填)
    from learning.outcome_metrics import compute_hit_tier
    for code, r in outcomes.items():
        r["hit_tier"] = compute_hit_tier(r.get("actual_pct"))

    log_outcomes(date_str, outcomes)
    print(f"[review] 已记录 {len(outcomes)} 只实际结果")

    # ── 回测 + 优化 ───────────────────────────────────────
    report = build_review_report(date_str)
    dr     = report["day_result"]
    if dr:
        print(f"[review] 今日命中率: {dr['accuracy']:.1%} ({dr['hits']}/{dr['total']})")
    print(f"[review] 近7日: {report['acc_7d']:.1%}  近30日: {report['acc_30d']:.1%}")
    print(f"[review] 策略调整: {report['change_desc']}")

    # ── 推送飞书 ──────────────────────────────────────────
    webhook = cfg.get("feishu", {}).get("webhook_url", "")
    if not webhook:
        print("[review] 未配置 webhook，跳过推送。")
        return

    from notify.feishu import send_review_report
    ok = send_review_report(webhook, report)
    print("[review] 飞书推送成功。" if ok else "[review] 飞书推送失败。")

    # ── 调用学习编排器 ──────────────────────────────────
    print("[review] 启动学习管线 ...")
    try:
        from learning.orchestrator import run_all
        results = run_all(date_str=date_str)
        ok_cnt  = sum(1 for r in results.values() if r["status"] == "ok")
        failed  = sum(1 for r in results.values() if r["status"] == "failed")
        print(f"[review] 学习管线完成: ok={ok_cnt} failed={failed}")
    except Exception as e:
        print(f"[review] 学习管线异常(忽略,不影响其他): {e}")

    # ── 5 日指标回填(对 7 日前的 outcome) ─────────────────
    try:
        from scripts.task_fill_5d_metrics import run as fill_5d_run
        fill_5d_run()
    except Exception as e:
        print(f"[review] 5 日指标回填异常(忽略): {e}")


if __name__ == "__main__":
    run()
