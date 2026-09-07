"""
盘中定时任务：交易时段每 15 min 对自选股预测并推送飞书。
调度窗口：09:30-11:30 / 13:00-15:00，仅交易日运行。

由 Windows 任务计划程序调用：
  python e:\\antenna\\scripts\\task_predict.py
"""
import sys
import io
import os
from datetime import datetime, time

# 强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import yaml

# ── 交易时段定义 ────────────────────────────────────────────
SESSIONS = [
    (time(9, 30), time(11, 30)),
    (time(13, 0), time(15, 0)),
]


def is_trading_time(now: datetime) -> bool:
    """判断 now 是否在交易时段内（含边界）。"""
    t = now.time()
    return any(start <= t <= end for start, end in SESSIONS)


def is_trading_day(dt: datetime) -> bool:
    """判断 dt 是否为 A 股交易日（排除周末+节假日）。"""
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XSHG")
        # exchange_calendars 要求 timezone-naive date
        ts = pd.Timestamp(dt.date())
        return cal.is_session(ts)
    except Exception as e:
        print(f"[task_predict] 节假日检查失败，降级为仅排除周末: {e}")
        return dt.weekday() < 5  # 0=Mon … 4=Fri


def run():
    now = datetime.now()

    if not is_trading_day(now):
        print(f"[task_predict] {now.date()} 非交易日，跳过。")
        return

    if not is_trading_time(now):
        print(f"[task_predict] {now.strftime('%H:%M')} 非交易时段，跳过。")
        return

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    codes = config["universe"]["watchlist"]
    model_dir = config["model"]["saved_dir"]

    from data.fetcher import fetch_stock_hist, fetch_realtime_prices, fetch_intraday_kline, _load_name_map
    from features.builder import build_features
    from features.technical import FEATURE_COLS, get_active_feature_cols
    from features.analyser import analyse, predict_range, text_intraday_kline
    from models.predictor import predict, load_model

    # P3 学习系统每周更新 feature_weights.json；用激活列表，回退全量特征
    try:
        active_feature_cols = get_active_feature_cols()
    except Exception as e:
        print(f"[task_predict] get_active_feature_cols 失败，回退 FEATURE_COLS: {e}")
        active_feature_cols = FEATURE_COLS

    print(f"[task_predict] {now.strftime('%H:%M')} 预测自选股 {len(codes)} 只 "
          f"(特征数={len(active_feature_cols)}) ...")
    model    = load_model(model_dir)
    name_map = _load_name_map()

    # 批量预取 alt_data（失败不阻断）
    today_str = now.strftime("%Y-%m-%d")
    alt_cache: dict = {}
    try:
        from data.alt_fetcher import fetch_alt_features
        alt_cache = fetch_alt_features(codes, today_str)
        print(f"[task_predict] alt_data 预取成功 {len(alt_cache)}/{len(codes)} 只")
    except Exception as _alt_err:
        print(f"[task_predict] alt_data 预取失败，继续无 alt 特征: {_alt_err}")

    # 批量获取所有自选股实时行情（一次请求）
    try:
        rt_prices = fetch_realtime_prices(codes)
        print(f"[task_predict] 实时行情获取成功 {len(rt_prices)}/{len(codes)} 只")
    except Exception as e:
        print(f"[task_predict] 实时行情批量获取失败: {e}，将降级使用历史收盘价")
        rt_prices = {}

    results = []
    for code in codes:
        try:
            df = fetch_stock_hist(code, days=365)
            alt_dict = alt_cache.get(code, {})
            df = build_features(df, alt=alt_dict)
            r = predict(df, active_feature_cols, model=model)
            last = df.iloc[-1].to_dict()
            r["reason"] = analyse(last)

            price_info = predict_range(df, r["rise_prob"])
            rt = rt_prices.get(code, {})
            if rt:
                price_info["price"]  = rt["price"]
                price_info["pct"]    = rt["pct"]
                price_info["high"]   = rt["high"]
                price_info["low"]    = rt["low"]
                price_info["open"]   = rt["open"]
            r["price"] = price_info
            r["name"]  = rt.get("name") or name_map.get(code, "") or code

            # 生成当日分时文字版
            df_intraday = fetch_intraday_kline(code, period_min=10)
            prev_close = float(df.iloc[-1]["close"]) if not df.empty else None
            r["text_kline"] = text_intraday_kline(
                df_intraday, code, name=r["name"], price_info=price_info
            )
            results.append({"code": code, **r})
            p = r["price"]
            cur = p.get("price") or p.get("close")
            print(f"  {code}: {r['signal']} 涨概率={r['rise_prob']:.1%}  "
                  f"现价={cur}({p.get('pct', 0):+.2f}%)  "
                  f"预测[{p.get('pred_low')}~{p.get('pred_high')}]  {r['reason']}")
        except Exception as e:
            print(f"  {code}: 失败 - {e}")

    if not results:
        print("[task_predict] 无有效结果，跳过推送。")
        return

    # ── 写入预测快照（供复盘使用）─────────────────────────
    # 只在每天 09:30-09:45 第一批写入，避免盘中信号漂移污染复盘基准
    if time(9, 30) <= now.time() <= time(9, 45):
        from learning.tracker import log_predictions
        from learning.optimizer import load_strategy
        buy_threshold = load_strategy().get("buy_threshold", 0.60)
        snapshot = []
        for res in results:
            pi = res.get("price", {})
            snapshot.append({
                "code":       res["code"],
                "name":       res.get("name", res["code"]),
                "signal":     "买入" if res["rise_prob"] >= buy_threshold else "观望",
                "rise_prob":  round(res["rise_prob"], 4),
                "confidence": res.get("confidence", ""),
                "pred_high":  pi.get("pred_high"),
                "pred_low":   pi.get("pred_low"),
                "open":       pi.get("open"),
                "scene":      "predict",
                "watchlist":  True,
            })
        log_predictions(now.strftime("%Y-%m-%d"), snapshot)
        print(f"[task_predict] 已写入 {len(snapshot)} 只自选股预测快照（复盘基准）")

    from notify import send_predict_results
    results_push = send_predict_results(results, now)
    ok = any(results_push.values()) if results_push else False
    print(f"[task_predict] 推送{'成功' if ok else '失败'}: {results_push}")


if __name__ == "__main__":
    run()
