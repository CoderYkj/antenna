"""
A股分析预测工具 CLI

用法:
  python cli.py fetch [--code 600519] [--days 365]
  python cli.py train
  python cli.py predict 600519
  python cli.py scan [--top 20]
  python cli.py report 600519
"""
import sys
import io

# Windows 终端默认 GBK，强制 UTF-8 输出避免中文乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import argparse
import sys
import webbrowser
from pathlib import Path

import yaml


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── fetch ──────────────────────────────────────────────────────────────────

def cmd_fetch(args, config):
    from data.fetcher import fetch_stock_hist, fetch_batch_parallel as fetch_all_parallel
    from data.universe import load_universe, get_all_codes

    codes = [args.code] if args.code else (
        get_all_codes() if args.all else load_universe(config)
    )
    days = args.days or config["data"]["default_days"]
    total = len(codes)

    if total <= 10:
        # 少量股票串行，实时显示进度
        for code in codes:
            print(f"  拉取 {code} ...")
            df = fetch_stock_hist(code, days=days)
            print(f"    {code}: {len(df)} 条，最新 {df['date'].max().date()}")
        print("完成。")
    else:
        # 大量股票并行拉取
        workers = args.workers if hasattr(args, "workers") and args.workers else 8
        print(f"  并行拉取 {total} 只股票（{workers} 线程）...")
        stats = fetch_all_parallel(codes, days=days, workers=workers)
        cached = stats.get("成功", 0) + stats.get("跳过", 0)
        print(f"  完成：成功 {stats['成功']}，跳过（已缓存）{stats['跳过']}，失败 {stats['失败']}")
        print(f"  本地共缓存 {cached} 只股票。")


# ── train ──────────────────────────────────────────────────────────────────

def cmd_train(args, config):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.trainer import build_labels, train, save_model

    codes = load_universe(config)
    days = config["data"]["default_days"]
    target_days = config["model"]["target_days"]
    threshold = config["model"]["threshold"]
    workers = getattr(args, "workers", None) or 8

    def _load_one(code):
        df = fetch_stock_hist(code, days=days, cache_only=True)
        df = build_features(df)
        df["label"] = build_labels(df, target_days=target_days, threshold=threshold)
        df["code"] = code
        return df

    dfs = []
    fail = 0
    total = len(codes)
    print(f"  并行加载 {total} 只股票（{workers} 线程）...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one, c): c for c in codes}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                dfs.append(future.result())
            except Exception:
                fail += 1
            if done % 500 == 0:
                print(f"  进度：{done}/{total}，已加载 {len(dfs)} 只，跳过 {fail} 只...")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  共加载 {len(dfs)} 只股票，跳过 {fail} 只，合计 {len(combined)} 行。")
    print(f"  Training on {len(combined)} rows ...")
    if getattr(args, "weighted", False):
        from learning.model_learner import retrain_with_weights
        print("  使用加权重训(P1):按历史 (signal, hit_tier) 查表加权")
        model = retrain_with_weights(combined, feature_cols=FEATURE_COLS)
    else:
        model = train(combined, feature_cols=FEATURE_COLS)
    save_model(model, saved_dir=config["model"]["saved_dir"])


# ── predict ────────────────────────────────────────────────────────────────

def cmd_predict(args, config):
    from data.fetcher import fetch_stock_hist
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict
    from reports.renderer import render_report

    df = fetch_stock_hist(args.code, days=365)
    df = build_features(df)
    result = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])

    print(f"\n{'='*40}")
    print(f"  股票代码:  {args.code}")
    print(f"  信号:      {result['signal']}")
    print(f"  涨概率:    {result['rise_prob']:.1%}")
    print(f"  跌概率:    {result['fall_prob']:.1%}")
    print(f"  置信度:    {result['confidence']}")
    print(f"{'='*40}\n")

    path = render_report(
        args.code, df.tail(120), result,
        output_dir=config["reports"]["output_dir"],
    )
    print(f"  报告: {path}")
    webbrowser.open(f"file:///{Path(path).resolve()}")


# ── scan ───────────────────────────────────────────────────────────────────

def cmd_scan(args, config):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import fetch_stock_hist, cached_codes
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import load_model, predict

    pool_cfg = config["universe"].get("scan_pool", "watchlist")
    if pool_cfg == "all":
        codes = cached_codes()
        print(f"  扫描模式：全 A 股（已缓存 {len(codes)} 只）")
    else:
        codes = load_universe(config)
        print(f"  扫描模式：自选股（{len(codes)} 只）")

    top_n = args.top
    workers = getattr(args, "workers", None) or 8
    model_dir = config["model"]["saved_dir"]

    # 预加载模型（所有线程共享，避免重复磁盘 IO）
    model = load_model(model_dir)

    results = []
    fail_count = 0
    lock = threading.Lock()

    def _scan_one(code):
        df = fetch_stock_hist(code, days=365, cache_only=True)
        df = build_features(df)
        r = predict(df, FEATURE_COLS, model=model)
        last = df.iloc[-1]
        momentum = (
            float(last.get("rsi6", 50) or 50) / 100
            + float(last.get("vol_ratio", 1) or 1) * 0.1
            + (1 if (last.get("macd_hist") or 0) > 0 else 0)
        )
        from features.analyser import predict_range
        price_info = predict_range(df, r["rise_prob"])
        return {"code": code, "momentum": momentum,
                "last": last.to_dict(), "price_info": price_info, **r}

    total = len(codes)
    done = 0
    print(f"  并行扫描 {total} 只（{workers} 线程）...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_one, c): c for c in codes}
        for future in as_completed(futures):
            done += 1
            try:
                results.append(future.result())
            except Exception:
                with lock:
                    fail_count += 1
            if done % 500 == 0:
                print(f"  进度：{done}/{total}，已入选 {len(results)} 只...")

    # 主排序：模型涨概率；同概率内按技术动量二次排序
    results.sort(key=lambda x: (x["rise_prob"], x["momentum"]), reverse=True)
    top = results[:top_n]

    # 基于全量结果分布动态计算阈值（Top 10% → 强势，Top 30% → 关注）
    all_probs = sorted([r["rise_prob"] for r in results], reverse=True)
    n = len(all_probs)
    thresh_strong = all_probs[max(0, int(n * 0.10) - 1)]  # 前 10%
    thresh_watch  = all_probs[max(0, int(n * 0.30) - 1)]  # 前 30%

    def _rel_signal(prob):
        if prob >= thresh_strong:
            return "强势"
        if prob >= thresh_watch:
            return "关注"
        return "观望"

    # 批量获取 Top N 股票实时行情，补充股票名称
    from data.fetcher import fetch_realtime_prices
    from features.analyser import suggest_trade_levels
    try:
        rt_map = fetch_realtime_prices([r["code"] for r in top])
    except Exception:
        rt_map = {}

    # 补充实时价格到 price_info，计算买卖建议
    for r in top:
        rt = rt_map.get(r["code"], {})
        if rt:
            r["price_info"].update(
                {k: rt[k] for k in ("price", "pct", "high", "low", "open") if k in rt}
            )
        r["trade"] = suggest_trade_levels(r["last"], r["price_info"], r["rise_prob"])

    print(f"\n{'='*64}")
    print(f"  Top {top_n} 候选股（模型概率+技术动量综合排序）  共扫描 {len(results)} 只，跳过 {fail_count} 只")
    print(f"  概率范围：{all_probs[-1]:.1%} ~ {all_probs[0]:.1%}  |  "
          f"强势线：{thresh_strong:.1%}  关注线：{thresh_watch:.1%}")
    print(f"{'='*64}")
    print(f"  {'代码':<10}{'名称':<12}{'信号':<8}{'涨概率':<10}{'动量分':<8}{'置信'}")
    print(f"  {'-'*68}")
    for r in top:
        sig  = _rel_signal(r["rise_prob"])
        name = rt_map.get(r["code"], {}).get("name", "")[:6]
        print(f"  {r['code']:<10}{name:<12}{sig:<8}{r['rise_prob']:.1%}      {r['momentum']:.2f}    {r['confidence']}")
    print()

    return {
        "top": [
            {**r,
             "signal": _rel_signal(r["rise_prob"]),
             "name":   rt_map.get(r["code"], {}).get("name", r["code"])}
            for r in top
        ],
        "total": len(results),
        "skipped": fail_count,
        "thresh_strong": thresh_strong,
        "thresh_watch": thresh_watch,
        "prob_min": all_probs[-1],
        "prob_max": all_probs[0],
    }


# ── report ─────────────────────────────────────────────────────────────────

def cmd_report(args, config):
    from data.fetcher import fetch_stock_hist
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict
    from reports.renderer import render_report

    df = fetch_stock_hist(args.code, days=365)
    df = build_features(df)
    result = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])
    path = render_report(
        args.code, df.tail(120), result,
        output_dir=config["reports"]["output_dir"],
    )
    print(f"Report: {path}")
    webbrowser.open(f"file:///{Path(path).resolve()}")


# ── learn ──────────────────────────────────────────────────────────────────

def cmd_learn(args, config):
    """
    learn:运行所有学习子模块(P0 仅含 market_state;P1-P4 阶段性追加)。

    用法:
      python cli.py learn                # 用今天日期
      python cli.py learn --date 2026-04-27
      python cli.py learn --dry-run
      python cli.py learn --check        # 仅做自检(不运行)
    """
    from datetime import datetime
    from learning import orchestrator

    if args.check:
        sys.exit(orchestrator.check())

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"[learn] 执行日期: {date_str}{' (dry-run)' if args.dry_run else ''}")

    results = orchestrator.run_all(date_str=date_str, dry_run=args.dry_run)

    ok      = sum(1 for r in results.values() if r["status"] == "ok")
    failed  = sum(1 for r in results.values() if r["status"] == "failed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    dry     = sum(1 for r in results.values() if r["status"] == "dry_run")

    print(f"[learn] 完成: ok={ok} failed={failed} skipped={skipped} dry_run={dry}")
    for name, r in results.items():
        print(f"  - {name}: {r['status']}")
        if r["status"] == "failed":
            print(f"    error: {r['error']}")

    sys.exit(1 if failed else 0)


# ── backtest ───────────────────────────────────────────────────────────────

def cmd_backtest(args, config):
    """
    Walk-Forward 历史回测。

    流程（逐日滚动）：
      对交易日 D：用截至 D 的历史数据运行扫描预测，写入 pred_{D+1}.jsonl
      对交易日 D+1：从缓存取 D+1 实际价格，写入 outcome_{D+1}.jsonl，
                    执行 evaluate_day + optimize，自动校正选股门槛
      循环至时段最后一个交易日。
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import cached_codes, fetch_stock_hist
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import load_model, predict
    from learning.tracker import log_predictions, log_outcomes
    from learning.optimizer import evaluate_day, optimize, load_strategy

    # ── 解析时间范围 ─────────────────────────────────────────
    if args.month:
        try:
            year, month = map(int, args.month.split("-"))
            start = pd.Timestamp(year=year, month=month, day=1)
            end   = start + pd.offsets.MonthEnd(1)
        except Exception:
            print("月份格式有误，请使用 YYYY-MM，如 2026-01")
            return
    elif args.year:
        try:
            year  = int(args.year)
            start = pd.Timestamp(year=year, month=1, day=1)
            end   = pd.Timestamp(year=year, month=12, day=31)
        except Exception:
            print("年份格式有误，请使用 YYYY，如 2026")
            return
    else:
        print("请指定 --month YYYY-MM 或 --year YYYY")
        return

    print(f"  回测区间：{start.date()} ～ {end.date()}")

    # ── 从缓存中取交易日列表（以首支有数据的股票为参考）─────
    codes = cached_codes()
    if not codes:
        print("  本地无缓存数据，请先执行 fetch 命令。")
        return

    ref_df = None
    for ref_code in codes[:10]:
        try:
            df = fetch_stock_hist(ref_code, days=1200, cache_only=True)
            if not df.empty:
                ref_df = df
                break
        except Exception:
            pass
    if ref_df is None:
        print("  无法读取参考数据。")
        return

    ref_df["date"] = pd.to_datetime(ref_df["date"])
    mask = (ref_df["date"] >= start) & (ref_df["date"] <= end)
    trading_days = sorted(ref_df[mask]["date"].tolist())
    if len(trading_days) < 2:
        print(f"  时段内交易日仅 {len(trading_days)} 天，数据不足，请先拉取该时段数据。")
        return

    print(f"  时段内共 {len(trading_days)} 个交易日，扫描股票 {len(codes)} 只")

    # ── 预加载模型 ────────────────────────────────────────────
    try:
        model = load_model(config["model"]["saved_dir"])
    except Exception as e:
        print(f"  模型加载失败：{e}")
        return

    top_n   = getattr(args, "top", 5)
    workers = getattr(args, "workers", 8)

    total_hits       = 0
    total_buy_sigs   = 0
    day_results      = []

    # ── 逐日滚动 ─────────────────────────────────────────────
    for i, trade_date in enumerate(trading_days[:-1]):
        pred_date  = trading_days[i + 1]
        trade_str  = trade_date.strftime("%Y-%m-%d")
        pred_str   = pred_date.strftime("%Y-%m-%d")

        print(f"\n  [{i+1:>3}/{len(trading_days)-1}] {trade_str} → 预测 {pred_str}", end="  ")

        strategy    = load_strategy()
        buy_top_pct = strategy.get("buy_top_pct", 0.10)

        # 对每只股票截断至 trade_date 并预测
        results = []

        def _predict_one(code, _date=trade_date):
            try:
                df = fetch_stock_hist(code, days=1200, cache_only=True)
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] <= _date].copy()
                if len(df) < 60:
                    return None
                df = build_features(df)
                if df.empty:
                    return None
                r    = predict(df, FEATURE_COLS, model=model, buy_top_pct=buy_top_pct)
                last = df.iloc[-1].to_dict()
                momentum = (
                    float(last.get("rsi6",     50) or 50) / 100
                    + float(last.get("vol_ratio", 1) or 1) * 0.1
                    + (1.0 if (last.get("macd_hist") or 0) > 0 else 0.0)
                )
                return {"code": code, "momentum": momentum, "last": last, **r}
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_predict_one, c): c for c in codes}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    results.append(r)

        if not results:
            print("无有效结果，跳过")
            continue

        results.sort(key=lambda x: (x["rise_prob"], x["momentum"]), reverse=True)
        buy_sigs = [r for r in results if r.get("signal") == "买入"]
        top      = (buy_sigs if buy_sigs else results)[:top_n]

        # 写预测快照
        snapshot = [{
            "code":       r["code"],
            "name":       r["code"],
            "signal":     r.get("signal", "观望"),
            "rise_prob":  round(r["rise_prob"], 4),
            "confidence": r.get("confidence", ""),
            "scan_date":  trade_str,
        } for r in top]
        log_predictions(pred_str, snapshot)

        # 取次日实际价格（从缓存）
        outcomes = {}
        for r in top:
            code = r["code"]
            try:
                df = fetch_stock_hist(code, days=1200, cache_only=True)
                df["date"] = pd.to_datetime(df["date"])
                row = df[df["date"] == pred_date]
                if not row.empty:
                    rx = row.iloc[0]
                    op = float(rx["open"])
                    cl = float(rx["close"])
                    outcomes[code] = {
                        "actual_open":  op,
                        "actual_close": cl,
                        "actual_high":  float(rx["high"]),
                        "actual_low":   float(rx["low"]),
                        "actual_pct":   round((cl / op - 1) * 100, 2) if op else 0,
                    }
            except Exception:
                pass
        if outcomes:
            log_outcomes(pred_str, outcomes)

        # 复盘 + 策略优化
        day_result = evaluate_day(pred_str)
        if day_result and day_result["total"] > 0:
            h   = day_result["hits"]
            t   = day_result["total"]
            acc = day_result["accuracy"]
            total_hits     += h
            total_buy_sigs += t
            day_results.append({"date": pred_str, "hits": h, "total": t, "accuracy": acc})
            print(f"买入 {t} 只　命中 {h}　精准率 {acc:.0%}", end="")
        else:
            print("无买入信号", end="")

        new_strategy, change = optimize(day_result)
        new_pct = new_strategy.get("buy_top_pct", 0.10)
        print(f"　门槛→{new_pct:.0%}")

    # ── 汇总报告 ─────────────────────────────────────────────
    overall = total_hits / total_buy_sigs if total_buy_sigs else 0
    print(f"\n  {'='*56}")
    print(f"  回测完成　{start.date()} ～ {end.date()}")
    print(f"  累计买入信号 {total_buy_sigs} 条　命中 {total_hits} 条")
    print(f"  整体买入精准率：{overall:.1%}")
    if day_results:
        best  = max(day_results, key=lambda x: x["accuracy"])
        worst = min(day_results, key=lambda x: x["accuracy"])
        print(f"  最佳单日：{best['date']} {best['accuracy']:.0%}（{best['hits']}/{best['total']}）")
        print(f"  最差单日：{worst['date']} {worst['accuracy']:.0%}（{worst['hits']}/{worst['total']}）")
    print(f"  最终选股门槛：{load_strategy().get('buy_top_pct', 0.10):.0%}")
    print(f"  {'='*56}")



def cmd_style(args, config):
    """
    style：从飞书聊天记录采集目标同事消息，提炼说话风格，写入 learning/persona.txt。

    用法：
      python cli.py style --find-user 方木木   # 按姓名搜索，获取 open_id
      python cli.py style --list-chats         # 列出所有 P2P 会话
      python cli.py style                      # 采集 + 分析 + 更新
      python cli.py style --dry-run            # 只分析，不写文件
      python cli.py style --days 90            # 覆盖采集天数
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from server.style_learner import run, print_p2p_chats, print_find_user

    if args.find_user:
        print_find_user(args.find_user)
        return

    if args.list_chats:
        print_p2p_chats()
        return

    days = args.days if args.days else None
    result = run(dry_run=args.dry_run, days=days)
    print(result)


# ── bot push ───────────────────────────────────────────────────────────────

def cmd_bot(args, config):
    """
    bot <指令>：执行任意指令并将结果推送到配置的飞书会话。

    示例：
      python cli.py bot 推荐
      python cli.py bot 推荐 3
      python cli.py bot 复盘
      python cli.py bot 行情 600519
      python cli.py bot 预测 002174
    """
    from server.commands import handle_command
    from server.feishu_push import push

    cmd_text = " ".join(args.cmd)
    if not cmd_text.strip():
        print("用法：python cli.py bot <指令>，如：python cli.py bot 推荐")
        return

    print(f"[bot] 执行指令：{cmd_text}")
    result = handle_command(cmd_text)

    print("[bot] 推送到飞书...")
    push(result)
    print("[bot] 推送完成。")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="antenna",
        description="A股分析&预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("fetch", help="拉取/更新历史数据")
    p.add_argument("--code", help="股票代码（不指定则更新自选列表）")
    p.add_argument("--all", action="store_true", help="拉取全部 A 股（约 5500 只）")
    p.add_argument("--days", type=int, help="历史天数（默认读取 config.yaml）")
    p.add_argument("--workers", type=int, default=8, help="并行线程数（默认 8）")

    p = sub.add_parser("train", help="训练预测模型")
    p.add_argument("--weighted", action="store_true",
                   help="P1 加权重训:按 (signal, hit_tier) 历史给样本加权(buy_miss=2.0 等)")
    p.add_argument("--workers", type=int, default=8, help="并行加载线程数(默认 8)")

    p = sub.add_parser("predict", help="预测单只股票并生成 HTML 报告")
    p.add_argument("code", help="股票代码，如 600519")

    p = sub.add_parser("scan", help="扫描股票池，输出 Top N 候选股")
    p.add_argument("--top", type=int, default=20, help="输出数量（默认 20）")
    p.add_argument("--workers", type=int, default=8, help="并行线程数（默认 8）")

    p = sub.add_parser("report", help="生成 HTML 报告（不重新预测）")
    p.add_argument("code", help="股票代码")

    p = sub.add_parser("backtest", help="Walk-Forward 历史回测（逐日预测→复盘→校正策略）")
    p.add_argument("--month",   help="回测月份，格式 YYYY-MM，如 2026-01")
    p.add_argument("--year",    help="回测年份，格式 YYYY，如 2026")
    p.add_argument("--top",     type=int, default=10, help="每日推荐 Top N（默认 10）")
    p.add_argument("--workers", type=int, default=8,  help="并行线程数（默认 8）")

    p = sub.add_parser("style", help="采集飞书同事聊天风格，更新方木木人设（需配置 colleague.p2p_chat_id 或 colleague.open_id）")
    p.add_argument("--list-chats",  action="store_true", help="列出 bot 所有 P2P 私聊及 chat_id")
    p.add_argument("--find-user",   metavar="NAME",       help="按姓名关键词搜索飞书用户，显示 open_id")
    p.add_argument("--dry-run",     action="store_true", help="只分析不写入 persona.txt")
    p.add_argument("--days",        type=int,            help="覆盖采集天数（默认读 config.yaml）")

    p = sub.add_parser("bot", help="执行指令并将结果推送到飞书（可配合定时任务使用）")
    p.add_argument("cmd", nargs="+", help="飞书指令，如：推荐  /  推荐 3  /  行情 600519")

    p = sub.add_parser("learn", help="运行学习反馈管线(P0 仅 market_state;后续阶段追加)")
    p.add_argument("--date", help="指定日期(默认今日),格式 YYYY-MM-DD")
    p.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p.add_argument("--check", action="store_true", help="仅校验学习产物文件的 JSON 合法性")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = _load_config()
    dispatch = {
        "fetch":    cmd_fetch,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "scan":     cmd_scan,
        "report":   cmd_report,
        "backtest": cmd_backtest,
        "style":    cmd_style,
        "bot":      cmd_bot,
        "learn":    cmd_learn,
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
