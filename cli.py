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
    from data.fetcher import fetch_stock_hist, fetch_all_parallel
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
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.trainer import build_labels, train, save_model

    codes = load_universe(config)
    days = config["data"]["default_days"]
    target_days = config["model"]["target_days"]
    threshold = config["model"]["threshold"]

    dfs = []
    for code in codes:
        print(f"  Loading {code} ...")
        df = fetch_stock_hist(code, days=days)
        df = build_features(df)
        # 必须在合并前按股票单独计算 label，否则 shift 会跨越股票边界
        df["label"] = build_labels(df, target_days=target_days, threshold=threshold)
        df["code"] = code
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Training on {len(combined)} rows ...")
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
    from data.fetcher import fetch_stock_hist, cached_codes
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict

    pool = config["universe"].get("scan_pool", "watchlist")
    if pool == "all":
        codes = cached_codes()
        print(f"  扫描模式：全 A 股（已缓存 {len(codes)} 只）")
    else:
        codes = load_universe(config)
        print(f"  扫描模式：自选股（{len(codes)} 只）")

    top_n = args.top
    results = []
    fail = 0

    for i, code in enumerate(codes, 1):
        try:
            df = fetch_stock_hist(code, days=365)
            df = build_features(df)
            r = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])
            results.append({"code": code, **r})
        except Exception:
            fail += 1
        if i % 100 == 0:
            print(f"  进度：{i}/{len(codes)}，已入选 {len(results)} 只...")

    results.sort(key=lambda x: x["rise_prob"], reverse=True)
    top = results[:top_n]

    print(f"\n{'='*54}")
    print(f"  Top {top_n} 候选股（按涨概率排序）  共扫描 {len(results)} 只，跳过 {fail} 只")
    print(f"{'='*54}")
    print(f"  {'代码':<10}{'信号':<8}{'涨概率':<10}置信度")
    print(f"  {'-'*50}")
    for r in top:
        print(f"  {r['code']:<10}{r['signal']:<8}{r['rise_prob']:.1%}      {r['confidence']}")
    print()


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


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="anatent",
        description="A股分析&预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("fetch", help="拉取/更新历史数据")
    p.add_argument("--code", help="股票代码（不指定则更新自选列表）")
    p.add_argument("--all", action="store_true", help="拉取全部 A 股（约 5500 只）")
    p.add_argument("--days", type=int, help="历史天数（默认读取 config.yaml）")
    p.add_argument("--workers", type=int, default=8, help="并行线程数（默认 8）")

    sub.add_parser("train", help="训练预测模型")

    p = sub.add_parser("predict", help="预测单只股票并生成 HTML 报告")
    p.add_argument("code", help="股票代码，如 600519")

    p = sub.add_parser("scan", help="扫描股票池，输出 Top N 候选股")
    p.add_argument("--top", type=int, default=20, help="输出数量（默认 20）")

    p = sub.add_parser("report", help="生成 HTML 报告（不重新预测）")
    p.add_argument("code", help="股票代码")

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
        "fetch": cmd_fetch,
        "train": cmd_train,
        "predict": cmd_predict,
        "scan": cmd_scan,
        "report": cmd_report,
        "learn": cmd_learn,
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
