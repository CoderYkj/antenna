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
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe

    codes = [args.code] if args.code else load_universe(config)
    days = args.days or config["data"]["default_days"]

    for code in codes:
        print(f"  Fetching {code} ...")
        df = fetch_stock_hist(code, days=days)
        print(f"    {code}: {len(df)} rows, latest {df['date'].max().date()}")
    print("Done.")


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
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict

    codes = load_universe(config)
    top_n = args.top

    results = []
    for code in codes:
        try:
            df = fetch_stock_hist(code, days=365)
            df = build_features(df)
            r = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])
            results.append({"code": code, **r})
            print(f"  {code}: {r['signal']} ({r['rise_prob']:.1%})")
        except Exception as e:
            print(f"  {code}: 跳过 — {e}")

    results.sort(key=lambda x: x["rise_prob"], reverse=True)
    top = results[:top_n]

    print(f"\n{'='*50}")
    print(f"  Top {top_n} 候选股（按涨概率排序）")
    print(f"{'='*50}")
    print(f"  {'代码':<10}{'信号':<8}{'涨概率':<10}置信度")
    print(f"  {'-'*46}")
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


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="anatent",
        description="A股分析&预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("fetch", help="拉取/更新历史数据")
    p.add_argument("--code", help="股票代码（不指定则更新全部关注列表）")
    p.add_argument("--days", type=int, help="历史天数（默认读取 config.yaml）")

    sub.add_parser("train", help="训练预测模型")

    p = sub.add_parser("predict", help="预测单只股票并生成 HTML 报告")
    p.add_argument("code", help="股票代码，如 600519")

    p = sub.add_parser("scan", help="扫描股票池，输出 Top N 候选股")
    p.add_argument("--top", type=int, default=20, help="输出数量（默认 20）")

    p = sub.add_parser("report", help="生成 HTML 报告（不重新预测）")
    p.add_argument("code", help="股票代码")

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
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
