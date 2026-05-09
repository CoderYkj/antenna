"""
backfill_market_state.py - 离线回填 198 天大盘状态序列。

Why:
  P1 校准器按 market_state(bull/bear/range) 分桶拟合,需要每个历史 outcome 日期
  有对应的 state 记录。当前 learning/market_state.json.history 只覆盖最近一周,
  早期 outcome 缺对应 state。

算法:
  1. 拉沪深 300 全量日线 K 线(akshare 新浪指数接口)
  2. 从 learning/data/outcome_*.jsonl 推导出所有交易日(就是有真实样本的日子)
  3. 按日期升序,对每日切片 close[:T+1],调 market_state.update_state(slice, T)
  4. update_state 内部带 3 日防抖 + 原子写,会自动追加 history 条目

注意:
  - 默认 --reset 行为:备份现有 market_state.json 到 .bak,从零重建 history
  - 30 天 checkpoint:每处理 30 天打印进度
  - 失败可续跑:--from YYYY-MM-DD 从指定日期接力
  - 不破坏 outcome 文件本身

用法:
  python scripts/backfill_market_state.py                    # 全量回填
  python scripts/backfill_market_state.py --reset            # 强制重建(默认行为)
  python scripts/backfill_market_state.py --from 2026-04-01  # 从某日续跑
  python scripts/backfill_market_state.py --dry-run          # 只统计不写
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATA_DIR = Path("learning/data")
STATE_FILE = Path("learning/market_state.json")


def list_outcome_dates(data_dir: Path = DATA_DIR) -> list[str]:
    """返回所有有 outcome 数据的日期(升序)。"""
    paths = sorted(data_dir.glob("outcome_*.jsonl"))
    return [p.stem.replace("outcome_", "") for p in paths]


def fetch_full_hs300() -> "pd.DataFrame":
    """拉沪深300全量日线。失败抛 RuntimeError。"""
    import akshare as ak  # 局部 import,允许 dry-run 时无 akshare 也能跑统计部分
    print("[INFO] 拉取沪深 300 全量日线(akshare 新浪指数接口)...")
    df = ak.stock_zh_index_daily(symbol="sh000300")
    if df is None or df.empty:
        raise RuntimeError("akshare 返回空数据")
    # 确保 date 列是 string YYYY-MM-DD
    df = df.copy()
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[INFO] 拉取成功: {len(df)} 行,首/末日期 {df['date'].iloc[0]} / {df['date'].iloc[-1]}")
    return df


def backup_current_state() -> Path | None:
    if not STATE_FILE.exists():
        return None
    bak = STATE_FILE.with_suffix(".json.bak")
    shutil.copy2(STATE_FILE, bak)
    print(f"[INFO] 已备份当前状态文件 → {bak}")
    return bak


def reset_state_file() -> None:
    """删除 market_state.json,update_state 第一次调用会创建初始结构。"""
    STATE_FILE.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量回填 198 天 market_state 序列")
    parser.add_argument("--from", dest="from_", metavar="YYYY-MM-DD",
                        help="起始日期(含);缺省从最早 outcome 日期")
    parser.add_argument("--to", dest="to_", metavar="YYYY-MM-DD",
                        help="结束日期(含);缺省到最近 outcome 日期")
    parser.add_argument("--reset", action="store_true", default=True,
                        help="强制重建 history(默认 True);备份原文件到 .bak")
    parser.add_argument("--no-reset", dest="reset", action="store_false",
                        help="不重建,从 history 最后一条之后续跑")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    args = parser.parse_args()

    # 1. 收集目标日期
    dates = list_outcome_dates()
    if args.from_:
        dates = [d for d in dates if d >= args.from_]
    if args.to_:
        dates = [d for d in dates if d <= args.to_]
    # 续跑模式:跳过 history 已有日期
    if not args.reset and STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            existing = {h["date"] for h in (data.get("history") or [])}
            before = len(dates)
            dates = [d for d in dates if d not in existing]
            print(f"[INFO] 续跑模式:跳过已有 {before - len(dates)} 个日期")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] 续跑读取 history 失败({e}),按 --reset 处理")
            args.reset = True

    if not dates:
        print("[INFO] 没有需要处理的日期")
        return 0

    print(f"[INFO] 目标 {len(dates)} 个日期({dates[0]} → {dates[-1]}),模式={'dry-run' if args.dry_run else '真实写入'}")

    if args.dry_run:
        print(f"[DONE] dry-run: 将处理 {len(dates)} 个日期(不写盘)")
        return 0

    # 2. 备份 + reset
    if args.reset:
        backup_current_state()
        reset_state_file()

    # 3. 拉沪深 300 全量
    try:
        hs300 = fetch_full_hs300()
    except Exception as e:
        print(f"[ERROR] 拉取沪深 300 失败: {type(e).__name__}: {e}")
        return 1

    # 4. 逐日 update_state
    from learning.market_state import update_state

    processed = 0
    failed = 0
    for i, date_str in enumerate(dates, 1):
        # 切片到该日及之前(含)的 K 线
        slice_df = hs300[hs300["date"] <= date_str]
        if len(slice_df) < 61:
            failed += 1
            continue
        try:
            update_state(slice_df, date_str)
            processed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {date_str}: {type(e).__name__}: {e}")
            continue
        if i % 30 == 0 or i == len(dates):
            print(f"  [{i}/{len(dates)}] 已处理 {processed} · 失败 {failed} · 最近: {date_str}")

    # 5. 终态校验
    final = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    print(f"[DONE] 处理 {processed} 天 · 失败 {failed} · 终态 history={len(final.get('history') or [])} 条 · current={final.get('current')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
