"""
check_alt_ic.py - 离线验证 alt 特征 IC 有效性。

读取 data/cache/alt/*.parquet 与 learning/data/outcome_*.jsonl，
对每个 alt 列按日计算 Spearman IC（alt 值 vs 5 日命中率 hit_5d），
汇总输出 IC 均值/标准差/有效天数/IC>0.02 占比，并给出保留建议。

用法：
    python scripts/check_alt_ic.py            # 分析全部可用日期
    python scripts/check_alt_ic.py --days 30  # 只看最近 30 天
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────
ALT_COLS = [
    "main_net_in_1d",
    "main_net_in_5d",
    "dragon_top_cnt_10d",
    "sector_heat_rank",
    "north_hold_chg_5d",
]
ALT_CACHE_DIR  = Path("data/cache/alt")
OUTCOME_DIR    = Path("learning/data")
MIN_SAMPLES    = 10   # 当天有效样本数下限，低于此则跳过当天
IC_KEEP_THRESH = 0.02 # IC 均值 >= 此值 → 建议保留


# ── 数据结构 ──────────────────────────────────────────────────

class DailyIC(NamedTuple):
    date: str
    col: str
    ic: float
    n: int


# ── IO 工具 ───────────────────────────────────────────────────

def _load_outcome(date_str: str) -> dict[str, float | None]:
    """读取 outcome_{date_str}.jsonl，返回 {code: actual_pct}。"""
    path = OUTCOME_DIR / f"outcome_{date_str}.jsonl"
    if not path.exists():
        return {}
    result: dict[str, float | None] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                code = rec.get("code")
                pct = rec.get("actual_pct")
                if code:
                    result[code] = float(pct) if pct is not None else None
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return result


def _load_alt(parquet_path: Path) -> pd.DataFrame:
    """读取 alt parquet，返回含 code + ALT_COLS 的 DataFrame。"""
    try:
        df = pd.read_parquet(parquet_path)
        present_cols = [c for c in ALT_COLS if c in df.columns]
        keep_cols = ["code"] + present_cols
        return df[keep_cols].copy()
    except Exception as exc:
        logger.warning("读取 parquet 失败 %s: %s", parquet_path, exc)
        return pd.DataFrame()


# ── IC 计算 ────────────────────────────────────────────────────

def _compute_spearman_ic(series_x: pd.Series, series_y: pd.Series) -> float | None:
    """计算 Spearman IC，优先用 scipy，回退到 pandas。"""
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(series_x, series_y)
        return float(corr)
    except ImportError:
        return float(series_x.corr(series_y, method="spearman"))
    except Exception:
        return None


def _compute_daily_ics(date_str: str, alt_df: pd.DataFrame) -> list[DailyIC]:
    """对单日数据计算每个 alt 列的 Spearman IC（abs），返回 DailyIC 列表。"""
    outcome = _load_outcome(date_str)
    if not outcome:
        return []

    # 构建 hit_5d：actual_pct >= 2% 视为命中（与 feature_learner 一致）
    hit_rows = [
        {"code": code, "hit_5d": 1 if (pct is not None and pct >= 2.0) else 0}
        for code, pct in outcome.items()
        if pct is not None
    ]
    if not hit_rows:
        return []

    hit_df = pd.DataFrame(hit_rows)
    merged = alt_df.merge(hit_df, on="code", how="inner")

    results: list[DailyIC] = []
    for col in ALT_COLS:
        if col not in merged.columns:
            continue
        valid = merged[[col, "hit_5d"]].dropna()
        if len(valid) < MIN_SAMPLES:
            continue
        ic = _compute_spearman_ic(valid[col], valid["hit_5d"])
        if ic is None:
            continue
        results.append(DailyIC(date=date_str, col=col, ic=abs(ic), n=len(valid)))

    return results


# ── 汇总输出 ───────────────────────────────────────────────────

def _summarize(records: list[DailyIC]) -> pd.DataFrame:
    """将 DailyIC 列表汇总为每列统计 DataFrame。"""
    if not records:
        return pd.DataFrame(
            columns=["col", "ic_mean", "ic_std", "valid_days", "pct_above_thresh", "advice"]
        )

    df = pd.DataFrame(records, columns=DailyIC._fields)
    grouped = df.groupby("col")["ic"]
    summary = pd.DataFrame(
        {
            "ic_mean": grouped.mean(),
            "ic_std":  grouped.std().fillna(0.0),
            "valid_days": grouped.count(),
            "pct_above_thresh": grouped.apply(lambda s: (s >= IC_KEEP_THRESH).mean()),
        }
    ).reset_index()
    summary.rename(columns={"col": "feature"}, inplace=True)
    summary["advice"] = summary["ic_mean"].apply(
        lambda m: "建议保留" if m >= IC_KEEP_THRESH else "建议移除"
    )
    summary = summary.sort_values("ic_mean", ascending=False).reset_index(drop=True)
    return summary


def _print_summary(summary: pd.DataFrame, n_dates: int) -> None:
    """格式化打印汇总表。"""
    print()
    print(f"  分析日期数: {n_dates}  /  有效（含 alt 且含 outcome）日期数汇总如下")
    print()
    col_w = max(len(r) for r in summary["feature"]) + 2
    header = (
        f"{'特征':<{col_w}}  {'IC均值':>8}  {'IC标准差':>8}  "
        f"{'有效天数':>8}  {'IC>0.02占比':>10}  {'建议'}"
    )
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        print(
            f"{row['feature']:<{col_w}}  "
            f"{row['ic_mean']:>8.4f}  "
            f"{row['ic_std']:>8.4f}  "
            f"{int(row['valid_days']):>8d}  "
            f"{row['pct_above_thresh']:>10.1%}  "
            f"{row['advice']}"
        )
    print()


# ── 主入口 ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线验证 alt 特征 IC 有效性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="只分析最近 N 天（按 parquet 文件日期降序取前 N 个）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 扫描 alt parquet 文件 ────────────────────────────────
    parquet_files = sorted(ALT_CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        print("暂无 alt 缓存数据，请等待 task_scan 运行至少一天")
        sys.exit(0)

    if args.days is not None and args.days > 0:
        # 取日期最新的 N 个文件
        parquet_files = parquet_files[-args.days:]

    logger.info("找到 %d 个 alt parquet 文件，开始分析…", len(parquet_files))

    all_ics: list[DailyIC] = []
    skipped = 0

    for pq_path in parquet_files:
        date_str = pq_path.stem  # 文件名即日期，如 2026-05-21
        alt_df = _load_alt(pq_path)
        if alt_df.empty:
            skipped += 1
            continue

        daily = _compute_daily_ics(date_str, alt_df)
        if not daily:
            logger.debug("  %s: outcome 为空或有效样本不足，跳过", date_str)
            skipped += 1
        else:
            all_ics.extend(daily)
            logger.debug("  %s: 新增 %d 条 IC 记录", date_str, len(daily))

    n_analyzed = len(parquet_files) - skipped
    logger.info("分析完成：%d 天有有效 IC，%d 天跳过", n_analyzed, skipped)

    summary = _summarize(all_ics)

    if summary.empty:
        print()
        print("  没有足够数据计算 IC（alt 与 outcome 日期无交集，或每天有效样本均不足 10 条）")
        print(f"  共扫描 {len(parquet_files)} 个 parquet 文件，全部跳过。")
        print()
        sys.exit(0)

    _print_summary(summary, n_dates=len(parquet_files))


if __name__ == "__main__":
    main()
