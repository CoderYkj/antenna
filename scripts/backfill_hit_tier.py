"""
backfill_hit_tier.py - 批量给 outcome_*.jsonl 回填 hit_tier 字段。

Why:
  P1 校准器训练样本需要 hit_tier ∈ {miss, weak, good, great}。
  早期 outcome 记录只有 actual_pct,没有 hit_tier 字段。
  本脚本按 outcome_metrics.compute_hit_tier 从 actual_pct 批量算出并原子覆盖写回。

特性:
  - 幂等:已有 hit_tier 的条目跳过不重算
  - 原子写:每个文件 tmp + os.replace
  - 范围筛:--from / --to 限定处理日期
  - 进度输出:每 20 个文件打印一次

用法:
  python scripts/backfill_hit_tier.py                         # 全量
  python scripts/backfill_hit_tier.py --from 2025-07-04       # 从某日起
  python scripts/backfill_hit_tier.py --from 2025-07-04 --to 2025-12-31
  python scripts/backfill_hit_tier.py --dry-run               # 只统计不写
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

# 路径 bootstrap,允许从任意目录运行
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from learning.outcome_metrics import compute_hit_tier

DATA_DIR = Path("learning/data")


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def process_file(path: Path, dry_run: bool = False) -> dict:
    """处理单个 outcome_*.jsonl,返回统计 dict。"""
    records: list[dict] = []
    added = 0
    already = 0
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("hit_tier") is not None:
                already += 1
                records.append(r)
                continue
            tier = compute_hit_tier(r.get("actual_pct"))
            if tier is not None:
                r["hit_tier"] = tier
                added += 1
            records.append(r)

    if added > 0 and not dry_run:
        _atomic_write_jsonl(path, records)

    return {"added": added, "already": already, "bad": bad, "total": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description="批量回填 outcome_*.jsonl 的 hit_tier 字段")
    parser.add_argument("--from", dest="from_", metavar="YYYY-MM-DD",
                        help="起始日期(含);缺省则处理全部")
    parser.add_argument("--to", dest="to_", metavar="YYYY-MM-DD",
                        help="结束日期(含);缺省则处理全部")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="outcome 目录(默认 learning/data)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return 1

    paths = sorted(data_dir.glob("outcome_*.jsonl"))
    if args.from_ or args.to_:
        def _in_range(p: Path) -> bool:
            d = p.stem.replace("outcome_", "")
            if args.from_ and d < args.from_:
                return False
            if args.to_ and d > args.to_:
                return False
            return True
        paths = [p for p in paths if _in_range(p)]

    if not paths:
        print("[INFO] 没有需要处理的 outcome 文件")
        return 0

    total_added = 0
    total_already = 0
    total_bad = 0
    total_files = len(paths)
    mode = "dry-run" if args.dry_run else "真实写入"
    print(f"[INFO] 共 {total_files} 个文件,模式={mode}")

    for i, p in enumerate(paths, 1):
        try:
            stat = process_file(p, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ✗ {p.name}: {type(e).__name__}: {e}")
            continue
        total_added += stat["added"]
        total_already += stat["already"]
        total_bad += stat["bad"]
        if i % 20 == 0 or i == total_files:
            print(f"  [{i}/{total_files}] 累计:新补 {total_added} · 跳过已有 {total_already} · 异常行 {total_bad}")

    print(f"[DONE] 文件={total_files} · 新补 hit_tier={total_added} · 已有={total_already} · bad={total_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
