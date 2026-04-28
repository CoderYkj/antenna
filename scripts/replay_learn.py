"""
replay_learn.py - 在已有历史 pred/outcome 上模拟滚动学习,输出基线精准率曲线。

P0 阶段:因编排器只挂了 market_state(无实际学习效果),本脚本只做"**基线精准率计算**":
  对 2025-07-01 起每一天:
    1. load_predictions(D) / load_outcomes(D)
    2. 计算当日买入精准率(区分 scan / tactic:* / predict 各场景)
    3. 按市场状态(从 market_state.json history 查)打标
    4. 写入 learning/replay/baseline_YYYYMMDD_HHMMSS.csv

P1 起会扩展本脚本在每日循环中调 orchestrator.run_all(date_str=D),
  以此验证"学习参数更新后,下一日精准率是否提升"。P0 先打基线。

用法:
  python scripts/replay_learn.py                          # 回放全部
  python scripts/replay_learn.py --from 2026-01-01 --to 2026-04-27
  python scripts/replay_learn.py --scene tactic           # 只统计 tactic 场景
"""
import sys
import io
import os
import argparse
import csv
from collections import defaultdict
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_DIR = "learning/replay"


def _evaluate_single(preds: list, outcomes: dict, scene_filter: str | None = None) -> dict:
    """对一天的 pred + outcome,计算分场景的买入精准率。"""
    from learning.scene_bucket import bucket_by_scene, DEFAULT_SCENE

    if scene_filter:
        preds = [p for p in preds if (p.get("scene") or DEFAULT_SCENE).startswith(scene_filter)]

    by_scene = bucket_by_scene(preds)
    result = {}
    for scene, records in by_scene.items():
        buys = [r for r in records if r.get("signal") == "买入"]
        hits = 0
        total = 0
        for r in buys:
            out = outcomes.get(r["code"])
            if not out or out.get("actual_pct") is None:
                continue
            total += 1
            if out["actual_pct"] >= 1.0:
                hits += 1
        result[scene] = {
            "buys":     len(buys),
            "scored":   total,
            "hits":     hits,
            "accuracy": round(hits / total, 4) if total else 0,
        }
    return result


def run(from_date: str | None, to_date: str | None, scene_filter: str | None):
    from learning.tracker import list_prediction_dates, load_predictions, load_outcomes
    from learning.market_state import load_state_on_date

    all_dates = list_prediction_dates()
    if from_date:
        all_dates = [d for d in all_dates if d >= from_date]
    if to_date:
        all_dates = [d for d in all_dates if d <= to_date]

    if not all_dates:
        print("无可用预测数据")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"baseline_{run_ts}.csv")

    rows = []
    print(f"回放区间: {all_dates[0]} ~ {all_dates[-1]},共 {len(all_dates)} 天")

    for i, d in enumerate(all_dates):
        preds    = load_predictions(d)
        outcomes = load_outcomes(d)
        if not preds or not outcomes:
            continue

        state = load_state_on_date(d)
        per_scene = _evaluate_single(preds, outcomes, scene_filter)

        for scene, stats in per_scene.items():
            rows.append({
                "date":         d,
                "market_state": state,
                "scene":        scene,
                "buys":         stats["buys"],
                "scored":       stats["scored"],
                "hits":         stats["hits"],
                "accuracy":     stats["accuracy"],
            })

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(all_dates)} 天...")

    if not rows:
        print("无有效结果写入")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"基线精准率曲线已写入: {out_path}")
    print(f"总行数: {len(rows)}")

    # 汇总
    agg: dict[str, dict] = defaultdict(lambda: {"hits": 0, "scored": 0})
    for r in rows:
        key = r["scene"]
        agg[key]["hits"]   += r["hits"]
        agg[key]["scored"] += r["scored"]
    print("\n场景汇总(整个回放期间):")
    for scene, s in agg.items():
        acc = (s["hits"] / s["scored"]) if s["scored"] else 0
        print(f"  {scene:<20} scored={s['scored']:>5} hits={s['hits']:>5} accuracy={acc:.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", help="起始日期 YYYY-MM-DD")
    p.add_argument("--to",   dest="to_date",   help="结束日期 YYYY-MM-DD")
    p.add_argument("--scene", help="只统计某场景,如 scan / tactic / predict")
    args = p.parse_args()
    run(args.from_date, args.to_date, args.scene)


if __name__ == "__main__":
    main()
