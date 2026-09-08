"""
replay_learn.py - 在已有历史 pred/outcome 上模拟滚动学习,输出基线与 P1 对比。

P0 基线模式(默认):
  对每一天 D:
    1. load_predictions(D) / load_outcomes(D)
    2. 计算当日买入精准率(区分 scan / tactic:* / predict 各场景)
    3. 按市场状态打标
    4. 写入 learning/replay/baseline_YYYYMMDD_HHMMSS.csv

P1 对比模式(--with-p1):
  对每一天 D:
    1. 收集前 N 天(默认 90 天)所有 (pred.rise_prob, outcome.hit_tier),按 market_state 分桶
    2. 对当日 market_state 对应桶拟合 IsotonicRegression
       - 样本不足 → IdentityCalibrator
       - P1 校准器"贴合当前模型"的理想做法需要重打分历史样本,本脚本为回放效率简化为
         直接用 pred 中已记录的 rise_prob(spec §5.1 允许)
    3. 对当日所有 pred 应用 calibrator.transform → prob_cal
    4. 按分位 + abs_threshold 双门槛重新判 signal,计算 P1 精准率
    5. 计算 Brier:raw(原 rise_prob) vs cal(校准后),delta = cal - raw(应 ≤ 0)
    6. 输出 learning/replay/p1_compare_YYYYMMDD_HHMMSS.csv

spec §9 验收硬指标:
  - 买入信号精准率 ≥ 35%
  - Brier ≤ raw × 0.95(delta_brier 均值 ≤ -0.05 × baseline_brier)

用法:
  python scripts/replay_learn.py                                   # 基线模式
  python scripts/replay_learn.py --with-p1                         # P1 对比模式
  python scripts/replay_learn.py --with-p1 --from 2026-01-01
  python scripts/replay_learn.py --scene tactic --with-p1
"""
import sys
import io
import os
import argparse
import csv
import json
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


# ── P1 回放核心函数 ─────────────────────────────────────────

def _is_hit(tier: str | None) -> int:
    """hit_tier ∈ {good, great} → 1,否则 0(含 None)。"""
    return 1 if tier in ("good", "great") else 0


def _collect_bucket_samples(
    start_date: str,
    end_date_exclusive: str,
    bucket_states: set[str],
) -> list[tuple[float, int, str]]:
    """收集 [start_date, end_date_exclusive) 区间所有符合 bucket_states 的样本。

    返回 [(rise_prob, hit_binary, scene), ...]。
    rise_prob 从 pred 取;hit 从 outcome.hit_tier 即时回补。
    """
    from learning.tracker import list_prediction_dates, load_predictions, load_outcomes
    from learning.market_state import load_state_on_date
    from learning.outcome_metrics import compute_hit_tier

    samples: list[tuple[float, int, str]] = []
    for d in list_prediction_dates():
        if d < start_date or d >= end_date_exclusive:
            continue
        state = load_state_on_date(d)
        if state not in bucket_states:
            continue
        preds = load_predictions(d)
        outcomes = load_outcomes(d)
        for p in preds:
            prob = p.get("rise_prob")
            if prob is None:
                continue
            o = outcomes.get(p.get("code"))
            if not o:
                continue
            # Prefer the five-day target used by the model; retain legacy fallback.
            tier = (
                compute_hit_tier(o.get("hit_5d"))
                if o.get("hit_5d") is not None
                else o.get("hit_tier") or compute_hit_tier(o.get("actual_pct"))
            )
            if tier is None:
                continue
            samples.append((float(prob), _is_hit(tier), p.get("scene") or "scan"))
    return samples


def _fit_calibrator(samples: list[tuple[float, int]], min_n: int = 50):
    """拟合 IsotonicRegression,不足或退化 → IdentityCalibrator。"""
    from learning.model_learner import IdentityCalibrator, fit_calibrator_for_bucket, load_config
    # 用 P1 的 fit_calibrator_for_bucket,保证行为对齐生产
    try:
        cfg = load_config()
    except ValueError:
        return IdentityCalibrator(reason="no_config"), False
    cal = fit_calibrator_for_bucket(samples, cfg)
    return cal, not isinstance(cal, IdentityCalibrator)


def _brier(probs: list[float], hits: list[int]) -> float | None:
    if not probs or len(probs) != len(hits):
        return None
    return sum((p - h) ** 2 for p, h in zip(probs, hits)) / len(probs)


def _shift_date(date_str: str, days: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def _evaluate_day_p1(
    date_str: str,
    preds: list,
    outcomes: dict,
    scene_filter: str | None,
    lookback_days: int,
    buy_top_pct: float,
    abs_threshold: float,
) -> dict | None:
    """对一天 D 做 P1 vs baseline 对比,返回汇总 dict。"""
    from learning.market_state import load_state_on_date
    from learning.outcome_metrics import compute_hit_tier
    from learning.scene_bucket import DEFAULT_SCENE

    if scene_filter:
        preds = [p for p in preds if (p.get("scene") or DEFAULT_SCENE).startswith(scene_filter)]

    state = load_state_on_date(date_str)

    # 1. 拟合校准器(以当日 state 为桶,lookback_days 内样本)
    lookback_start = _shift_date(date_str, -lookback_days)
    bucket_samples_full = _collect_bucket_samples(lookback_start, date_str, {state})
    # 兜底:单桶不足时用全部三状态样本(global fallback)
    if len(bucket_samples_full) < 50:
        bucket_samples_full = _collect_bucket_samples(lookback_start, date_str, {"bull", "bear", "range"})
    bucket_pair = [(s[0], s[1]) for s in bucket_samples_full]
    cal, fitted = _fit_calibrator(bucket_pair)

    # 2. 当日 pred 评估:baseline(signal=="买入") vs P1(分位 + abs_threshold 双门槛)
    # 收集 (prob_raw, prob_cal, hit, signal_p0) 元组
    rows = []
    for p in preds:
        prob_raw = p.get("rise_prob")
        if prob_raw is None:
            continue
        o = outcomes.get(p.get("code"))
        if not o:
            continue
        tier = o.get("hit_tier") or compute_hit_tier(o.get("actual_pct"))
        if tier is None:
            continue
        hit = _is_hit(tier)
        prob_cal = float(cal.transform([float(prob_raw)])[0]) if fitted else float(prob_raw)
        rows.append({
            "prob_raw":  float(prob_raw),
            "prob_cal":  prob_cal,
            "hit":       hit,
            "signal_p0": p.get("signal"),
            "scene":     p.get("scene") or "scan",
        })

    if not rows:
        return None

    # 3. baseline 精准率:signal="买入" 的 hit 比例
    p0_buys  = [r for r in rows if r["signal_p0"] == "买入"]
    p0_total = len(p0_buys)
    p0_hits  = sum(r["hit"] for r in p0_buys)
    p0_acc   = p0_hits / p0_total if p0_total else None

    # 4. P1 精准率:按 prob_raw 排名前 buy_top_pct 且 prob_cal >= abs_threshold 视为买入
    rows_sorted = sorted(rows, key=lambda x: x["prob_raw"], reverse=True)
    n = len(rows_sorted)
    top_cut = max(1, int(n * buy_top_pct))
    p1_buys = [r for r in rows_sorted[:top_cut] if r["prob_cal"] >= abs_threshold]
    p1_total = len(p1_buys)
    p1_hits  = sum(r["hit"] for r in p1_buys)
    p1_acc   = p1_hits / p1_total if p1_total else None

    # 5. Brier(基于所有有 outcome 的 pred,不限买入)
    probs_raw = [r["prob_raw"] for r in rows]
    probs_cal = [r["prob_cal"] for r in rows]
    hits      = [r["hit"]      for r in rows]
    brier_raw = _brier(probs_raw, hits)
    brier_cal = _brier(probs_cal, hits)
    delta_brier = (brier_cal - brier_raw) if (brier_raw is not None and brier_cal is not None) else None

    return {
        "date":         date_str,
        "market_state": state,
        "calibrator_fitted": "yes" if fitted else "no",
        "cal_samples":  len(bucket_pair),
        "total_rows":   n,
        "baseline_buys": p0_total,
        "baseline_hits": p0_hits,
        "baseline_acc":  round(p0_acc, 4) if p0_acc is not None else "",
        "p1_buys":       p1_total,
        "p1_hits":       p1_hits,
        "p1_acc":        round(p1_acc, 4) if p1_acc is not None else "",
        "delta_acc":     round((p1_acc or 0) - (p0_acc or 0), 4) if (p0_acc is not None and p1_acc is not None) else "",
        "brier_raw":     round(brier_raw, 4) if brier_raw is not None else "",
        "brier_cal":     round(brier_cal, 4) if brier_cal is not None else "",
        "delta_brier":   round(delta_brier, 4) if delta_brier is not None else "",
    }


def run_baseline(from_date: str | None, to_date: str | None, scene_filter: str | None):
    """原 P0 基线模式(保留)。"""
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

    agg: dict[str, dict] = defaultdict(lambda: {"hits": 0, "scored": 0})
    for r in rows:
        key = r["scene"]
        agg[key]["hits"]   += r["hits"]
        agg[key]["scored"] += r["scored"]
    print("\n场景汇总(整个回放期间):")
    for scene, s in agg.items():
        acc = (s["hits"] / s["scored"]) if s["scored"] else 0
        print(f"  {scene:<20} scored={s['scored']:>5} hits={s['hits']:>5} accuracy={acc:.1%}")


def run_p1_compare(
    from_date: str | None,
    to_date: str | None,
    scene_filter: str | None,
    lookback_days: int,
    buy_top_pct: float,
    abs_threshold: float,
):
    """P1 对比模式:逐日拟合 calibrator,输出 baseline vs P1 精准率与 Brier 对比。"""
    from learning.tracker import list_prediction_dates, load_predictions, load_outcomes

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
    out_path = os.path.join(OUT_DIR, f"p1_compare_{run_ts}.csv")

    print(f"[P1 回放] 区间 {all_dates[0]} ~ {all_dates[-1]},共 {len(all_dates)} 天")
    print(f"[P1 配置] lookback={lookback_days} buy_top_pct={buy_top_pct} abs_threshold={abs_threshold}")

    rows = []
    for i, d in enumerate(all_dates):
        preds    = load_predictions(d)
        outcomes = load_outcomes(d)
        if not preds or not outcomes:
            continue

        summary = _evaluate_day_p1(
            date_str=d, preds=preds, outcomes=outcomes,
            scene_filter=scene_filter, lookback_days=lookback_days,
            buy_top_pct=buy_top_pct, abs_threshold=abs_threshold,
        )
        if summary is None:
            continue
        rows.append(summary)

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(all_dates)} 天,最近: {d}")

    if not rows:
        print("无有效结果写入")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nP1 对比曲线已写入: {out_path}")
    print(f"总行数: {len(rows)}")

    # 整体汇总
    total_p0_buys = sum(r["baseline_buys"] for r in rows)
    total_p0_hits = sum(r["baseline_hits"] for r in rows)
    total_p1_buys = sum(r["p1_buys"] for r in rows)
    total_p1_hits = sum(r["p1_hits"] for r in rows)
    p0_acc = (total_p0_hits / total_p0_buys) if total_p0_buys else 0
    p1_acc = (total_p1_hits / total_p1_buys) if total_p1_buys else 0

    # Brier 均值(排除 None)
    brier_raw_vals = [r["brier_raw"] for r in rows if isinstance(r["brier_raw"], (int, float))]
    brier_cal_vals = [r["brier_cal"] for r in rows if isinstance(r["brier_cal"], (int, float))]
    mean_brier_raw = sum(brier_raw_vals) / len(brier_raw_vals) if brier_raw_vals else None
    mean_brier_cal = sum(brier_cal_vals) / len(brier_cal_vals) if brier_cal_vals else None

    print("\n========== P1 验收汇总 ==========")
    print(f"Baseline 买入: scored={total_p0_buys:>5} hits={total_p0_hits:>5} accuracy={p0_acc:.2%}")
    print(f"P1 双门槛   : scored={total_p1_buys:>5} hits={total_p1_hits:>5} accuracy={p1_acc:.2%}")
    print(f"Δ accuracy  : {(p1_acc - p0_acc) * 100:+.2f}pt  (硬指标:≥ +2pt 且 p1_acc ≥ 35%)")
    if mean_brier_raw is not None and mean_brier_cal is not None:
        ratio = mean_brier_cal / mean_brier_raw if mean_brier_raw else 0
        print(f"Mean Brier  : raw={mean_brier_raw:.4f}  cal={mean_brier_cal:.4f}  ratio={ratio:.3f}  (硬指标:≤ 0.95)")

    # calibrator 拟合率
    fitted_cnt = sum(1 for r in rows if r["calibrator_fitted"] == "yes")
    print(f"Calibrator  : 成功拟合 {fitted_cnt}/{len(rows)} 天 ({fitted_cnt*100/len(rows):.0f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", help="起始日期 YYYY-MM-DD")
    p.add_argument("--to",   dest="to_date",   help="结束日期 YYYY-MM-DD")
    p.add_argument("--scene", help="只统计某场景,如 scan / tactic / predict")
    p.add_argument("--with-p1", action="store_true", help="启用 P1 对比模式")
    p.add_argument("--lookback", type=int, default=90, help="P1 回溯窗口天数(默认 90)")
    p.add_argument("--buy-top-pct", type=float, default=0.10, help="P1 分位门槛(默认 0.10)")
    p.add_argument("--abs-threshold", type=float, default=0.45, help="P1 绝对阈值门槛(默认 0.45)")
    args = p.parse_args()

    if args.with_p1:
        run_p1_compare(
            from_date=args.from_date, to_date=args.to_date, scene_filter=args.scene,
            lookback_days=args.lookback, buy_top_pct=args.buy_top_pct,
            abs_threshold=args.abs_threshold,
        )
    else:
        run_baseline(args.from_date, args.to_date, args.scene)


if __name__ == "__main__":
    main()
