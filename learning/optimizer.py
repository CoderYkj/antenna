"""
optimizer.py - 基于历史买入信号精准率，动态调整"买入"信号的相对百分位门槛。

核心指标：买入精准率（Precision）
  = 被标记为"买入"且未来 5 日涨幅 >= 2% 的股票数 / 全部"买入"信号数
  只追踪买入信号，彻底规避「观望 bias」
  （观望/回避信号占多数，若也计入准确率，在震荡市中会虚高）

调整目标：55% 精准率（A股短线现实水平，专业选手约55-65%）
策略文件：learning/strategy.json

调整参数：buy_top_pct（模型相对排名前 N% 视为买入，默认 15%）
  - 精准率 < 35%：门槛小幅收严（-0.01），减少误报
  - 精准率 35-55%：维持当前门槛（不缩紧，避免死循环）
  - 精准率 > 65%：门槛放宽（+0.02），挖掘更多机会
  - 精准率 > 55%：门槛微放（+0.01）
  范围限制：[0.08, 0.25]
  自救机制：若 buy_top_pct 连续 7 天处于下限 0.08 且精准率无改善，重置为 0.15
"""
import copy
import json
from datetime import datetime, date, timedelta
from pathlib import Path

from learning.tracker import load_predictions, load_outcomes, list_prediction_dates
from learning.market_state import load_current_state

STRATEGY_FILE   = Path("learning/strategy.json")
TARGET_ACCURACY = 0.55   # 目标买入精准率（A股短线现实水平）
RISE_THRESHOLD  = 1.0    # 旧次日 outcome 的兼容阈值（%）
FIVE_DAY_TARGET = 2.0    # 模型训练目标：未来 5 个交易日累计涨幅（%）
GUARDRAIL_REASON_LABELS = {
    "state_bound_clamp": "状态区间限幅",
    "low_acc_30d_cap": "30日低精准率上限收敛",
    "floor_reset": "下限连续触发重置",
}


def load_strategy() -> dict:
    if STRATEGY_FILE.exists():
        try:
            with open(STRATEGY_FILE, encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                return _merge_with_defaults(json.loads(content))
        except (json.JSONDecodeError, OSError):
            pass  # 文件为空或读取失败，回退默认值
    return _default_strategy()


def save_strategy(s: dict):
    """原子写：先写临时文件，再 replace，避免并发读到空文件。"""
    import os
    import tempfile
    STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STRATEGY_FILE.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, STRATEGY_FILE)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _default_strategy() -> dict:
    return {
        "buy_top_pct":    0.15,   # 涨概率排名前 N% → 买入信号（相对排名门槛）
        "buy_threshold":  0.60,   # 兼容旧字段，当前不作推荐过滤依据
        "rise_target_pct": RISE_THRESHOLD,
        "target_accuracy": TARGET_ACCURACY,
        "accuracy_7d":    None,
        "accuracy_30d":   None,
        "last_updated":   None,
        "floor_days":     0,      # buy_top_pct 连续处于 0.05 下限的天数
        "positioning": {          # Regime 仓位建议（供面板展示与人工执行）
            "bull": 1.0,
            "range": 0.6,
            "bear": 0.3,
        },
        "risk_guardrails": {
            # 按市场状态限制 buy_top_pct 上下限，防止高风险状态过度放宽
            "by_state_bounds": {
                "bull": {"min": 0.08, "max": 0.25},
                "range": {"min": 0.08, "max": 0.20},
                "bear": {"min": 0.06, "max": 0.14},
            },
            # 长周期准确率偏低时，额外收紧上限
            "low_accuracy_cap": {
                "acc_30d_threshold": 0.40,
                "max_buy_top_pct": 0.12,
            },
        },
        "history":        [],
    }


def _merge_with_defaults(strategy: dict) -> dict:
    """向后兼容旧 strategy.json：补齐新增字段，不覆盖用户已有配置。"""
    merged = copy.deepcopy(_default_strategy())
    if not isinstance(strategy, dict):
        return merged
    for k, v in strategy.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def _outcome_hit(outcome: dict, *, rise_target: float = RISE_THRESHOLD) -> bool | None:
    """Evaluate an outcome using the model target, with legacy fallback.

    New outcomes contain ``hit_5d`` (five trading day cumulative return).
    Older records only contain next-day ``actual_pct`` and remain readable.
    """
    if not isinstance(outcome, dict):
            return None
    hit_5d = outcome.get("hit_5d")
    if hit_5d is not None:
            try:
                return float(hit_5d) >= FIVE_DAY_TARGET
            except (TypeError, ValueError):
                return None
    actual_pct = outcome.get("actual_pct")
    if actual_pct is None:
            return None
    try:
            return float(actual_pct) >= rise_target
    except (TypeError, ValueError):
            return None


def _resolve_buy_top_bounds(strategy: dict, acc_30d: float) -> tuple[float, float, str]:
    """按市场状态与长期准确率解析 buy_top_pct 的动态上下限。"""
    current_state = load_current_state().get("current", "range")
    guards = strategy.get("risk_guardrails", {})
    state_bounds = guards.get("by_state_bounds", {})
    state_cfg = state_bounds.get(current_state, state_bounds.get("range", {}))
    min_pct = float(state_cfg.get("min", 0.08))
    max_pct = float(state_cfg.get("max", 0.25))

    low_acc = guards.get("low_accuracy_cap", {})
    acc_30d_threshold = float(low_acc.get("acc_30d_threshold", 0.40))
    if acc_30d < acc_30d_threshold:
        max_pct = min(max_pct, float(low_acc.get("max_buy_top_pct", 0.12)))

    min_pct = max(0.01, min(min_pct, 0.50))
    max_pct = max(min_pct, min(max_pct, 0.50))
    return min_pct, max_pct, current_state


def _build_guardrail_trace(strategy: dict, *, acc_30d: float, buy_top_pct_before: float) -> dict:
    """构建风险护栏追踪信息，用于复盘可追溯展示。"""
    min_pct, max_pct, current_state = _resolve_buy_top_bounds(strategy, acc_30d)
    guards = strategy.get("risk_guardrails", {})
    low_acc = guards.get("low_accuracy_cap", {})
    acc_30d_threshold = float(low_acc.get("acc_30d_threshold", 0.40))
    cap_max_pct = float(low_acc.get("max_buy_top_pct", 0.12))
    cap_triggered = acc_30d < acc_30d_threshold
    return {
        "state": current_state,
        "min_pct": min_pct,
        "max_pct": max_pct,
        "buy_top_pct_before": float(buy_top_pct_before),
        "buy_top_pct_after": float(buy_top_pct_before),
        "cap_triggered": cap_triggered,
        "acc_30d": float(acc_30d),
        "acc_30d_threshold": acc_30d_threshold,
        "cap_max_pct": cap_max_pct,
        "triggered": False,
        "reasons": [],
    }


def format_guardrail_reasons(reasons: list[str] | tuple[str, ...] | None) -> str:
    """将护栏触发原因 code 转为中文标签；未知 code 原样保留。"""
    if not reasons:
        return "无"
    labels = [GUARDRAIL_REASON_LABELS.get(str(r), str(r)) for r in reasons if r]
    return "、".join(labels) if labels else "无"


def build_guardrail_trace_text(trace: dict | None) -> str:
    """将护栏追踪结构化信息格式化为单段可读文本。"""
    if not trace:
        return "护栏追踪：暂无数据"
    state = trace.get("state", "range")
    min_pct = float(trace.get("min_pct", 0.0))
    max_pct = float(trace.get("max_pct", 0.0))
    before = float(trace.get("buy_top_pct_before", 0.0))
    after = float(trace.get("buy_top_pct_after", 0.0))
    reason_text = format_guardrail_reasons(trace.get("reasons") or [])
    return (
        f"护栏追踪：{state} 区间 {min_pct:.0%}~{max_pct:.0%}，"
        f"门槛 {before:.0%}→{after:.0%}，触发原因：{reason_text}"
    )


def summarize_guardrail_history(history: list[dict] | None, window_days: int = 30) -> dict:
    """汇总近 N 日护栏触发情况，供策略/复盘展示。"""
    rows = history or []
    if window_days <= 0:
        window_days = 30
    recent = rows[-window_days:]
    total_days = len(recent)
    events = [r for r in recent if r.get("guardrail_triggered")]
    trigger_days = len(events)
    trigger_rate = (trigger_days / total_days) if total_days else 0.0

    reason_counts: dict[str, int] = {}
    for r in events:
        for code in (r.get("guardrail_reasons") or []):
            key = str(code)
            reason_counts[key] = reason_counts.get(key, 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_reason_items = [
        {"code": code, "label": GUARDRAIL_REASON_LABELS.get(code, code), "count": cnt}
        for code, cnt in top_reasons
    ]
    recent_events = [
        {
            "date": r.get("date", ""),
            "reason_text": r.get("guardrail_reason_text", "无"),
            "buy_top_pct": r.get("buy_top_pct"),
            "acc_7d": r.get("acc_7d"),
        }
        for r in events[-3:]
    ]
    return {
        "window_days": window_days,
        "total_days": total_days,
        "trigger_days": trigger_days,
        "trigger_rate": round(trigger_rate, 4),
        "top_reasons": top_reason_items,
        "recent_events": recent_events,
    }


# ── 单日回测 ──────────────────────────────────────────────

def evaluate_day(date_str: str) -> dict | None:
    """
    回测某日预测结果。

    统计推荐股（非自选股）整体准确率（Accuracy）：
      - 推荐股买入 → 实际涨幅 >= rise_target_pct 算命中
      - 推荐股观望/回避 → 实际涨幅 < rise_target_pct 算命中（正确回避）
      - 自选股 → 单独显示命中/未命中，不计入策略优化计数器

    信号来源：优先读快照中存储的 signal 字段；
    缺失时回退到 rise_prob >= threshold 判断（兼容旧数据）。
    """
    strategy = load_strategy()
    preds    = load_predictions(date_str)
    outcomes = load_outcomes(date_str)
    if not preds or not outcomes:
        return None

    # 自选股只取 predict scene（authoritative），过滤掉 scan scene 的重复条目
    wl_predict_codes = {p["code"] for p in preds if p.get("watchlist") and p.get("scene", "scan") == "predict"}
    preds = [p for p in preds if not (p.get("watchlist") and p.get("scene", "scan") != "predict" and p["code"] in wl_predict_codes)]

    threshold    = strategy.get("buy_threshold", 0.60)  # 仅旧数据回退用
    rise_target  = strategy.get("rise_target_pct", RISE_THRESHOLD)
    hits, total  = 0, 0
    details      = []

    for pred in preds:
        code = pred["code"]
        if code not in outcomes:
            continue

        is_watchlist = bool(pred.get("watchlist"))

        # 优先使用存储的 signal（模型相对信号），兼容旧记录
        stored_signal = pred.get("signal")
        if stored_signal:
            is_buy = (stored_signal == "买入")
        else:
            is_buy = pred["rise_prob"] >= threshold

        outcome = outcomes[code]
        hit = _outcome_hit(outcome, rise_target=rise_target)
        if hit is None:
            # 停牌或缺数据，跳过不计入精准率
            continue

        # 策略准确率只统计推荐池中的买入信号；回避率单独用于展示。
        # 自选股按信号计算命中，但不计入策略优化计数器。
        if is_watchlist:
            hit = hit if is_buy else not hit
        elif is_buy:
            hits  += int(hit)
            total += 1
        else:
            hit = not hit  # 正确回避仅供展示，不混入买入精准率

        details.append({
            "code":        code,
            "name":        pred.get("name", code),
            "is_buy":      is_buy,
            "is_watchlist": is_watchlist,
            "signal":      stored_signal or ("买入" if is_buy else "观望"),
            "rise_prob":   pred["rise_prob"],
            "pred_high":   pred.get("pred_high"),
            "pred_low":    pred.get("pred_low"),
            "actual_pct":  round(float(outcome.get("actual_pct")), 2)
            if outcome.get("actual_pct") is not None else None,
            "actual_pct_5d": outcome.get("hit_5d"),
            "hit":         hit,           # None = 不计入精准率
        })

    buy_count   = sum(1 for d in details if d["is_buy"])
    watch_count = sum(1 for d in details if not d["is_buy"])

    return {
        "date":        date_str,
        "total":       total,            # 买入信号数（分母）
        "hits":        hits,             # 买入命中数（分子）
        "accuracy":    round(hits / total, 4) if total else 0,  # 买入精准率
        "buy_count":   buy_count,
        "watch_count": watch_count,
        "details":     details,
    }


# ── 滚动精准率 ────────────────────────────────────────────

def rolling_accuracy(window_days: int = 30) -> tuple[float, int, int]:
    """计算最近 window_days 个有记录交易日的买入精准率（只统计买入信号）。"""
    all_dates   = list_prediction_dates()
    recent      = all_dates[-window_days:] if len(all_dates) >= window_days else all_dates
    total_hits  = 0
    total_buys  = 0
    for d in recent:
        r = evaluate_day(d)
        if r:
            total_hits += r["hits"]
            total_buys += r["total"]
    acc = total_hits / total_buys if total_buys else 0.0
    return round(acc, 4), total_hits, total_buys


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_max_drawdown(trade_returns: list[float]) -> float:
    """
    基于逐笔收益率序列计算最大回撤（负数，示例 -0.12）。
    trade_returns: [0.01, -0.02, ...]（已是小数）
    """
    if not trade_returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in trade_returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = equity / peak - 1.0
            if dd < max_dd:
                max_dd = dd
    return float(max_dd)


def _iter_window_samples(window_days: int) -> dict:
    """
    聚合窗口内样本，返回监控指标原始统计所需结构。
    仅统计推荐股中的买入信号作为精准率/收益样本；自选股不计入。
    """
    all_dates = list_prediction_dates()
    recent = all_dates[-window_days:] if len(all_dates) >= window_days else all_dates

    buy_returns: list[float] = []
    buy_hits = 0
    buy_total = 0

    signal_counts = {"买入": 0, "观望": 0, "回避": 0}
    confidence_buckets = {"高": {"hits": 0, "total": 0}, "中": {"hits": 0, "total": 0}, "低": {"hits": 0, "total": 0}}
    prob_bins = [
        {"label": "[0.55,0.60)", "lo": 0.55, "hi": 0.60, "hits": 0, "total": 0},
        {"label": "[0.60,0.65)", "lo": 0.60, "hi": 0.65, "hits": 0, "total": 0},
        {"label": "[0.65,1.00]", "lo": 0.65, "hi": 1.01, "hits": 0, "total": 0},
    ]
    topn = {3: {"hits": 0, "total": 0}, 5: {"hits": 0, "total": 0}, 10: {"hits": 0, "total": 0}}

    rise_target = float(load_strategy().get("rise_target_pct", RISE_THRESHOLD))

    for d in recent:
        preds = load_predictions(d) or []
        outcomes = load_outcomes(d) or {}
        if not preds or not outcomes:
            continue

        ranked = []
        for p in preds:
            if p.get("watchlist"):
                continue
            code = p.get("code")
            if code not in outcomes:
                continue
            signal = p.get("signal") or "观望"
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            ranked.append(p)

            if signal != "买入":
                continue

            outcome = outcomes[code]
            hit = _outcome_hit(outcome, rise_target=rise_target)
            if hit is None:
                continue
            buy_total += 1
            buy_hits += int(hit)
            return_pct = outcome.get("hit_5d")
            if return_pct is None:
                return_pct = outcome.get("actual_pct")
            if return_pct is not None:
                buy_returns.append(float(return_pct) / 100.0)

            conf = str(p.get("confidence", "") or "")
            conf_key = conf if conf in confidence_buckets else "中"
            confidence_buckets[conf_key]["total"] += 1
            confidence_buckets[conf_key]["hits"] += int(hit)

            rp = float(p.get("rise_prob", 0.0) or 0.0)
            for b in prob_bins:
                if b["lo"] <= rp < b["hi"]:
                    b["total"] += 1
                    b["hits"] += int(hit)
                    break

        ranked.sort(
            key=lambda x: (
                float(x.get("global_rank_pct", 1.0) or 1.0),
                -float(x.get("rise_prob", 0.0) or 0.0),
            )
        )
        for n in (3, 5, 10):
            picks = [x for x in ranked[:n] if x.get("signal") == "买入"]
            for p in picks:
                code = p.get("code")
                outcome = outcomes.get(code) or {}
                hit = _outcome_hit(outcome, rise_target=rise_target)
                if hit is None:
                    continue
                topn[n]["total"] += 1
                topn[n]["hits"] += int(hit)

    return {
        "window_days": window_days,
        "buy_hits": buy_hits,
        "buy_total": buy_total,
        "buy_returns": buy_returns,
        "signal_counts": signal_counts,
        "confidence_buckets": confidence_buckets,
        "prob_bins": prob_bins,
        "topn": topn,
    }


def build_monitor_dashboard_metrics() -> dict:
    """构建 7日/30日监控看板指标结构化结果。"""
    strategy = load_strategy()
    history = strategy.get("history") or []
    latest7 = _iter_window_samples(7)
    latest30 = _iter_window_samples(30)

    def _pack_window(raw: dict) -> dict:
        buy_total = int(raw.get("buy_total", 0))
        buy_hits = int(raw.get("buy_hits", 0))
        hit_rate = (buy_hits / buy_total) if buy_total else 0.0
        returns = raw.get("buy_returns", [])
        avg_return = _safe_mean(returns) if returns else 0.0
        max_dd = _compute_max_drawdown(returns)
        rar = (avg_return / abs(max_dd)) if max_dd < 0 else None
        return {
            "window_days": int(raw.get("window_days", 0)),
            "hit_rate": round(hit_rate, 4),
            "samples": buy_total,
            "hits": buy_hits,
            "avg_return": round(avg_return, 4),
            "max_drawdown": round(max_dd, 4),
            "risk_adjusted_return": round(rar, 4) if isinstance(rar, float) else None,
            "signal_counts": raw.get("signal_counts", {}),
            "topn_hit_rate": {
                f"top{n}": round((v["hits"] / v["total"]), 4) if v["total"] else 0.0
                for n, v in raw.get("topn", {}).items()
            },
            "topn_samples": {f"top{n}": int(v["total"]) for n, v in raw.get("topn", {}).items()},
            "confidence_hit_rate": {
                k: (round(v["hits"] / v["total"], 4) if v["total"] else 0.0)
                for k, v in (raw.get("confidence_buckets") or {}).items()
            },
            "confidence_samples": {k: int(v["total"]) for k, v in (raw.get("confidence_buckets") or {}).items()},
            "prob_bin_hit_rate": {
                b["label"]: (round(b["hits"] / b["total"], 4) if b["total"] else 0.0)
                for b in (raw.get("prob_bins") or [])
            },
            "prob_bin_samples": {b["label"]: int(b["total"]) for b in (raw.get("prob_bins") or [])},
        }

    guardrail_7 = summarize_guardrail_history(history, 7)
    guardrail_30 = summarize_guardrail_history(history, 30)
    changes_7 = len([h for h in history[-7:] if h.get("change")])
    changes_30 = len([h for h in history[-30:] if h.get("change")])
    w7 = _pack_window(latest7)
    w30 = _pack_window(latest30)

    def _build_precision_alerts() -> list[dict]:
        alerts: list[dict] = []
        s7 = int(w7.get("samples", 0))
        s30 = int(w30.get("samples", 0))
        h7 = float(w7.get("hit_rate", 0.0))
        h30 = float(w30.get("hit_rate", 0.0))
        g7 = float(guardrail_7.get("trigger_rate", 0.0))
        dd7 = float(w7.get("max_drawdown", 0.0))
        dd30 = float(w30.get("max_drawdown", 0.0))

        low_sample_floor = max(3, int(s30 * 0.50)) if s30 else 0
        if (
            s30 >= 10
            and s7 > 0
            and s7 < low_sample_floor
            and h7 <= h30 + 0.02
            and g7 >= 0.40
        ):
            alerts.append({
                "level": "warning",
                "code": "over_demotion_risk",
                "message": (
                    f"近7日买入样本仅 {s7}（30日 {s30}），且命中率未明显优于30日，"
                    "存在过度降级风险，建议回调 weak 模式闸门强度。"
                ),
            })

        if s7 >= 8 and (h30 - h7) >= 0.08:
            alerts.append({
                "level": "warning",
                "code": "hit_rate_drift",
                "message": (
                    f"近7日命中率 {h7:.1%} 低于30日 {h30:.1%}，短期质量走弱，"
                    "建议优先排查近期触发最多的降级来源。"
                ),
            })

        if s7 >= 8 and (dd7 - dd30) <= -0.05:
            alerts.append({
                "level": "warning",
                "code": "drawdown_drift",
                "message": (
                    f"近7日回撤 {dd7:.2%} 明显劣于30日 {dd30:.2%}，"
                    "建议收紧高波动与趋势过热相关阈值。"
                ),
            })
        return alerts

    return {
        "as_of": date.today().isoformat(),
        "windows": {
            "7d": w7,
            "30d": w30,
        },
        "guardrail": {
            "7d": guardrail_7,
            "30d": guardrail_30,
        },
        "parameter_changes": {
            "7d": changes_7,
            "30d": changes_30,
        },
        "alerts": _build_precision_alerts(),
    }


# ── 策略优化 ──────────────────────────────────────────────

def optimize(day_result: dict | None = None) -> tuple[dict, str]:
    """
    根据近期买入信号精准率调整 buy_top_pct，返回 (updated_strategy, 变更说明)。
    buy_top_pct：涨概率排名前 N% 的股票才发出"买入"信号。
      精准率低 → 收窄百分位（门槛更严，减少误报）
      精准率高 → 放宽百分位（门槛更松，挖掘更多机会）
    """
    strategy = load_strategy()
    old_pct  = strategy.get("buy_top_pct", 0.10)

    acc_7d,  h7,  t7  = rolling_accuracy(7)
    acc_30d, h30, t30 = rolling_accuracy(30)

    strategy["accuracy_7d"]  = acc_7d
    strategy["accuracy_30d"] = acc_30d
    min_pct, max_pct, cur_state = _resolve_buy_top_bounds(strategy, acc_30d)
    guardrail_trace = _build_guardrail_trace(
        strategy,
        acc_30d=acc_30d,
        buy_top_pct_before=float(old_pct),
    )

    change = ""
    if t7 < 3:
        change = f"买入样本不足（近7日仅 {t7} 条买入信号），暂不调整选股门槛 {old_pct:.0%}"
    elif acc_7d < TARGET_ACCURACY - 0.20:
        new_pct = max(min_pct, round(old_pct - 0.01, 4))
        strategy["buy_top_pct"] = new_pct
        change = (f"近7日买入精准率 {acc_7d:.1%} 远低于目标 {TARGET_ACCURACY:.0%}，"
                  f"选股门槛小幅收严 {old_pct:.0%} → {new_pct:.0%}")
    elif acc_7d < TARGET_ACCURACY:
        change = f"近7日买入精准率 {acc_7d:.1%} 低于目标，策略维持选股门槛 {old_pct:.0%}"
    elif acc_7d > TARGET_ACCURACY + 0.10:
        new_pct = min(max_pct, round(old_pct + 0.02, 4))
        strategy["buy_top_pct"] = new_pct
        change = (f"近7日买入精准率 {acc_7d:.1%} 远超目标，"
                  f"选股门槛放宽 {old_pct:.0%} → {new_pct:.0%}（挖掘更多机会）")
    elif acc_7d > TARGET_ACCURACY:
        new_pct = min(max_pct, round(old_pct + 0.01, 4))
        strategy["buy_top_pct"] = new_pct
        change = (f"近7日买入精准率 {acc_7d:.1%} 达标，"
                  f"选股门槛微放宽 {old_pct:.0%} → {new_pct:.0%}")
    else:
        change = f"近7日买入精准率 {acc_7d:.1%}，策略维持选股门槛 {old_pct:.0%}"

    # 自救机制：若 buy_top_pct 连续处于 0.08 下限超过 7 天，强制重置为 0.15
    new_pct_after = strategy.get("buy_top_pct", old_pct)
    clamped_pct = min(max(new_pct_after, min_pct), max_pct)
    if clamped_pct != new_pct_after:
        strategy["buy_top_pct"] = round(clamped_pct, 4)
        change += f"  🛡 风险护栏生效（{cur_state}）→ 门槛限制到 {clamped_pct:.0%}"
        guardrail_trace["triggered"] = True
        guardrail_trace["reasons"].append("state_bound_clamp")
    new_pct_after = strategy.get("buy_top_pct", old_pct)
    guardrail_trace["buy_top_pct_after"] = float(new_pct_after)
    floor_days = strategy.get("floor_days", 0)
    if new_pct_after <= min_pct:
        floor_days += 1
        strategy["floor_days"] = floor_days
        if floor_days >= 7 and t7 >= 3:
            reset_pct = min(max(0.15, min_pct), max_pct)
            strategy["buy_top_pct"] = round(reset_pct, 4)
            strategy["floor_days"] = 0
            change += f"  ⚠️ 门槛已在下限 7 天，自动重置为 {reset_pct:.0%} 重新探索"
            guardrail_trace["triggered"] = True
            guardrail_trace["reasons"].append("floor_reset")
    else:
        strategy["floor_days"] = 0

    if guardrail_trace["cap_triggered"]:
        guardrail_trace["triggered"] = True
        guardrail_trace["reasons"].append("low_acc_30d_cap")
    guardrail_trace["reasons"] = list(dict.fromkeys(guardrail_trace.get("reasons") or []))
    guardrail_trace["reason_text"] = format_guardrail_reasons(guardrail_trace["reasons"])
    guardrail_trace["summary"] = build_guardrail_trace_text(guardrail_trace)
    strategy["last_guardrail_trace"] = guardrail_trace
    strategy["last_updated"] = datetime.now().isoformat()
    today_str = (day_result or {}).get("date", date.today().isoformat())
    new_entry = {
        "date":        today_str,
        "acc_7d":      acc_7d,
        "acc_30d":     acc_30d,
        "samples_7":   t7,
        "buy_top_pct": strategy.get("buy_top_pct", old_pct),
        "change":      change,
        "guardrail_triggered": bool(guardrail_trace.get("triggered")),
        "guardrail_reason_text": guardrail_trace.get("reason_text", "无"),
        "guardrail_reasons": list(guardrail_trace.get("reasons") or []),
    }
    # 同一天只保留最后一条（覆盖旧记录）
    hist = [h for h in strategy.setdefault("history", []) if h.get("date") != today_str]
    hist.append(new_entry)
    strategy["history"] = hist[-90:]   # 保留最近 90 天

    save_strategy(strategy)
    return strategy, change


# ── 报告生成 ──────────────────────────────────────────────

def build_review_report(date_str: str) -> dict:
    """
    生成某日回测报告，供推送到飞书。
    Returns: {date, day_result, strategy, change_desc, acc_7d, acc_30d}
    """
    day_result  = evaluate_day(date_str)
    strategy, change_desc = optimize(day_result)
    acc_7d,  _, t7  = rolling_accuracy(7)
    acc_30d, _, t30 = rolling_accuracy(30)

    return {
        "date":        date_str,
        "day_result":  day_result,
        "strategy":    strategy,
        "guardrail_trace": strategy.get("last_guardrail_trace", {}),
        "guardrail_trace_text": build_guardrail_trace_text(strategy.get("last_guardrail_trace", {})),
        "guardrail_summary": summarize_guardrail_history(strategy.get("history") or [], 30),
        "dashboard_metrics": build_monitor_dashboard_metrics(),
        "change_desc": change_desc,
        "acc_7d":      acc_7d,
        "acc_30d":     acc_30d,
        "samples_7":   t7,
        "samples_30":  t30,
    }
