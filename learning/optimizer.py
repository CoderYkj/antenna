"""
optimizer.py - 基于历史买入信号精准率，动态调整"买入"信号的相对百分位门槛。

核心指标：买入精准率（Precision）
  = 被标记为"买入"且实际涨幅 >= RISE_THRESHOLD 的股票数 / 全部"买入"信号数
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
import json
from datetime import datetime, date, timedelta
from pathlib import Path

from learning.tracker import load_predictions, load_outcomes, list_prediction_dates

STRATEGY_FILE   = Path("learning/strategy.json")
TARGET_ACCURACY = 0.55   # 目标买入精准率（A股短线现实水平）
RISE_THRESHOLD  = 1.0    # 买入命中 = 实际涨幅 >= 此值（%，从1.5%降至1.0%更贴近实际）


def load_strategy() -> dict:
    if STRATEGY_FILE.exists():
        try:
            with open(STRATEGY_FILE, encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                return json.loads(content)
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
        "history":        [],
    }


# ── 单日回测 ──────────────────────────────────────────────

def evaluate_day(date_str: str) -> dict | None:
    """
    回测某日预测结果。

    只统计推荐股（非自选股）中信号为「买入」的精准率（Precision）：
      - 推荐股买入 → 实际涨幅 >= rise_target_pct 算命中（True Positive）
      - 推荐股买入 → 实际涨幅 < rise_target_pct 算未命中（False Positive）
      - 推荐股观望/回避 → 不计入精准率（规避观望 bias）
      - 自选股 → 单独显示命中/未命中，不计入策略优化的精准率分母

    信号来源：优先读快照中存储的 signal 字段；
    缺失时回退到 rise_prob >= threshold 判断（兼容旧数据）。
    """
    strategy = load_strategy()
    preds    = load_predictions(date_str)
    outcomes = load_outcomes(date_str)
    if not preds or not outcomes:
        return None

    threshold    = strategy.get("buy_threshold", 0.60)  # 仅旧数据回退用
    rise_target  = strategy["rise_target_pct"]
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

        actual_pct = outcomes[code].get("actual_pct")
        if actual_pct is None:
            # 停牌或缺数据，跳过不计入精准率
            continue

        # 推荐股：买入信号才计入策略优化精准率，观望/回避 hit=None
        # 自选股：始终计算 hit 供显示，但不计入策略优化计数器
        if is_watchlist:
            hit = (actual_pct >= rise_target)
        elif is_buy:
            hit = (actual_pct >= rise_target)
            hits  += int(hit)
            total += 1
        else:
            hit = None  # 推荐股观望/回避不参与精准率统计

        details.append({
            "code":        code,
            "name":        pred.get("name", code),
            "is_buy":      is_buy,
            "is_watchlist": is_watchlist,
            "signal":      stored_signal or ("买入" if is_buy else "观望"),
            "rise_prob":   pred["rise_prob"],
            "pred_high":   pred.get("pred_high"),
            "pred_low":    pred.get("pred_low"),
            "actual_pct":  round(actual_pct, 2),
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

    change = ""
    if t7 < 3:
        change = f"买入样本不足（近7日仅 {t7} 条买入信号），暂不调整选股门槛 {old_pct:.0%}"
    elif acc_7d < TARGET_ACCURACY - 0.20:
        new_pct = max(0.08, round(old_pct - 0.01, 4))
        strategy["buy_top_pct"] = new_pct
        change = (f"近7日买入精准率 {acc_7d:.1%} 远低于目标 {TARGET_ACCURACY:.0%}，"
                  f"选股门槛小幅收严 {old_pct:.0%} → {new_pct:.0%}")
    elif acc_7d < TARGET_ACCURACY:
        change = f"近7日买入精准率 {acc_7d:.1%} 低于目标，策略维持选股门槛 {old_pct:.0%}"
    elif acc_7d > TARGET_ACCURACY + 0.10:
        new_pct = min(0.25, round(old_pct + 0.02, 4))
        strategy["buy_top_pct"] = new_pct
        change = (f"近7日买入精准率 {acc_7d:.1%} 远超目标，"
                  f"选股门槛放宽 {old_pct:.0%} → {new_pct:.0%}（挖掘更多机会）")
    elif acc_7d > TARGET_ACCURACY:
        new_pct = min(0.25, round(old_pct + 0.01, 4))
        strategy["buy_top_pct"] = new_pct
        change = (f"近7日买入精准率 {acc_7d:.1%} 达标，"
                  f"选股门槛微放宽 {old_pct:.0%} → {new_pct:.0%}")
    else:
        change = f"近7日买入精准率 {acc_7d:.1%}，策略维持选股门槛 {old_pct:.0%}"

    # 自救机制：若 buy_top_pct 连续处于 0.08 下限超过 7 天，强制重置为 0.15
    new_pct_after = strategy.get("buy_top_pct", old_pct)
    floor_days = strategy.get("floor_days", 0)
    if new_pct_after <= 0.08:
        floor_days += 1
        strategy["floor_days"] = floor_days
        if floor_days >= 7 and t7 >= 3:
            strategy["buy_top_pct"] = 0.15
            strategy["floor_days"] = 0
            change += f"  ⚠️ 门槛已在下限 7 天，自动重置为 15% 重新探索"
    else:
        strategy["floor_days"] = 0

    strategy["last_updated"] = datetime.now().isoformat()
    today_str = (day_result or {}).get("date", date.today().isoformat())
    new_entry = {
        "date":        today_str,
        "acc_7d":      acc_7d,
        "acc_30d":     acc_30d,
        "samples_7":   t7,
        "buy_top_pct": strategy.get("buy_top_pct", old_pct),
        "change":      change,
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
        "change_desc": change_desc,
        "acc_7d":      acc_7d,
        "acc_30d":     acc_30d,
        "samples_7":   t7,
        "samples_30":  t30,
    }
