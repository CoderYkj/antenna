"""
tactic_learner.py - P2 战法层:4 战法阈值自适应 + 共振权重学习。

核心交付:
  - 4 战法 × 3 状态 = 12 桶独立统计 90 日精准率
  - 阈值微调:逐阈值按 direction 标记应用步长(D5)
  - 战法权重:归一化精准率,共振股调 rank_pct 不动 prob_cal(D3)
  - bear 桶 defaults_bear_override 提供保守兜底(D4)

产物:
  - learning/tactic_params.json(主状态文件,含 params + history)

参数全部走 learning/tactic_learner.yaml,代码内不硬编码任何数值。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml

from learning import feedback_io, market_state, tracker
from learning.outcome_metrics import compute_hit_tier

logger = logging.getLogger(__name__)

MarketState = Literal["bull", "bear", "range"]
Tactic = Literal["value", "growth", "leader", "contra"]
TACTICS: tuple[Tactic, ...] = ("value", "growth", "leader", "contra")
STATES: tuple[MarketState, ...] = ("bull", "bear", "range")

# ── 路径常量 ────────────────────────────────────────────────
CONFIG_PATH = Path("learning/tactic_learner.yaml")
STATE_FILE  = Path("learning/tactic_params.json")
SCENE_PREFIX = "tactic:"  # tracker scene 格式


# ── 配置加载 ────────────────────────────────────────────────

@dataclass(frozen=True)
class TacticConfig:
    defaults:                dict
    defaults_bear_override:  dict
    bounds:                  dict
    step:                    dict
    evaluation:              dict
    weights:                 dict


def load_config(path: str | Path | None = None) -> TacticConfig:
    """读 yaml → frozen TacticConfig。文件缺失/字段缺失抛 ValueError。

    path=None 时在调用时解析 CONFIG_PATH(支持 monkeypatch);早绑定会让测试 fixture 失效。
    """
    if path is None:
        path = CONFIG_PATH
    p = Path(path)
    if not p.exists():
        raise ValueError(f"config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in ("defaults", "bounds", "step", "evaluation", "weights"):
        if key not in data:
            raise ValueError(f"config missing required section: {key}")
    return TacticConfig(
        defaults=dict(data["defaults"]),
        defaults_bear_override=dict(data.get("defaults_bear_override") or {}),
        bounds=dict(data["bounds"]),
        step=dict(data["step"]),
        evaluation=dict(data["evaluation"]),
        weights=dict(data["weights"]),
    )


# ── 阈值结构辅助 ────────────────────────────────────────────

def _is_threshold_entry(v) -> bool:
    """判断 yaml 节点是否是 {value, direction} 形式的阈值条目。"""
    return isinstance(v, dict) and "value" in v and "direction" in v


def _extract_value(entry):
    """从 {value, direction} 取数值;布尔/标量直通。"""
    if _is_threshold_entry(entry):
        return entry["value"]
    return entry


def get_defaults_for_state(state: MarketState, cfg: TacticConfig) -> dict[Tactic, dict[str, float]]:
    """返回某状态下所有战法的默认值(扁平化,移除 direction 包装)。

    bear 状态下:defaults_bear_override 覆盖同名阈值;其余沿用 defaults。
    """
    flat: dict[Tactic, dict[str, float]] = {}
    for tactic in TACTICS:
        params: dict[str, float] = {}
        for name, entry in cfg.defaults.get(tactic, {}).items():
            params[name] = _extract_value(entry)
        # bear override
        if state == "bear":
            override = cfg.defaults_bear_override.get(tactic, {})
            for name, value in override.items():
                params[name] = value
        flat[tactic] = params
    return flat


def _get_direction(tactic: Tactic, threshold_name: str, cfg: TacticConfig) -> str | None:
    """查 yaml 中某 (tactic, threshold) 的 direction。布尔/缺失返回 None。"""
    entry = cfg.defaults.get(tactic, {}).get(threshold_name)
    if not _is_threshold_entry(entry):
        return None
    return entry.get("direction")


def _step_for(threshold_name: str, cfg: TacticConfig) -> float:
    """阈值对应步长。drawdown 特殊处理,其余用 default。"""
    if "drawdown" in threshold_name:
        return float(cfg.step.get("drawdown", 0.01))
    return float(cfg.step.get("default", 1.0))


# ── 关键纯函数 ──────────────────────────────────────────────

def adjust_thresholds(
    current: dict[str, float],
    acc_90d: float | None,
    samples: int,
    cfg: TacticConfig,
    tactic: Tactic,
) -> tuple[dict[str, float], str]:
    """按 spec §3.2 表返回 (新阈值字典, 调整原因)。

    符号由每阈值的 direction 决定(D5):
      - tighten_up:   收严 +step,放宽 -step
      - tighten_down: 收严 -step,放宽 +step
    样本不足 → 维持。无 direction(如布尔标志) → 跳过该阈值。
    """
    min_samples = int(cfg.evaluation.get("min_samples", 30))
    acc_low     = float(cfg.evaluation.get("acc_low", 0.30))
    acc_high    = float(cfg.evaluation.get("acc_high", 0.60))

    if samples < min_samples:
        return dict(current), "insufficient_samples"
    if acc_90d is None:
        return dict(current), "no_acc_90d"
    if acc_low <= acc_90d <= acc_high:
        return dict(current), f"acc_90d {acc_90d:.3f} 维持区间"

    tighten = acc_90d < acc_low
    new_thresholds: dict[str, float] = {}
    bounds_for_tactic = cfg.bounds.get(tactic, {})

    for name, val in current.items():
        direction = _get_direction(tactic, name, cfg)
        if direction is None:
            # 无方向标记(布尔字段或未配置),原样保留
            new_thresholds[name] = val
            continue

        step = _step_for(name, cfg)
        # 计算符号:tighten_up 收严 → +step;tighten_down 收严 → -step
        if tighten:
            delta = +step if direction == "tighten_up" else -step
        else:
            delta = -step if direction == "tighten_up" else +step

        new_val = float(val) + delta

        # 边界夹紧
        lo, hi = bounds_for_tactic.get(name, [None, None]) if name in bounds_for_tactic else (None, None)
        if lo is not None and new_val < lo:
            new_val = lo
        if hi is not None and new_val > hi:
            new_val = hi

        new_thresholds[name] = round(new_val, 4)

    action = "收严" if tighten else "放宽"
    return new_thresholds, f"acc_90d {acc_90d:.3f}, {action}({samples} 样本)"


def compute_resonance_weights(
    bucket_accs: dict[Tactic, float | None],
    cfg: TacticConfig,
) -> dict[Tactic, float]:
    """根据精准率归一化战法权重,夹到 [weight_min, weight_max]。

    None / 0 精准率 → 用 weight_min 兜底。
    """
    w_min = float(cfg.weights.get("weight_min", 0.05))
    w_max = float(cfg.weights.get("weight_max", 0.40))

    # 第一步:有效精准率(None 视为 0)
    raw = {t: (bucket_accs.get(t) or 0.0) for t in TACTICS}
    total = sum(raw.values())

    if total <= 0:
        # 全 0 → 所有战法 weight_min 均分
        return {t: w_min for t in TACTICS}

    out: dict[Tactic, float] = {}
    for t in TACTICS:
        normalized = raw[t] / total
        out[t] = round(max(w_min, min(w_max, normalized)), 4)
    return out


# ── 数据装载辅助 ────────────────────────────────────────────

def _is_hit(tier: str | None) -> int:
    return 1 if tier in ("good", "great") else 0


def evaluate_tactic(
    state: MarketState,
    tactic: Tactic,
    date_str: str,
    lookback_days: int = 90,
) -> dict:
    """统计 (state, tactic) 桶近 90 日 signal=买入 的精准率。

    扫 pred[scene=tactic:<tactic>],按 outcome.hit_tier 计命中率。
    返回 {acc_90d: float|None, samples: int, hits: int}。
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    target_scene = f"{SCENE_PREFIX}{tactic}"
    samples = 0
    hits = 0

    for offset in range(1, lookback_days + 1):
        d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
        # 该日状态匹配才计入
        day_state = market_state.load_state_on_date(d)
        if day_state != state:
            continue
        preds = tracker.load_predictions_by_scene(d, target_scene)
        if not preds:
            continue
        outcomes = tracker.load_outcomes(d)
        for p in preds:
            if p.get("signal") != "买入":
                continue
            o = outcomes.get(p.get("code"))
            if not o:
                continue
            tier = o.get("hit_tier") or compute_hit_tier(o.get("actual_pct"))
            if tier is None:
                continue
            samples += 1
            hits += _is_hit(tier)

    acc = (hits / samples) if samples else None
    return {"acc_90d": acc, "samples": samples, "hits": hits}


# ── 加载产物:供 predict_cmd 等外部使用 ──────────────────

def load_params(state: MarketState, cfg: TacticConfig | None = None) -> dict[Tactic, dict[str, float]]:
    """读 tactic_params.json[state]。文件缺失/损坏 → 回退 yaml.defaults(含 bear override)。

    供 server/predict_cmd._enrich_tactic_scores 调用;永不抛异常。
    """
    if cfg is None:
        try:
            cfg = load_config()
        except Exception:
            return {}

    fallback = get_defaults_for_state(state, cfg)

    if not STATE_FILE.exists():
        return fallback
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("params", {}).get(state)
        if not params:
            return fallback
        # 转成扁平 {tactic: {name: value}}
        out: dict[Tactic, dict[str, float]] = {}
        for tactic in TACTICS:
            entry = params.get(tactic, {})
            tactic_params = {}
            for name, val in entry.items():
                # 跳过 weight/acc_90d/samples/status 等 meta 字段
                if name in ("weight", "acc_90d", "samples", "status"):
                    continue
                tactic_params[name] = val
            # 没拿到任何阈值 → 兜底
            out[tactic] = tactic_params if tactic_params else fallback.get(tactic, {})
        return out
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return fallback


def load_resonance_weights(state: MarketState) -> dict[Tactic, float]:
    """读 tactic_params.json[state][tactic].weight。失败回退均分 weight_min。"""
    try:
        cfg = load_config()
    except Exception:
        return {t: 0.05 for t in TACTICS}

    w_min = float(cfg.weights.get("weight_min", 0.05))
    fallback = {t: w_min for t in TACTICS}

    if not STATE_FILE.exists():
        return fallback
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("params", {}).get(state, {})
        weights = {}
        for t in TACTICS:
            weights[t] = float(params.get(t, {}).get("weight", w_min))
        return weights
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return fallback


# ── 主入口 ──────────────────────────────────────────────────

def _persist_state(date_str: str, summary: dict, cfg: TacticConfig) -> None:
    """写入 tactic_params.json + history 追加 + 7 份历史快照。"""
    prev = _load_state_safely()
    history = list(prev.get("history") or [])

    # 收集本次每个 (state, tactic) 的调整记录
    for state_key, tactics_map in summary.items():
        for tactic_key, info in tactics_map.items():
            if info.get("changed"):
                history.append({
                    "date":    date_str,
                    "state":   state_key,
                    "tactic":  tactic_key,
                    "old":     info.get("old", {}),
                    "new":     {k: v for k, v in info.items()
                                if k not in ("old", "changed", "weight", "acc_90d",
                                             "samples", "status")},
                    "acc_90d": info.get("acc_90d"),
                    "samples": info.get("samples"),
                    "change":  info.get("status", ""),
                })
    history = history[-200:]

    # 清理 summary 里的 changed/old meta 字段再落盘
    clean_params: dict[str, dict] = {}
    for state_key, tactics_map in summary.items():
        clean_params[state_key] = {}
        for tactic_key, info in tactics_map.items():
            clean_params[state_key][tactic_key] = {
                k: v for k, v in info.items() if k not in ("changed", "old")
            }

    state_data = {
        "version":         1,
        "last_calibrated": datetime.now().isoformat(timespec="seconds"),
        "current_state":   market_state.load_current_state().get("current", "range"),
        "params":          clean_params,
        "history":         history,
    }
    feedback_io.atomic_write_json(STATE_FILE, state_data)
    feedback_io.snapshot_file(STATE_FILE, date_str)


def _load_state_safely() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def fit_tactic_params(date_str: str, cfg: TacticConfig | None = None) -> dict:
    """日级入口:遍历 (state × tactic) 12 桶,统计精准率 + 应用阈值微调。

    返回 summary{state: {tactic: {阈值..., weight, acc_90d, samples, status}}}。
    """
    cfg = cfg or load_config()
    lookback = int(cfg.evaluation.get("lookback_days", 90))

    # 读上版参数(若存在) 作为本次调整起点;否则用 defaults
    prev = _load_state_safely()
    prev_params = prev.get("params") or {}

    summary: dict[str, dict] = {}

    for state in STATES:
        summary[state] = {}
        bucket_accs: dict[Tactic, float | None] = {}

        # 该状态下每战法的起始阈值
        if prev_params.get(state):
            # 从上版恢复
            start_params: dict[Tactic, dict[str, float]] = {}
            for t in TACTICS:
                entry = prev_params[state].get(t, {})
                tps = {k: v for k, v in entry.items()
                       if k not in ("weight", "acc_90d", "samples", "status")}
                if not tps:
                    tps = get_defaults_for_state(state, cfg).get(t, {})
                start_params[t] = tps
        else:
            start_params = get_defaults_for_state(state, cfg)

        for tactic in TACTICS:
            eval_res = evaluate_tactic(state, tactic, date_str, lookback)
            bucket_accs[tactic] = eval_res["acc_90d"]

            old_thresholds = start_params.get(tactic, {})
            new_thresholds, reason = adjust_thresholds(
                current=old_thresholds,
                acc_90d=eval_res["acc_90d"],
                samples=eval_res["samples"],
                cfg=cfg,
                tactic=tactic,
            )

            changed = any(
                new_thresholds.get(k) != old_thresholds.get(k)
                for k in new_thresholds
            )

            summary[state][tactic] = {
                **new_thresholds,
                "acc_90d": round(eval_res["acc_90d"], 4) if eval_res["acc_90d"] is not None else None,
                "samples": eval_res["samples"],
                "status":  reason,
                "changed": changed,
                "old":     dict(old_thresholds) if changed else {},
            }

        # 状态内归一化权重
        weights = compute_resonance_weights(bucket_accs, cfg)
        for tactic in TACTICS:
            summary[state][tactic]["weight"] = weights[tactic]

    _persist_state(date_str, summary, cfg)

    # 清理 summary 中的 changed/old 字段(只用于持久化历史时)
    for state in STATES:
        for tactic in TACTICS:
            summary[state][tactic].pop("changed", None)
            summary[state][tactic].pop("old", None)

    return summary


def run(date_str: str | None = None) -> dict:
    """编排器入口:调 fit_tactic_params。异常上抛,由 orchestrator 捕获并发告警。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    cfg = load_config()
    return fit_tactic_params(date_str, cfg)
