"""
models/predictor.py - LightGBM 推理 + 信号分配。

职责:
  - load_model(model_dir): 进程级缓存加载 LightGBM Booster
  - predict(df, feature_cols, model=..., buy_top_pct=...): 单股推理,返回概率与信号
  - assign_global_signals(results, buy_top_pct): 批量全市场重新分配信号(双门槛)

P1 新增字段(rise_prob_cal / rise_prob_raw / abs_threshold_snapshot):
  - 校准器 pkl 缺失/sha 失配 → 回退 IdentityCalibrator,rise_prob_cal == rise_prob_raw
  - 不阻断推荐路径:calibrator 异常永不抛出

signal 分配规则(spec §3.1,双门槛):
  - buy_top_pct=None 时走旧绝对阈值(0.6/0.4)用于单股直查
  - buy_top_pct 传入时按近 60 日自身分位切,并受 abs_threshold 约束
"""
from __future__ import annotations

from models.trainer import load_latest_model

# ── 模型缓存 ────────────────────────────────────────────────
_model_cache: dict = {}


def load_model(model_dir: str = "models/saved"):
    """加载并进程内缓存最新 pkl。重复调用同一目录零磁盘 IO。"""
    if model_dir not in _model_cache:
        _model_cache[model_dir] = load_latest_model(model_dir)
    return _model_cache[model_dir]


# ── P1 校准器接入(惰性加载避免循环 import) ────────────────

def _apply_calibration(prob_raw: float) -> tuple[float, float]:
    """(rise_prob_raw) → (rise_prob_cal, abs_threshold_snapshot)。

    任何异常均回退到恒等映射,确保推荐路径不阻断。
    """
    try:
        from learning import model_learner, market_state
        state = market_state.load_current_state().get("current", "range")
        cal = model_learner.load_calibrator(state)
        transformed = cal.transform([prob_raw])
        prob_cal = float(transformed[0]) if transformed else float(prob_raw)
        abs_thres = model_learner.load_abs_threshold()
        return prob_cal, abs_thres
    except Exception:
        return float(prob_raw), 0.30


# ── 单股预测 ────────────────────────────────────────────────

def predict(
    df,
    feature_cols: list,
    model_dir: str = "models/saved",
    model=None,
    buy_top_pct: float | None = None,
) -> dict:
    """对 df 最新一行做预测。

    参数:
      model         - 已加载的 Booster;None 时走 load_latest_model(model_dir)
      buy_top_pct   - 单股查询模式:
                        None  → 旧绝对阈值 (0.6/0.4)
                        float → 近 60 日自身分布分位切

    返回:
      rise_prob        - 校准后概率(对外主字段,语义:命中率)
      rise_prob_raw    - 模型原始输出
      rise_prob_cal    - 校准后(与 rise_prob 相等,冗余便于调试)
      fall_prob        - 1 - rise_prob
      confidence       - 高 / 中 / 低
      signal           - 买入 / 观望 / 回避
      abs_threshold_snapshot - 当次推理时的 abs_threshold(便于回溯)
      self_rank_pct    - 近 60 日自身分布分位(分位模式时,越小越强)
      momentum         - 技术动量(预留,供调用方填充)
    """
    if model is None:
        model = load_latest_model(model_dir) if "load_latest_model" in globals() else load_model(model_dir)

    # 补齐缺失特征列（dropped 特征不在 df 中时填 0）
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    valid = df[feature_cols].dropna()
    if valid.empty:
        raise ValueError("No valid rows after dropping NaN — need more history data.")

    prob_raw = float(model.predict(valid.iloc[[-1]])[0])
    prob_cal, abs_threshold = _apply_calibration(prob_raw)

    # 近 60 行分布:用于 self_rank_pct 与分位切
    tail = valid.tail(60)
    recent_probs = sorted([float(p) for p in model.predict(tail)], reverse=True)
    n = len(recent_probs)
    rank_idx = sum(1 for p in recent_probs if p > prob_raw)
    self_rank_pct = round(rank_idx / n, 4) if n > 0 else 0.5

    if buy_top_pct is None:
        # 旧语义:绝对阈值切分(保留向后兼容)
        if prob_cal >= 0.65:
            confidence = "高"
        elif prob_cal >= 0.5:
            confidence = "中"
        else:
            confidence = "低"
        if prob_cal >= 0.6:
            signal = "买入"
        elif prob_cal >= 0.4:
            signal = "观望"
        else:
            signal = "回避"
    else:
        # 新语义:分位切 + 绝对阈值双门槛
        watch_top_pct = min(buy_top_pct * 3, 0.40)
        idx_buy   = max(0, int(n * buy_top_pct)   - 1)
        idx_watch = max(0, int(n * watch_top_pct) - 1)
        p_buy   = recent_probs[idx_buy]   if n else 1.0
        p_watch = recent_probs[idx_watch] if n else 1.0

        if prob_raw >= p_buy and prob_cal >= abs_threshold:
            signal, confidence = "买入", "高"
        elif prob_raw >= p_buy:
            # 分位够但绝对阈值不过 → 降级观望
            signal, confidence = "观望", "中"
        elif prob_raw >= p_watch:
            signal, confidence = "观望", "中"
        else:
            signal, confidence = "回避", "低"

    return {
        "rise_prob":      round(prob_cal, 4),
        "rise_prob_raw":  round(prob_raw, 4),
        "rise_prob_cal":  round(prob_cal, 4),
        "fall_prob":      round(1 - prob_cal, 4),
        "confidence":     confidence,
        "signal":         signal,
        "abs_threshold_snapshot": round(abs_threshold, 4),
        "self_rank_pct":  self_rank_pct,
    }


# ── 全市场信号重新分配(spec §3.1 双门槛) ────────────────

def assign_global_signals(results: list[dict], buy_top_pct: float) -> list[dict]:
    """就地按全市场 prob 分布切 signal,并叠加 abs_threshold 第二门槛。

    results 每条至少含 rise_prob;若有 rise_prob_raw/rise_prob_cal 优先用它们。
    修改 results 内每条 dict 的字段:signal / confidence / global_rank / global_rank_pct。

    规则(spec §3.1):
      1. rank_pct < buy_top_pct AND prob_cal >= abs_threshold → 买入
      2. rank_pct < buy_top_pct AND prob_cal <  abs_threshold → 观望(降级)
      3. buy_top_pct ≤ rank_pct < watch_top_pct               → 观望
      4. rank_pct ≥ watch_top_pct                             → 回避

    watch_top_pct = min(buy_top_pct * 3, 0.40)。
    """
    if not results:
        return results

    # 用 raw 做排名(分位基于模型原始输出,校准不影响排名)
    ranked = sorted(results, key=lambda r: r.get("rise_prob_raw", r.get("rise_prob", 0.0)), reverse=True)
    n = len(ranked)
    watch_top_pct = min(buy_top_pct * 3, 0.40)

    # 尝试读 abs_threshold(失败则回退宽松)
    abs_threshold_failed = False
    try:
        from learning import model_learner
        abs_threshold = model_learner.load_abs_threshold()
    except Exception:
        abs_threshold = 0.0  # 失败时等价于单门槛
        abs_threshold_failed = True

    # 模型质量检测：全批次最高校准概率低于最低信任水位(0.10) → 模型严重退化
    # 但当 abs_threshold 无法加载时（calibrator/状态异常）不应把低校准概率误判为模型退化，
    # 因此在 abs_threshold_failed 情况下跳过 degraded 判定。
    max_cal_prob = max(
        (r.get("rise_prob_cal", r.get("rise_prob", 0.0)) for r in ranked),
        default=0.0,
    )
    _DEGRADED_FLOOR = 0.10
    model_degraded = (not abs_threshold_failed) and (max_cal_prob < _DEGRADED_FLOOR)

    for idx, r in enumerate(ranked):
        rank_pct = (idx + 1) / n
        r["global_rank"]     = idx + 1
        r["global_rank_pct"] = round(rank_pct, 4)
        r["max_cal_prob"]    = round(max_cal_prob, 4)
        prob_cal = r.get("rise_prob_cal", r.get("rise_prob", 0.0))

        if model_degraded:
            # 退化模型：全批次无股票达到 abs_threshold，强制所有信号为观望/回避
            r["signal"], r["confidence"] = ("观望", "中") if rank_pct <= watch_top_pct else ("回避", "低")
        elif rank_pct <= buy_top_pct and prob_cal >= abs_threshold:
            r["signal"], r["confidence"] = "买入", "高"
        elif rank_pct <= buy_top_pct:
            r["signal"], r["confidence"] = "观望", "中"
        elif rank_pct <= watch_top_pct:
            r["signal"], r["confidence"] = "观望", "中"
        else:
            r["signal"], r["confidence"] = "回避", "低"

    return results
