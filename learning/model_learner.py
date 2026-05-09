"""
model_learner.py - P1 模型层:概率校准 + 错样本加权重训 + 绝对阈值 gate。

核心交付:
  - 按市场状态(bull/bear/range)分桶拟合 IsotonicRegression 校准器
  - sample_weight 查表(walk-forward 隔离 T-6),供周级加权重训用
  - abs_threshold 月度自校(基于近 30 日买入精准率)
  - calibrator pkl ↔ model pkl 通过 sha 配对,失配回退恒等

产物:
  - models/saved/calibrator_{bull,bear,range}.pkl
  - learning/model_learner.json(主状态文件,含阈值与历史)

参数全部走 learning/model_learner.yaml,代码内不硬编码任何数值。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml

from learning import feedback_io, market_state, tracker
from learning.outcome_metrics import compute_hit_tier

logger = logging.getLogger(__name__)

MarketState = Literal["bull", "bear", "range"]
STATES: tuple[MarketState, ...] = ("bull", "bear", "range")

# ── 路径常量 ────────────────────────────────────────────────
CONFIG_PATH    = Path("learning/model_learner.yaml")
STATE_FILE     = Path("learning/model_learner.json")
CALIBRATOR_DIR = Path("models/saved")
DATA_DIR       = Path("learning/data")
WALK_FORWARD_GAP_DAYS = 6  # 训练样本 T 只允许引用 outcome 日期 ≤ T-6


# ── 配置加载 ────────────────────────────────────────────────

@dataclass(frozen=True)
class LearnerConfig:
    sample_weights:     dict[str, float]
    calibration:        dict
    absolute_threshold: dict


def load_config(path: str | Path = CONFIG_PATH) -> LearnerConfig:
    """读 yaml → frozen LearnerConfig。文件缺失/字段缺失抛 ValueError。"""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in ("sample_weights", "calibration", "absolute_threshold"):
        if key not in data:
            raise ValueError(f"config missing required section: {key}")
    return LearnerConfig(
        sample_weights=dict(data["sample_weights"]),
        calibration=dict(data["calibration"]),
        absolute_threshold=dict(data["absolute_threshold"]),
    )


# ── 校准器:Identity 兜底 + sklearn IsotonicRegression 包装 ──

class IdentityCalibrator:
    """无样本/异常时的兜底:transform 返回输入本身。

    序列化兼容 pickle;字段 reason 仅供日志/JSON 记录使用。
    """

    def __init__(self, reason: str = "identity") -> None:
        self.reason = reason

    def transform(self, probs):
        # 接受 list / np.ndarray / 标量,统一返回 list[float]
        try:
            return [float(p) for p in probs]
        except TypeError:
            return [float(probs)]

    # 与 sklearn IsotonicRegression 的 predict 接口对齐
    def predict(self, probs):
        return self.transform(probs)


def _calibrator_path(state: MarketState) -> Path:
    return CALIBRATOR_DIR / f"calibrator_{state}.pkl"


def _model_sha(path: Path) -> str:
    """返回模型 pkl 文件内容的 sha256 前 12 位。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _save_calibrator(state: MarketState, calibrator, model_sha: str) -> Path:
    """以 dict 包装写入 pkl,带 model_sha 便于配对校验。"""
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
    path = _calibrator_path(state)
    payload = {"calibrator": calibrator, "model_sha": model_sha}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_calibrator(state: MarketState, current_model_sha: str | None = None):
    """加载某状态对应的校准器。

    - pkl 缺失 → IdentityCalibrator(reason="missing")
    - sha 不匹配且传入了 current_model_sha → IdentityCalibrator(reason="sha_mismatch")
    - 解码失败 → IdentityCalibrator(reason="load_error")
    永不抛异常,推荐路径绝不阻断。
    """
    path = _calibrator_path(state)
    if not path.exists():
        return IdentityCalibrator(reason="missing")
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        cal = payload.get("calibrator")
        sha = payload.get("model_sha")
        if current_model_sha and sha and sha != current_model_sha:
            return IdentityCalibrator(reason="sha_mismatch")
        return cal if cal is not None else IdentityCalibrator(reason="empty_payload")
    except Exception as e:  # pragma: no cover - 兜底,正常路径不会触发
        logger.warning("calibrator load failed for %s: %s", state, e)
        return IdentityCalibrator(reason="load_error")


def load_abs_threshold(default: float | None = None) -> float:
    """从 model_learner.json 读取当前 abs_threshold。文件缺失返回 default(None → 0.30)。

    默认值 0.30 对齐 yaml.absolute_threshold.initial,与 isotonic 实测输出上限 0.36 匹配;
    更高默认值(如 0.45)在当前模型下会导致买入信号数为 0。
    """
    if default is None:
        default = 0.30
    if not STATE_FILE.exists():
        return float(default)
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("abs_threshold")
        return float(v) if v is not None else float(default)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return float(default)


# ── 关键纯函数(单元测试主战场) ─────────────────────────────

def _classify_weight(signal: str | None, hit_tier: str | None, cfg: LearnerConfig) -> float:
    """(signal, hit_tier) → weight。供 resolve_sample_weight 内部用。"""
    weights = cfg.sample_weights
    if signal != "买入":
        # 观望/回避/None 都按 non_buy(若无对应记录则上层走 default)
        return float(weights["non_buy"])
    if hit_tier in ("miss", "weak", "good", "great"):
        return float(weights[f"buy_{hit_tier}"])
    return float(weights["default"])


def _shift_date(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def build_history_lookup(cutoff_date: str, source_dir: Path | None = None) -> dict:
    """聚合 outcome ≤ cutoff_date 的 (code) → [(date, signal, hit_tier), ...] 记录,按日期升序。

    signal 来自同日 pred_<date>.jsonl 中同 code 的记录;hit_tier 优先用 outcome 中已写入字段,
    否则按 actual_pct 即时算出。

    设计:lookup 一次构建多次查询,避免 resolve_sample_weight 在每次调用都全盘扫描。
    source_dir 默认在调用时解析 DATA_DIR(支持 monkeypatch);早绑定会导致测试 fixture 失效。
    """
    if source_dir is None:
        source_dir = DATA_DIR
    table: dict[str, list[tuple[str, str | None, str | None]]] = {}
    if not source_dir.exists():
        return table

    for outcome_path in sorted(source_dir.glob("outcome_*.jsonl")):
        date_str = outcome_path.stem.replace("outcome_", "")
        if date_str > cutoff_date:
            continue

        # 加载同日 pred 索引(code → signal),缺失时 signal 为 None
        pred_signal: dict[str, str] = {}
        pred_path = source_dir / f"pred_{date_str}.jsonl"
        if pred_path.exists():
            with open(pred_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        # 多 scene 同 code 时取 scan 优先,其次第一条出现的
                        if r.get("scene") in (None, "scan"):
                            pred_signal[r["code"]] = r.get("signal")
                        elif r["code"] not in pred_signal:
                            pred_signal[r["code"]] = r.get("signal")
                    except (json.JSONDecodeError, KeyError):
                        continue

        with open(outcome_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = r.get("code")
                if not code:
                    continue
                tier = r.get("hit_tier") or compute_hit_tier(r.get("actual_pct"))
                signal = pred_signal.get(code)
                table.setdefault(code, []).append((date_str, signal, tier))

    # 升序保留即可,resolve 时倒序扫描取最近
    return table


def resolve_sample_weight(
    date: str,
    code: str,
    cfg: LearnerConfig,
    lookup: dict | None = None,
) -> float:
    """给定训练样本 (date, code),按 pred/outcome 历史返回 sample_weight。

    walk-forward 约束:只允许引用 outcome 日期 ≤ date - WALK_FORWARD_GAP_DAYS 的记录。

    - lookup=None 时即时从磁盘构建(测试用);生产路径请预先调 build_history_lookup 复用
    - 找不到 code 任何合规记录 → cfg.sample_weights['default']
    - 找到多条,取 cutoff_date 之内最近一条
    """
    cutoff = _shift_date(date, -WALK_FORWARD_GAP_DAYS)
    if lookup is None:
        lookup = build_history_lookup(cutoff)
    records = lookup.get(code, [])
    valid = [r for r in records if r[0] <= cutoff]
    if not valid:
        return float(cfg.sample_weights["default"])
    # 倒序找最近
    valid.sort(key=lambda x: x[0], reverse=True)
    _, signal, hit_tier = valid[0]
    return _classify_weight(signal, hit_tier, cfg)


def compute_abs_threshold_delta(
    acc_30d: float | None,
    buy_signals: int,
    cfg: LearnerConfig,
) -> tuple[float, str]:
    """按 spec §3.2 表返回 (delta, reason)。

    | acc_30d <  35%  → +step (收严)
    | 35% ~ 55%       → 0    (维持)
    | acc_30d >  55%  → -step (放宽)
    | 买入样本 < min  → 0    (insufficient_samples)

    返回值 delta 是相对当前阈值的增量,需要由调用方应用边界夹逼。
    """
    cfg_t = cfg.absolute_threshold
    step = float(cfg_t["step"])
    if buy_signals < int(cfg_t.get("min_buy_signals", 10)):
        return 0.0, "insufficient_samples"
    if acc_30d is None:
        return 0.0, "no_acc_30d"
    if acc_30d < 0.35:
        return +step, f"acc_30d {acc_30d:.3f} < 35%, 收严"
    if acc_30d > 0.55:
        return -step, f"acc_30d {acc_30d:.3f} > 55%, 放宽"
    return 0.0, f"acc_30d {acc_30d:.3f} 维持区间"


def fit_calibrator_for_bucket(samples: list[tuple[float, int]], cfg: LearnerConfig):
    """单桶拟合 IsotonicRegression。

    samples: [(prob_raw, hit_binary), ...]
    样本不足 → IdentityCalibrator(reason='insufficient_data')。
    样本全 0 或全 1 → IdentityCalibrator(reason='degenerate_labels')(isotonic 在退化标签上无意义)。
    """
    min_n = int(cfg.calibration["min_samples_per_bucket"])
    if len(samples) < min_n:
        return IdentityCalibrator(reason="insufficient_data")

    xs = [float(s[0]) for s in samples]
    ys = [int(s[1]) for s in samples]
    if len(set(ys)) < 2:
        return IdentityCalibrator(reason="degenerate_labels")

    # sklearn 1.x 的 IsotonicRegression
    from sklearn.isotonic import IsotonicRegression  # 局部 import:测试时可 monkey-patch
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(xs, ys)
    return cal


def brier_score(probs: list[float], hits: list[int]) -> float | None:
    """Brier = mean((p - y)^2)。空输入 → None。"""
    if not probs or len(probs) != len(hits):
        return None
    return round(sum((p - h) ** 2 for p, h in zip(probs, hits)) / len(probs), 4)


# ── 数据装载辅助 ────────────────────────────────────────────

def _load_latest_model_path() -> Path | None:
    if not CALIBRATOR_DIR.exists():
        return None
    pkls = sorted(CALIBRATOR_DIR.glob("model_*.pkl"))
    return pkls[-1] if pkls else None


def _rebuild_features_for(code: str, day: str):
    """从 parquet 缓存复原特征,返回最后一行(对应 day 收盘后)。

    直接读 data/cache/{code}.parquet,不走 fetcher(避免回放时触发 akshare 拉取)。
    cache 缺失 / 当日之前无数据 / build_features 失败 → None。
    """
    try:
        from features.builder import build_features
    except ImportError:
        return None
    import pandas as pd
    cache_file = Path("data/cache") / f"{code}.parquet"
    if not cache_file.exists():
        return None
    try:
        df = pd.read_parquet(cache_file)
    except Exception:
        return None
    if df is None or df.empty or "date" not in df.columns:
        return None
    # 规范化 date 列为 string 进行比较
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[dates <= day]
    if df.empty:
        return None
    try:
        feats = build_features(df.copy())
    except Exception:
        return None
    if feats is None or feats.empty:
        return None
    return feats.iloc[[-1]]


def _collect_calibration_samples(
    date_str: str,
    cfg: LearnerConfig,
    model,
    feature_cols: list[str],
) -> dict[str, list[tuple[float, int]]]:
    """按 market_state 分桶收集 (rise_prob_raw, hit_binary) 三元组。

    rise_prob_raw 用最新 model 重打分历史样本,而非读 pred 中的旧概率。
    hit_binary = 1 当 hit_tier ∈ {good, great},否则 0。
    """
    lookback = int(cfg.calibration["lookback_days"])
    samples: dict[str, list[tuple[float, int]]] = {s: [] for s in STATES}

    base = datetime.strptime(date_str, "%Y-%m-%d")
    for offset in range(1, lookback + 1):
        d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
        outcome_path = DATA_DIR / f"outcome_{d}.jsonl"
        if not outcome_path.exists():
            continue
        outcomes = tracker.load_outcomes(d)
        if not outcomes:
            continue
        state = market_state.load_state_on_date(d)
        if state not in samples:
            state = market_state.DEFAULT_STATE

        for code, o in outcomes.items():
            tier = o.get("hit_tier") or compute_hit_tier(o.get("actual_pct"))
            if tier is None:
                continue
            features_row = _rebuild_features_for(code, d)
            if features_row is None:
                continue
            try:
                prob = float(model.predict(features_row[feature_cols])[0])
            except Exception:
                continue
            hit = 1 if tier in ("good", "great") else 0
            samples[state].append((prob, hit))
    return samples


# ── 主入口 ──────────────────────────────────────────────────

def fit_calibrators(date_str: str, cfg: LearnerConfig | None = None) -> dict:
    """日级入口:按市场状态拟合 3 个校准器,保存 pkl 并更新 model_learner.json。

    返回 dict:{state: {samples, brier_before, brier_after, status}}
    """
    cfg = cfg or load_config()

    model_path = _load_latest_model_path()
    if model_path is None:
        # 无模型 → 全部恒等校准 + 写入 status=no_model
        for state in STATES:
            _save_calibrator(state, IdentityCalibrator(reason="no_model"), model_sha="")
        result = {state: {"status": "no_model"} for state in STATES}
        _persist_state(date_str, result, cfg)
        return result

    # 加载模型 + 计算 sha
    from models.trainer import load_latest_model
    model = load_latest_model(str(model_path.parent))
    model_sha = _model_sha(model_path)

    # 收集样本(按 market_state 分桶)
    feature_cols = _feature_cols_from_model(model)
    bucket_samples = _collect_calibration_samples(date_str, cfg, model, feature_cols)

    # 冷启动回退
    fallback = cfg.calibration.get("cold_start_fallback", "global")
    min_n = int(cfg.calibration["min_samples_per_bucket"])
    pooled = [s for arr in bucket_samples.values() for s in arr]

    summary: dict[str, dict] = {}
    for state in STATES:
        samples = bucket_samples.get(state, [])
        used_fallback = False
        if len(samples) < min_n and fallback == "global" and len(pooled) >= min_n:
            samples = pooled
            used_fallback = True

        cal = fit_calibrator_for_bucket(samples, cfg)
        # Brier 对比:before 用原 prob,after 用校准后 prob
        if isinstance(cal, IdentityCalibrator):
            brier_before = brier_after = brier_score(
                [s[0] for s in samples], [s[1] for s in samples]
            )
            status = cal.reason
        else:
            xs = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            brier_before = brier_score(xs, ys)
            brier_after  = brier_score(list(cal.predict(xs)), ys)
            status = "ok"

        _save_calibrator(state, cal, model_sha=model_sha)
        summary[state] = {
            "samples":      len(samples),
            "brier_before": brier_before,
            "brier_after":  brier_after,
            "status":       status,
            "used_fallback_global": used_fallback,
        }

    _persist_state(date_str, summary, cfg, model_path=model_path, model_sha=model_sha)
    return summary


def _feature_cols_from_model(model) -> list[str]:
    """优先用 model.feature_name();失败回退 features.technical.FEATURE_COLS。"""
    try:
        cols = list(model.feature_name())
        if cols:
            return cols
    except Exception:
        pass
    from features.technical import FEATURE_COLS
    return list(FEATURE_COLS)


def _persist_state(
    date_str: str,
    summary: dict,
    cfg: LearnerConfig,
    model_path: Path | None = None,
    model_sha: str | None = None,
) -> None:
    """更新 learning/model_learner.json:abs_threshold + calibrators 块 + 历史。"""
    prev = _load_state_safely()
    cur_threshold = float(prev.get("abs_threshold") or cfg.absolute_threshold["initial"])

    # 计算 acc_30d 与 buy_signals 用于阈值自校
    acc_30d, buy_signals = _compute_recent_buy_accuracy(date_str, lookback_days=int(cfg.absolute_threshold["lookback_days"]))
    delta, reason = compute_abs_threshold_delta(acc_30d, buy_signals, cfg)
    new_threshold = cur_threshold + delta
    lo = float(cfg.absolute_threshold["min"])
    hi = float(cfg.absolute_threshold["max"])
    new_threshold = min(hi, max(lo, new_threshold))
    if new_threshold != cur_threshold + delta:
        reason += " (clamped)"

    threshold_history = list(prev.get("threshold_history") or [])
    threshold_history.append({
        "date":          date_str,
        "acc_30d":       round(acc_30d, 4) if acc_30d is not None else None,
        "buy_signals":   buy_signals,
        "abs_threshold": round(new_threshold, 4),
        "change":        f"{cur_threshold:.2f} → {new_threshold:.2f}: {reason}",
    })
    threshold_history = threshold_history[-90:]

    state_data = {
        "abs_threshold":   round(new_threshold, 4),
        "last_calibrated": datetime.now().isoformat(timespec="seconds"),
        "last_retrained":  prev.get("last_retrained"),
        "model_file":      str(model_path) if model_path else prev.get("model_file"),
        "model_sha":       model_sha or prev.get("model_sha"),
        "calibrators":     summary,
        "threshold_history": threshold_history,
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


def _compute_recent_buy_accuracy(date_str: str, lookback_days: int) -> tuple[float | None, int]:
    """近 lookback_days 日 signal=买入 的精准率(hit_tier ∈ {good,great})与买入数。

    无足够数据返回 (None, 0)。
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    hits = 0
    total = 0
    for offset in range(1, lookback_days + 1):
        d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
        preds = tracker.load_predictions(d)
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
            total += 1
            if tier in ("good", "great"):
                hits += 1
    if total == 0:
        return None, 0
    return hits / total, total


# ── 周级重训:加权 sample_weight ────────────────────────

def retrain_with_weights(
    df,
    feature_cols: list[str],
    cfg: LearnerConfig | None = None,
):
    """周级入口:把 sample_weight 列附加到 df,然后调 trainer.train_weighted。

    spec §4.4 关键步骤:
      1. 按训练样本日期与 code,经 build_history_lookup 一次性获取历史 lookup
      2. resolve_sample_weight 逐行查 (signal, hit_tier) → weight
         - walk-forward 约束:只读 outcome 日期 ≤ T-6
      3. df['sample_weight'] = ...
      4. lgb.Dataset(weight=...) 训练
      5. 返回 LightGBM Booster

    df 必须含:
      - 'date' (str | datetime;datetime 会被转 str)
      - 'code' (str)
      - 'label' (int 0/1)
      - feature_cols 中所有列
    """
    cfg = cfg or load_config()
    if df is None or len(df) == 0:
        raise ValueError("retrain_with_weights: 训练集为空")

    # 标准化 date 为 str
    import pandas as pd
    if "date" not in df.columns or "code" not in df.columns or "label" not in df.columns:
        raise ValueError("retrain_with_weights: df 须含 date / code / label 列")

    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["_date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        df["_date_str"] = df["date"].astype(str).str.slice(0, 10)

    # 一次性预构 lookup(整个训练集最大 cutoff = max_date - 6 天)
    max_date = df["_date_str"].max()
    cutoff = _shift_date(max_date, -WALK_FORWARD_GAP_DAYS)
    lookup = build_history_lookup(cutoff)

    df["sample_weight"] = [
        resolve_sample_weight(d, c, cfg, lookup=lookup)
        for d, c in zip(df["_date_str"], df["code"])
    ]
    df = df.drop(columns=["_date_str"])

    # 委托给 trainer
    from models.trainer import train_weighted
    return train_weighted(df, feature_cols=feature_cols, weight_col="sample_weight")


# ── 编排器入口 ──────────────────────────────────────────────

def run(date_str: str | None = None) -> dict:
    """编排器入口:调 fit_calibrators + 更新 abs_threshold,返回 summary。

    异常上抛,由 orchestrator.run_all 捕获并发告警。
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    cfg = load_config()
    return fit_calibrators(date_str, cfg)
