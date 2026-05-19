"""
learning/feature_learner.py - P3 特征层:Permutation Importance + 动态淘汰。

职责:
  - 每周日运行（内部检查 weekday，非周日返回 skipped）
  - 对最新 LightGBM 模型跑 sklearn permutation_importance
  - 连续 candidate_weeks 低重要 → candidate_drop；连续 drop_weeks → dropped
  - alt_data 特征按 IC 阈值决定是否加入 active
  - 产物: learning/feature_weights.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from features.technical import FEATURE_COLS, get_active_feature_cols
from learning import feedback_io

logger = logging.getLogger(__name__)

CONFIG_PATH          = Path("learning/feature_learner.yaml")
FEATURE_WEIGHTS_PATH = Path("learning/feature_weights.json")
DATA_DIR             = Path("learning/data")
ALT_COLS             = [
    "main_net_in_1d", "main_net_in_5d",
    "dragon_top_cnt_10d", "sector_heat_rank", "north_hold_chg_5d",
]


@dataclass(frozen=True)
class FeatureLearnerConfig:
    n_repeats:            int
    lookback_days:        int
    importance_threshold: float
    candidate_weeks:      int
    drop_weeks:           int
    ic_threshold:         float
    ic_lookback_days:     int


def load_config(path: str | Path | None = None) -> FeatureLearnerConfig:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise ValueError(f"feature_learner.yaml 未找到: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FeatureLearnerConfig(
        n_repeats=raw["permutation"]["n_repeats"],
        lookback_days=raw["permutation"]["lookback_days"],
        importance_threshold=raw["drop"]["importance_threshold"],
        candidate_weeks=raw["drop"]["candidate_weeks"],
        drop_weeks=raw["drop"]["drop_weeks"],
        ic_threshold=raw["alt_data"]["ic_threshold"],
        ic_lookback_days=raw["alt_data"]["ic_lookback_days"],
    )


def _load_feature_weights() -> dict:
    if not FEATURE_WEIGHTS_PATH.exists():
        return {"candidate_drop": {}, "dropped": [], "candidates": [], "importance": {}}
    try:
        return json.loads(FEATURE_WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"candidate_drop": {}, "dropped": [], "candidates": [], "importance": {}}


def _update_drop_tracker(
    state: dict,
    importances: dict[str, float],
    threshold: float,
    candidate_weeks: int,
    drop_weeks: int,
) -> dict:
    """返回更新后的 {candidate_drop, dropped} state（纯函数）。"""
    candidate_drop: dict = dict(state.get("candidate_drop") or {})
    dropped: list = list(state.get("dropped") or [])

    for feat, score in importances.items():
        if feat in dropped:
            continue
        if score < threshold:
            entry = candidate_drop.get(feat, {"weeks_below": 0})
            entry = {"weeks_below": entry["weeks_below"] + 1}
            if entry["weeks_below"] >= drop_weeks:
                dropped.append(feat)
                candidate_drop.pop(feat, None)
            else:
                candidate_drop[feat] = entry
        else:
            candidate_drop.pop(feat, None)

    return {"candidate_drop": candidate_drop, "dropped": dropped}


def _rebuild_features_row(code: str, day: str) -> dict | None:
    """从 parquet 缓存复原 day 当天特征（dict 形式）。"""
    from features.builder import build_features
    cache = Path("data/cache") / f"{code}.parquet"
    if not cache.exists():
        return None
    try:
        df = pd.read_parquet(cache)
        if df is None or df.empty or "date" not in df.columns:
            return None
        df["date"] = df["date"].astype(str).str[:10]
        row_df = df[df["date"] <= day]
        if row_df.empty:
            return None
        row_df = build_features(row_df)
        last = row_df.iloc[-1]
        return {col: last.get(col) for col in FEATURE_COLS}
    except Exception:
        return None


def _load_outcomes(date_str: str) -> dict[str, dict]:
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                result[r["code"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def _load_alt_for(code: str, date_str: str) -> dict:
    path = Path("data/cache/alt") / f"{date_str}.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        row = df[df["code"] == code]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
    except Exception:
        return {}


def _build_sample_matrix(date_str: str, lookback_days: int) -> tuple:
    """返回 (X: pd.DataFrame, y: pd.Series, alt_df: pd.DataFrame)。"""
    from learning.tracker import load_predictions_by_scene
    end = datetime.strptime(date_str, "%Y-%m-%d")
    start = end - timedelta(days=lookback_days)
    rows = []
    d = start
    while d <= end:
        d_str = d.strftime("%Y-%m-%d")
        preds = {r["code"]: r for r in load_predictions_by_scene(d_str, "scan")}
        outcomes = _load_outcomes(d_str)
        for code, pred in preds.items():
            outcome = outcomes.get(code)
            if outcome is None:
                continue
            hit = 1 if (outcome.get("actual_pct") or 0.0) >= 0.02 else 0
            feats = _rebuild_features_row(code, d_str)
            if feats is None:
                continue
            row = feats.copy()
            row["hit"] = hit
            alt = _load_alt_for(code, d_str)
            for col in ALT_COLS:
                row[col] = alt.get(col, 0.0) if alt.get(col) is not None else 0.0
            rows.append(row)
        d += timedelta(days=1)

    if not rows:
        return pd.DataFrame(), pd.Series([], dtype=int), pd.DataFrame()
    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS]
    y = df["hit"]
    alt_df = df[["hit"] + [c for c in ALT_COLS if c in df.columns]]
    return X, y, alt_df


class _LGBEstimator:
    """sklearn 兼容包装，使 permutation_importance 可用 LightGBM Booster。"""
    def __init__(self, booster):
        self._b = booster

    def predict_proba(self, X):
        import numpy as np
        preds = self._b.predict(np.asarray(X, dtype=float))
        return np.column_stack([1 - preds, preds])


def _compute_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int) -> dict[str, float]:
    from sklearn.inspection import permutation_importance
    est = _LGBEstimator(model)
    result = permutation_importance(
        est, X.values, y.values,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )
    return {col: float(v) for col, v in zip(X.columns, result.importances_mean)}


def _compute_ic(alt_df: pd.DataFrame, ic_threshold: float) -> list[str]:
    active_alt = []
    for col in ALT_COLS:
        if col not in alt_df.columns or "hit" not in alt_df.columns:
            continue
        valid = alt_df[[col, "hit"]].dropna()
        if len(valid) < 20:
            continue
        ic = abs(float(valid[col].corr(valid["hit"])))
        if ic >= ic_threshold:
            active_alt.append(col)
    return active_alt


def _load_latest_model(model_dir: str = "models/saved"):
    """加载最新的 LightGBM 模型 pkl。"""
    import pickle
    pkl_files = sorted(Path(model_dir).glob("*.pkl"))
    # 优先非 calibrator
    model_files = [f for f in pkl_files if "calibrat" not in f.name.lower()]
    if not model_files:
        raise FileNotFoundError(f"models/saved 中无 pkl 文件")
    with open(model_files[-1], "rb") as f:
        return pickle.load(f)


def fit_feature_weights(date_str: str, cfg: FeatureLearnerConfig | None = None) -> dict:
    cfg = cfg or load_config()
    try:
        model = _load_latest_model()
    except Exception:
        return {"status": "no_model"}

    X, y, alt_df = _build_sample_matrix(date_str, cfg.lookback_days)
    if len(X) < 30:
        return {"status": "cold_start", "samples": len(X)}

    importances = _compute_importance(model, X, y, cfg.n_repeats)

    current = _load_feature_weights()
    new_state = _update_drop_tracker(
        current, importances,
        cfg.importance_threshold, cfg.candidate_weeks, cfg.drop_weeks,
    )

    active_alt = _compute_ic(alt_df, cfg.ic_threshold)
    active = [f for f in FEATURE_COLS if f not in new_state["dropped"]] + active_alt

    state = {
        "version":    1,
        "updated_at": datetime.now().isoformat(),
        "active":     active,
        "candidate_drop": new_state["candidate_drop"],
        "dropped":    new_state["dropped"],
        "candidates": [c for c in ALT_COLS if c not in active_alt],
        "importance": {k: round(v, 4) for k, v in importances.items()},
    }
    feedback_io.atomic_write_json(FEATURE_WEIGHTS_PATH, state)
    feedback_io.write_feedback(
        "feature_learner", date_str,
        {"active": len(active), "dropped": len(new_state["dropped"])},
    )
    return {"status": "ok", "active": len(active), "dropped": len(new_state["dropped"])}


def run(date_str: str | None = None) -> dict:
    """编排器入口。非周日返回 skipped；异常上抛，由 orchestrator 捕获。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    if datetime.strptime(date_str, "%Y-%m-%d").weekday() != 6:  # 0=Mon, 6=Sun
        return {"status": "skipped", "reason": "weekly only (Sunday)"}
    cfg = load_config()
    return fit_feature_weights(date_str, cfg)
