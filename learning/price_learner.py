"""
learning/price_learner.py - P4 价位层:ATR/振幅系数网格搜索。

职责:
  - 每周日运行（内部检查 weekday）
  - 读最近 90 日买入信号 + outcome[hit_5d, max_drawdown_5d]
  - 对 4 个参数做网格搜索，分 bull/bear/range 各一套
  - 样本 < min_samples 回退默认值
  - 产物: learning/price_params.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from learning import feedback_io

logger = logging.getLogger(__name__)

CONFIG_PATH       = Path("learning/price_learner.yaml")
PRICE_PARAMS_PATH = Path("learning/price_params.json")
DATA_DIR          = Path("learning/data")


@dataclass(frozen=True)
class PriceLearnerConfig:
    lookback_days: int
    min_samples:   int
    grid:          dict
    defaults:      dict


def load_config(path: str | Path | None = None) -> PriceLearnerConfig:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise ValueError(f"price_learner.yaml 未找到: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PriceLearnerConfig(
        lookback_days=raw["lookback_days"],
        min_samples=raw["min_samples"],
        grid=raw["grid"],
        defaults=raw["defaults"],
    )


def load_price_params(state: str) -> dict:
    """读 price_params.json[state]；文件缺失或解析失败 → 从 yaml 读 defaults。永不抛。"""
    try:
        if PRICE_PARAMS_PATH.exists():
            data = json.loads(PRICE_PARAMS_PATH.read_text(encoding="utf-8"))
            if state in data:
                return data[state]
    except Exception:
        pass
    try:
        cfg = load_config()
        return dict(cfg.defaults)
    except Exception:
        return {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
                "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}


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


def _get_atr_pct(code: str, day: str) -> float | None:
    cache = Path("data/cache") / f"{code}.parquet"
    if not cache.exists():
        return None
    try:
        df = pd.read_parquet(cache)
        df["date"] = df["date"].astype(str).str[:10]
        row = df[df["date"] <= day]
        if row.empty:
            return None
        last = row.iloc[-1]
        atr   = float(last.get("atr") or 0)
        close = float(last.get("close") or 1)
        return atr / close if close > 0 else None
    except Exception:
        return None


def _build_buy_samples(date_str: str, state: str, lookback_days: int) -> pd.DataFrame:
    """返回含 hit_5d, max_drawdown_5d, atr_pct, up_ratio 的 DataFrame。"""
    from learning.tracker import load_predictions_by_scene
    end   = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=7)
    start = end - timedelta(days=lookback_days)

    rows = []
    d = start
    while d <= end:
        d_str = d.strftime("%Y-%m-%d")
        preds    = load_predictions_by_scene(d_str, "scan")
        outcomes = _load_outcomes(d_str)
        for pred in preds:
            if pred.get("signal") != "买入":
                continue
            if pred.get("market_state") != state:
                continue
            code    = pred["code"]
            outcome = outcomes.get(code)
            if outcome is None:
                continue
            hit_5d   = outcome.get("hit_5d")
            drawdown = outcome.get("max_drawdown_5d")
            if hit_5d is None or drawdown is None:
                continue
            atr_pct = _get_atr_pct(code, d_str)
            pi       = pred.get("price_info") or pred.get("dual_trade") or {}
            up_ratio = float(pi.get("up_ratio") or 2.0) / 100
            rows.append({
                "hit_5d":          float(hit_5d),
                "max_drawdown_5d": float(drawdown),
                "atr_pct":         float(atr_pct) if atr_pct else 0.015,
                "up_ratio":        up_ratio,
            })
        d += timedelta(days=1)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _arange(lo: float, hi: float, step: float) -> list[float]:
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def _score_params(
    samples: pd.DataFrame,
    short_atr_mult: float,
    short_gain_mult: float,
    long_amp_mult: float,
    long_ma60_buffer: float,
) -> float:
    """score = (wins - losses) / n。win: hit_5d >= 止盈%; loss: drawdown <= -止损%。"""
    wins = losses = 0
    for _, row in samples.iterrows():
        sell_pct = float(row["up_ratio"]) * short_gain_mult
        stop_pct = float(row["atr_pct"]) * short_atr_mult
        if float(row["hit_5d"]) >= sell_pct:
            wins += 1
        elif float(row["max_drawdown_5d"]) <= -stop_pct:
            losses += 1
    n = len(samples)
    return (wins - losses) / n if n > 0 else -1.0


def _grid_search(samples: pd.DataFrame, cfg: PriceLearnerConfig) -> dict:
    g = cfg.grid
    best_params = dict(cfg.defaults)
    best_score  = -float("inf")
    for s_atr in _arange(g["short_atr_mult"]["min"],   g["short_atr_mult"]["max"],   g["short_atr_mult"]["step"]):
        for s_gain in _arange(g["short_gain_mult"]["min"], g["short_gain_mult"]["max"], g["short_gain_mult"]["step"]):
            for l_amp in _arange(g["long_amp_mult"]["min"], g["long_amp_mult"]["max"], g["long_amp_mult"]["step"]):
                for l_buf in _arange(g["long_ma60_buffer"]["min"], g["long_ma60_buffer"]["max"], g["long_ma60_buffer"]["step"]):
                    score = _score_params(samples, s_atr, s_gain, l_amp, l_buf)
                    if score > best_score:
                        best_score  = score
                        best_params = {
                            "short_atr_mult":   s_atr,
                            "short_gain_mult":  s_gain,
                            "long_amp_mult":    l_amp,
                            "long_ma60_buffer": l_buf,
                        }
    return best_params


def fit_price_params(date_str: str, cfg: PriceLearnerConfig | None = None) -> dict:
    cfg = cfg or load_config()
    state_params: dict[str, dict] = {}
    cold_states: list[str] = []

    for state in ("bull", "bear", "range"):
        samples = _build_buy_samples(date_str, state, cfg.lookback_days)
        if len(samples) < cfg.min_samples:
            state_params[state] = dict(cfg.defaults)
            cold_states.append(state)
        else:
            state_params[state] = _grid_search(samples, cfg)

    result = {
        "version":    1,
        "updated_at": datetime.now().isoformat(),
        **state_params,
    }
    feedback_io.atomic_write_json(PRICE_PARAMS_PATH, result)
    feedback_io.write_feedback(
        "price_learner", date_str, {"cold_states": cold_states},
    )
    status = "cold_start" if len(cold_states) == 3 else "ok"
    return {"status": status, "cold_states": cold_states}


def run(date_str: str | None = None) -> dict:
    """编排器入口。非周日返回 skipped；异常上抛，由 orchestrator 捕获。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    if datetime.strptime(date_str, "%Y-%m-%d").weekday() != 6:
        return {"status": "skipped", "reason": "weekly only (Sunday)"}
    cfg = load_config()
    return fit_price_params(date_str, cfg)
