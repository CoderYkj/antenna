"""
tracker.py - 记录每次推荐的预测快照,以及收盘后的实际结果。

预测文件:learning/data/pred_YYYY-MM-DD.jsonl
结果文件:learning/data/outcome_YYYY-MM-DD.jsonl

v2 变更(P0):
  - 每条 pred 记录新增 scene 字段 ("scan" | "tactic:<key>" | "predict")
  - 同一天同一 code 不同 scene 可共存,按 (code, scene) 联合去重
  - 旧记录无 scene 字段时默认视作 "scan"(向后兼容)
  - load_predictions_by_scene(date_str, scene) 过滤工具

schema(pred 每条):
{
  "ts":        "2026-04-27T10:15:32",     # 可选,新记录带
  "scene":     "scan" | "tactic:value" | "tactic:growth" | "tactic:leader"
             | "tactic:contra" | "predict",
  "code":      str,
  "name":      str,
  "signal":    "买入" | "观望" | "回避",
  "rise_prob": float,
  "confidence":str,
  "rank_pct":  float,                     # 可选
  "market_state": str,                    # 可选,由 market_state 模块写入
  "tactic_hits": list[str],               # 可选
  "features":  dict,                      # 可选
  "dual_trade":dict,                      # 可选
  "alt":       dict,                      # 可选(P3 填充)
  "watchlist": bool,
}
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("learning/data")

SCENE_SCAN = "scan"
SCENE_PREDICT = "predict"
SCENE_TACTIC_PREFIX = "tactic:"
DEFAULT_SCENE = SCENE_SCAN  # 向后兼容


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_scene(record: dict) -> str:
    """取 scene 字段,缺失时回退 DEFAULT_SCENE。"""
    return record.get("scene") or DEFAULT_SCENE


def _record_key(record: dict) -> tuple[str, str]:
    """(code, scene) 作为联合去重 key。"""
    return (record["code"], _get_scene(record))


# ── 预测记录 ──────────────────────────────────────────────

def log_predictions(date_str: str, records: Iterable[dict]):
    """
    保存一批推荐记录。同一天同 (code, scene) 去重(后写覆盖);
    不同 scene 同 code 可共存。

    records 每项至少:{code, name, signal, rise_prob}
    推荐字段:scene, confidence, rank_pct, market_state, tactic_hits,
             features, dual_trade, alt, watchlist
    """
    _ensure()
    path = DATA_DIR / f"pred_{date_str}.jsonl"

    existing: dict[tuple[str, str], dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        existing[_record_key(r)] = r
                    except Exception:
                        pass

    for r in records:
        if "scene" not in r:
            r["scene"] = DEFAULT_SCENE
        if "ts" not in r:
            r["ts"] = datetime.now().isoformat(timespec="seconds")
        existing[_record_key(r)] = r

    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(existing.values(), key=lambda x: (x["code"], _get_scene(x))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_predictions(date_str: str) -> list[dict]:
    """加载当日所有 pred 记录(含所有 scene)。"""
    path = DATA_DIR / f"pred_{date_str}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_predictions_by_scene(date_str: str, scene: str) -> list[dict]:
    """
    按 scene 过滤。精确匹配或前缀匹配:
      - scene="scan"         → 只返回 scan
      - scene="predict"      → 只返回 predict
      - scene="tactic:value" → 只返回 tactic:value
      - scene="tactic"       → 所有 tactic:* 子类
    无 scene 字段的旧记录被视作 "scan"。
    """
    records = load_predictions(date_str)
    out = []
    for r in records:
        rs = _get_scene(r)
        matched = (scene == rs) or (scene == "tactic" and rs.startswith(SCENE_TACTIC_PREFIX))
        if matched:
            # Normalize: stamp scene onto backward-compat records missing the field
            if "scene" not in r:
                r = {**r, "scene": rs}
            out.append(r)
    return out


def list_prediction_dates() -> list[str]:
    """返回所有有预测记录的日期列表(升序)。"""
    _ensure()
    return sorted(p.stem[5:] for p in DATA_DIR.glob("pred_*.jsonl"))


# ── 实际结果记录 ──────────────────────────────────────────

def log_outcomes(date_str: str, outcomes: dict[str, dict]):
    """
    保存当日实际行情结果(覆盖当日)。
    outcomes 每项推荐字段:
      - actual_open, actual_close, actual_high, actual_low, actual_pct(必填)
      - hit_tier(P0 新增): "miss" | "weak" | "good" | "great"
      - hit_5d(P0 新增): 5 日累计涨幅(当天写入为 None,第 6 日回填)
      - max_drawdown_5d(P0 新增): 5 日最大回撤
    """
    _ensure()
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for code, r in outcomes.items():
            r["code"] = code
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_outcomes(date_str: str) -> dict[str, dict]:
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["code"]] = r
    return out
