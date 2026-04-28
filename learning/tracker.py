"""
tracker.py - 记录每次扫描推荐的预测快照，以及收盘后的实际结果。

预测文件：learning/data/pred_YYYY-MM-DD.jsonl
结果文件：learning/data/outcome_YYYY-MM-DD.jsonl
"""
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("learning/data")


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── 预测记录 ──────────────────────────────────────────────

def log_predictions(date_str: str, records: list[dict]):
    """
    保存一批扫描推荐记录，同一天多次扫描时按 code 去重合并（后写覆盖同代码旧记录）。
    records 每项：{code, name, signal, rise_prob, pred_high, pred_low,
                   open, confidence, indicators: {...}}
    """
    _ensure()
    path = DATA_DIR / f"pred_{date_str}.jsonl"

    # 读取当日已有记录，以 code 为 key
    existing: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        existing[r["code"]] = r
                    except Exception:
                        pass

    # 用新记录覆盖同 code 旧记录
    for r in records:
        existing[r["code"]] = r

    # 写回（按 code 排序，方便人工查看）
    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(existing.values(), key=lambda x: x["code"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_predictions(date_str: str) -> list[dict]:
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
                    pass  # 跳过损坏行，不影响其余记录
    return out


def list_prediction_dates() -> list[str]:
    """返回所有有预测记录的日期列表（升序）。"""
    _ensure()
    dates = sorted(p.stem[5:] for p in DATA_DIR.glob("pred_*.jsonl"))
    return dates


# ── 实际结果记录 ──────────────────────────────────────────

def log_outcomes(date_str: str, outcomes: dict[str, dict]):
    """
    保存当日实际行情结果（覆盖当日）。
    outcomes: {code: {actual_open, actual_close, actual_high, actual_low, actual_pct}}
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
