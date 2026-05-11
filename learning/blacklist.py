"""
blacklist.py - 横向黑名单:同 (code, state) 连错 N 次 → 拉黑 M 天。

职责:
  - 按 (code, state) 聚合近 90 日 pred[scene=scan] × outcome
  - streak ≥ threshold 且最近 miss 在 last_activity_max_days 内 → 加入黑名单
  - 自动过期:expires < today 自动清理
  - watchlist 自动白名单(用户主动跟踪的股票永不拉黑)
  - 推荐路径 cmd_scan_bot 在 assign_global_signals 前调 is_blocked 过滤

产物: learning/blacklist.json(entries + history + version)
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
STATES: tuple[MarketState, ...] = ("bull", "bear", "range")
SCAN_SCENE = "scan"

# ── 路径常量 ────────────────────────────────────────────────
CONFIG_PATH = Path("learning/blacklist.yaml")
STATE_FILE  = Path("learning/blacklist.json")


# ── 配置加载 ────────────────────────────────────────────────

@dataclass(frozen=True)
class BlacklistConfig:
    streak_threshold:       int
    block_days:             int
    last_activity_max_days: int
    lookback_days:          int
    enable_scan_filter:     bool
    enable_tactic_filter:   bool
    enable_predict_filter:  bool
    per_state_threshold:    dict[str, int]


def load_config(path: str | Path | None = None) -> BlacklistConfig:
    """读 yaml → frozen BlacklistConfig。文件缺失抛 ValueError。

    path=None 时运行时解析 CONFIG_PATH(支持 monkeypatch)。
    """
    if path is None:
        path = CONFIG_PATH
    p = Path(path)
    if not p.exists():
        raise ValueError(f"blacklist config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return BlacklistConfig(
        streak_threshold=       int(data.get("streak_threshold",       3)),
        block_days=             int(data.get("block_days",             30)),
        last_activity_max_days= int(data.get("last_activity_max_days", 7)),
        lookback_days=          int(data.get("lookback_days",          90)),
        enable_scan_filter=     bool(data.get("enable_scan_filter",    True)),
        enable_tactic_filter=   bool(data.get("enable_tactic_filter",  False)),
        enable_predict_filter=  bool(data.get("enable_predict_filter", False)),
        per_state_threshold=    dict(data.get("per_state_threshold")   or {}),
    )


def _threshold_for(state: str, cfg: BlacklistConfig) -> int:
    """取某状态的 streak 阈值;per_state 优先,否则用默认。"""
    return int(cfg.per_state_threshold.get(state, cfg.streak_threshold))


# ── 核心算法 ────────────────────────────────────────────────

def _entry_key(code: str, state: str) -> str:
    """(code, state) 组合键,用 '|' 分隔以兼容 JSON 字符串 key。"""
    return f"{code}|{state}"


def _shift_date(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d")
            + timedelta(days=days)).strftime("%Y-%m-%d")


def evaluate_streaks(date_str: str, cfg: BlacklistConfig) -> dict[str, dict]:
    """扫近 lookback_days 日 pred[scene=scan] × outcome,返回
      {(code|state): {count: int, last_date: str}}
    count 是"从最近一条 pred 向前数,连续 miss/weak 直到遇到 good/great 的次数"。
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    events_by_key: dict[str, list[dict]] = {}

    for offset in range(1, cfg.lookback_days + 1):
        d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
        state = market_state.load_state_on_date(d)
        if state not in STATES:
            state = market_state.DEFAULT_STATE
        preds = tracker.load_predictions_by_scene(d, SCAN_SCENE)
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
            key = _entry_key(p["code"], state)
            events_by_key.setdefault(key, []).append({
                "date":   d,
                "tier":   tier,
                "signal": p["signal"],
            })

    # 对每 key,按日期降序排列,从最近一条向前数连续 miss/weak
    result: dict[str, dict] = {}
    for key, events in events_by_key.items():
        events.sort(key=lambda x: x["date"], reverse=True)
        streak = 0
        for e in events:
            if e["tier"] in ("miss", "weak"):
                streak += 1
            else:
                break
        if streak > 0:
            result[key] = {
                "count":     streak,
                "last_date": events[0]["date"],
            }
    return result


def apply_decay(blacklist: dict, today: str) -> tuple[dict, list[dict]]:
    """清理 expires < today 的条目。返回 (存活 blacklist, 过期 history 记录列表)。"""
    survivors: dict = {}
    expired: list[dict] = []
    for key, entry in (blacklist or {}).items():
        until = entry.get("until", "")
        if until >= today:
            survivors[key] = entry
        else:
            code, _, state = key.partition("|")
            expired.append({
                "date":   today,
                "action": "expire",
                "code":   code,
                "state":  state,
                "until":  until,
            })
    return survivors, expired


def _load_watchlist() -> set[str]:
    """从 config.yaml 读 watchlist。失败返回空集。"""
    try:
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return set((cfg.get("universe") or {}).get("watchlist") or [])
    except Exception:
        return set()


# ── Blacklist 查询接口(供 predict_cmd 调用,永不抛) ──────

class Blacklist:
    """in-memory 查询接口,is_blocked 永不抛异常。"""

    def __init__(self, entries: dict, watchlist: set[str] | None = None):
        self._entries = entries or {}
        self._watchlist = watchlist or set()

    def is_blocked(self, code: str, state: str) -> bool:
        if code in self._watchlist:
            return False
        key = _entry_key(code, state)
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def load_blacklist() -> Blacklist:
    """读 blacklist.json + config.yaml watchlist,返回 Blacklist 对象。

    文件缺失/损坏 → 空黑名单;永不抛。
    """
    entries: dict = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries") or {}
        except (json.JSONDecodeError, OSError):
            entries = {}
    return Blacklist(entries=entries, watchlist=_load_watchlist())


# ── 主入口 ──────────────────────────────────────────────────

def _load_state_safely() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def fit_blacklist(date_str: str, cfg: BlacklistConfig | None = None) -> dict:
    """主编排入口:
      1. 加载上版 blacklist + apply_decay
      2. evaluate_streaks
      3. streak ≥ threshold 且 last_date 在 last_activity_max_days 内 → 加入
      4. watchlist 白名单 override
      5. 写盘 + 7 份历史快照

    返回 summary {added: list, expired: list, kept: int, total: int}。
    """
    cfg = cfg or load_config()
    today = date_str
    prev = _load_state_safely()
    blacklist = prev.get("entries") or {}
    history = list(prev.get("history") or [])

    # 1. 清理过期
    blacklist, expired_records = apply_decay(blacklist, today)
    history.extend(expired_records)

    # 2. 评估新增
    watchlist = _load_watchlist()
    streaks = evaluate_streaks(today, cfg)
    added_records: list[dict] = []

    for key, info in streaks.items():
        code, _, state = key.partition("|")
        # 白名单 override
        if code in watchlist:
            continue
        # 阈值检查
        threshold = _threshold_for(state, cfg)
        if info["count"] < threshold:
            continue
        # 活跃性检查:最后一次 miss 必须在 last_activity_max_days 内
        days_since = (datetime.strptime(today, "%Y-%m-%d")
                      - datetime.strptime(info["last_date"], "%Y-%m-%d")).days
        if days_since > cfg.last_activity_max_days:
            continue
        # 已在黑名单内 → 刷新 until(防止刚 expire 又立即触发时状态不一致)
        until = _shift_date(today, cfg.block_days)
        reason = f"{state} 状态下连续 {info['count']} 次 buy+miss/weak"
        blacklist[key] = {
            "until":        until,
            "streak_count": info["count"],
            "last_miss":    info["last_date"],
            "added_at":     today,
            "reason":       reason,
        }
        added_records.append({
            "date":    today,
            "action":  "add",
            "code":    code,
            "state":   state,
            "streak":  info["count"],
            "until":   until,
        })

    history.extend(added_records)
    history = history[-200:]

    # 3. 落盘
    state_data = {
        "version":    1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "entries":    blacklist,
        "history":    history,
    }
    feedback_io.atomic_write_json(STATE_FILE, state_data)
    feedback_io.snapshot_file(STATE_FILE, today)

    return {
        "added":   len(added_records),
        "expired": len(expired_records),
        "total":   len(blacklist),
    }


def run(date_str: str | None = None) -> dict:
    """编排器入口。异常上抛,由 orchestrator 捕获发告警。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    cfg = load_config()
    return fit_blacklist(date_str, cfg)
