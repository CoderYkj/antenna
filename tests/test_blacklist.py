"""tests/test_blacklist.py — 横向黑名单全套单元 + 集成测试。

覆盖:
  - load_config:文件缺失 / 正常加载 / per_state 默认回退
  - evaluate_streaks:
      * 空数据 → 空结果
      * 1 次 miss → count=1
      * 3 次连续 miss → count=3
      * miss 中间夹 good → streak 中断(从最新往回数到 good 截止)
      * 仅"买入"信号算,观望/回避忽略
  - apply_decay:清理过期条目 + 返回 expire 记录
  - Blacklist.is_blocked:
      * watchlist 自动白名单
      * (code, state) 匹配 / 不匹配
  - load_blacklist:文件缺失/损坏回退空黑名单
  - fit_blacklist:
      * streak < 阈值不拉黑
      * streak >= 阈值且 last_date 新鲜 → 拉黑
      * last_date 超过 last_activity_max_days → 跳过
      * bear 状态独立 per_state_threshold
      * 自动过期 + 新增一次执行内共存
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_cfg(tmp_path: Path, **overrides) -> Path:
    """辅助:写一份 blacklist.yaml。overrides 可覆盖默认值。"""
    yaml_path = tmp_path / "blacklist.yaml"
    defaults = {
        "streak_threshold":       3,
        "block_days":             30,
        "last_activity_max_days": 7,
        "lookback_days":          90,
        "enable_scan_filter":     True,
        "enable_tactic_filter":   False,
        "enable_predict_filter":  False,
    }
    defaults.update(overrides)
    per_state = overrides.pop("per_state_threshold", {"bull": 3, "bear": 2, "range": 3})
    content = "\n".join(f"{k}: {v}" for k, v in defaults.items())
    content += "\nper_state_threshold:\n"
    for st, thr in per_state.items():
        content += f"  {st}: {thr}\n"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _write_pred(data_dir: Path, date: str, code: str, signal: str, scene: str = "scan"):
    import json as _j
    path = data_dir / f"pred_{date}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(_j.dumps({"code": code, "name": code, "signal": signal,
                          "rise_prob": 0.5, "scene": scene}) + "\n")


def _write_outcome(data_dir: Path, date: str, code: str, hit_tier: str,
                   actual_pct: float = 0.0):
    import json as _j
    path = data_dir / f"outcome_{date}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(_j.dumps({"code": code, "actual_pct": actual_pct,
                          "hit_tier": hit_tier}) + "\n")


# ── load_config ──────────────────────────────────────────

class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        from learning import blacklist
        with pytest.raises(ValueError, match="not found"):
            blacklist.load_config(tmp_path / "no.yaml")

    def test_loads_defaults(self, tmp_path):
        from learning import blacklist
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        assert cfg.streak_threshold == 3
        assert cfg.block_days == 30
        assert cfg.enable_scan_filter is True
        assert cfg.enable_tactic_filter is False
        assert cfg.per_state_threshold["bear"] == 2

    def test_threshold_for_state(self, tmp_path):
        from learning import blacklist
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        assert blacklist._threshold_for("bear", cfg) == 2    # per_state
        assert blacklist._threshold_for("bull", cfg) == 3    # per_state
        assert blacklist._threshold_for("unknown", cfg) == 3  # fallback 默认


# ── evaluate_streaks ─────────────────────────────────────

class TestEvaluateStreaks:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        from learning import blacklist, tracker, market_state
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(tracker, "DATA_DIR", data_dir)
        # market_state 默认全 range
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "range")
        self.data_dir = data_dir

    def test_no_data_returns_empty(self, tmp_path):
        from learning import blacklist
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        result = blacklist.evaluate_streaks("2026-05-11", cfg)
        assert result == {}

    def test_three_consecutive_misses(self, tmp_path):
        """stock A 连续 3 天 buy+miss → count=3。"""
        from learning import blacklist
        for offset in range(1, 4):  # 1, 2, 3 天前
            from datetime import datetime, timedelta
            d = (datetime.strptime("2026-05-11", "%Y-%m-%d")
                 - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, "A", "买入")
            _write_outcome(self.data_dir, d, "A", "miss")
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        result = blacklist.evaluate_streaks("2026-05-11", cfg)
        assert "A|range" in result
        assert result["A|range"]["count"] == 3

    def test_streak_broken_by_good_hit(self, tmp_path):
        """从最新往前数,遇到 good 就停止。"""
        from learning import blacklist
        # T-1: miss (最新)
        # T-2: miss
        # T-3: good(break streak)
        # T-4: miss
        from datetime import datetime, timedelta
        base = datetime.strptime("2026-05-11", "%Y-%m-%d")
        for offset, tier in [(1, "miss"), (2, "miss"), (3, "good"), (4, "miss")]:
            d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, "B", "买入")
            _write_outcome(self.data_dir, d, "B", tier)
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        result = blacklist.evaluate_streaks("2026-05-11", cfg)
        assert result["B|range"]["count"] == 2   # 只数最新 2 条 miss

    def test_weak_also_counts_as_miss(self, tmp_path):
        """weak 也算"不达预期",计入 streak。"""
        from learning import blacklist
        from datetime import datetime, timedelta
        base = datetime.strptime("2026-05-11", "%Y-%m-%d")
        for offset, tier in [(1, "weak"), (2, "miss"), (3, "weak")]:
            d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, "C", "买入")
            _write_outcome(self.data_dir, d, "C", tier)
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        result = blacklist.evaluate_streaks("2026-05-11", cfg)
        assert result["C|range"]["count"] == 3

    def test_non_buy_signal_ignored(self, tmp_path):
        """观望/回避信号不计入 streak。"""
        from learning import blacklist
        from datetime import datetime, timedelta
        base = datetime.strptime("2026-05-11", "%Y-%m-%d")
        for offset, signal in [(1, "观望"), (2, "回避"), (3, "买入")]:
            d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, "D", signal)
            _write_outcome(self.data_dir, d, "D", "miss")
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        result = blacklist.evaluate_streaks("2026-05-11", cfg)
        # 只有 T-3 的"买入" miss 会被计入
        assert result.get("D|range", {}).get("count") == 1

    def test_non_scan_scene_ignored(self, tmp_path):
        """只看 scene=scan 的预测。"""
        from learning import blacklist
        from datetime import datetime, timedelta
        base = datetime.strptime("2026-05-11", "%Y-%m-%d")
        for offset in range(1, 4):
            d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, "E", "买入", scene="tactic:value")
            _write_outcome(self.data_dir, d, "E", "miss")
        cfg = blacklist.load_config(_write_cfg(tmp_path))
        result = blacklist.evaluate_streaks("2026-05-11", cfg)
        assert "E|range" not in result  # tactic 场景不算


# ── apply_decay ──────────────────────────────────────────

class TestApplyDecay:
    def test_all_valid_no_expiry(self):
        from learning.blacklist import apply_decay
        bl = {
            "A|range": {"until": "2026-06-15", "streak_count": 3},
            "B|bear":  {"until": "2026-06-20", "streak_count": 2},
        }
        survivors, expired = apply_decay(bl, "2026-05-11")
        assert len(survivors) == 2
        assert expired == []

    def test_expired_removed(self):
        from learning.blacklist import apply_decay
        bl = {
            "A|range": {"until": "2026-04-01", "streak_count": 3},   # 过期
            "B|bull":  {"until": "2026-06-20", "streak_count": 2},   # 存活
        }
        survivors, expired = apply_decay(bl, "2026-05-11")
        assert "A|range" not in survivors
        assert "B|bull" in survivors
        assert len(expired) == 1
        assert expired[0]["action"] == "expire"
        assert expired[0]["code"] == "A"
        assert expired[0]["state"] == "range"

    def test_today_boundary_still_valid(self):
        """until == today 仍视为有效(≥ today)。"""
        from learning.blacklist import apply_decay
        bl = {"A|range": {"until": "2026-05-11"}}
        survivors, expired = apply_decay(bl, "2026-05-11")
        assert "A|range" in survivors
        assert expired == []

    def test_empty_blacklist(self):
        from learning.blacklist import apply_decay
        survivors, expired = apply_decay({}, "2026-05-11")
        assert survivors == {}
        assert expired == []

    def test_none_blacklist(self):
        from learning.blacklist import apply_decay
        survivors, expired = apply_decay(None, "2026-05-11")
        assert survivors == {}
        assert expired == []


# ── Blacklist 查询 ───────────────────────────────────────

class TestBlacklistQuery:
    def test_is_blocked_true_when_key_matches(self):
        from learning.blacklist import Blacklist
        bl = Blacklist(entries={"A|range": {"until": "2026-06-10"}})
        assert bl.is_blocked("A", "range") is True

    def test_is_blocked_false_different_state(self):
        from learning.blacklist import Blacklist
        bl = Blacklist(entries={"A|range": {"until": "2026-06-10"}})
        assert bl.is_blocked("A", "bull") is False

    def test_is_blocked_false_different_code(self):
        from learning.blacklist import Blacklist
        bl = Blacklist(entries={"A|range": {"until": "2026-06-10"}})
        assert bl.is_blocked("B", "range") is False

    def test_watchlist_override(self):
        """watchlist 内的股票 is_blocked 始终 False。"""
        from learning.blacklist import Blacklist
        bl = Blacklist(
            entries={"A|range": {"until": "2026-06-10"}},
            watchlist={"A"},
        )
        assert bl.is_blocked("A", "range") is False

    def test_empty_blacklist(self):
        from learning.blacklist import Blacklist
        bl = Blacklist(entries={})
        assert bl.is_blocked("anything", "range") is False


# ── load_blacklist 兜底 ──────────────────────────────────

class TestLoadBlacklist:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        from learning import blacklist
        monkeypatch.setattr(blacklist, "STATE_FILE", tmp_path / "no.json")
        bl = blacklist.load_blacklist()
        assert len(bl) == 0
        assert bl.is_blocked("any", "range") is False

    def test_corrupt_file_returns_empty(self, tmp_path, monkeypatch):
        from learning import blacklist
        bad = tmp_path / "bad.json"
        bad.write_text("not-json{", encoding="utf-8")
        monkeypatch.setattr(blacklist, "STATE_FILE", bad)
        bl = blacklist.load_blacklist()
        assert len(bl) == 0

    def test_valid_file_loads_entries(self, tmp_path, monkeypatch):
        from learning import blacklist
        f = tmp_path / "bl.json"
        f.write_text(json.dumps({
            "entries": {
                "A|range": {"until": "2026-06-10", "streak_count": 3},
            }
        }), encoding="utf-8")
        monkeypatch.setattr(blacklist, "STATE_FILE", f)
        bl = blacklist.load_blacklist()
        assert bl.is_blocked("A", "range") is True


# ── fit_blacklist 端到端 ─────────────────────────────────

class TestFitBlacklist:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        from learning import blacklist, tracker, market_state, feedback_io
        monkeypatch.setattr(blacklist, "STATE_FILE", tmp_path / "bl.json")
        monkeypatch.setattr(blacklist, "CONFIG_PATH", _write_cfg(tmp_path))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(tracker, "DATA_DIR", data_dir)
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "range")
        # watchlist 置空
        monkeypatch.setattr(blacklist, "_load_watchlist", lambda: set())
        self.data_dir = data_dir
        self.tmp_path = tmp_path

    def _write_streak(self, code, miss_count, state_on_date_callable=None):
        """辅助:给某股写 miss_count 天连续 miss,date 是最近 miss_count 天。"""
        from datetime import datetime, timedelta
        base = datetime.strptime("2026-05-11", "%Y-%m-%d")
        for offset in range(1, miss_count + 1):
            d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, code, "买入")
            _write_outcome(self.data_dir, d, code, "miss")

    def test_streak_below_threshold_not_blocked(self, tmp_path):
        from learning import blacklist
        self._write_streak("X", miss_count=2)  # < 阈值 3
        result = blacklist.fit_blacklist("2026-05-11")
        assert result["added"] == 0
        bl = blacklist.load_blacklist()
        assert bl.is_blocked("X", "range") is False

    def test_streak_meets_threshold_blocked(self, tmp_path):
        from learning import blacklist
        self._write_streak("Y", miss_count=3)
        result = blacklist.fit_blacklist("2026-05-11")
        assert result["added"] == 1
        bl = blacklist.load_blacklist()
        assert bl.is_blocked("Y", "range") is True

    def test_last_miss_too_old_skipped(self, monkeypatch, tmp_path):
        """最后一次 miss 超过 last_activity_max_days=7 天 → 不拉黑。"""
        from learning import blacklist
        from datetime import datetime, timedelta
        base = datetime.strptime("2026-05-11", "%Y-%m-%d")
        # 3 次 miss 都在 30 天前
        for offset in range(30, 33):
            d = (base - timedelta(days=offset)).strftime("%Y-%m-%d")
            _write_pred(self.data_dir, d, "Z", "买入")
            _write_outcome(self.data_dir, d, "Z", "miss")
        result = blacklist.fit_blacklist("2026-05-11")
        assert result["added"] == 0
        bl = blacklist.load_blacklist()
        assert bl.is_blocked("Z", "range") is False

    def test_watchlist_override_in_fit(self, tmp_path, monkeypatch):
        """fit 阶段就过滤 watchlist,连加入都不发生。"""
        from learning import blacklist
        monkeypatch.setattr(blacklist, "_load_watchlist", lambda: {"W"})
        self._write_streak("W", miss_count=5)
        result = blacklist.fit_blacklist("2026-05-11")
        assert result["added"] == 0
        bl = blacklist.load_blacklist()
        assert bl.is_blocked("W", "range") is False

    def test_expired_entries_cleaned(self, tmp_path):
        from learning import blacklist
        # 预置一个已过期条目
        blacklist.STATE_FILE.write_text(json.dumps({
            "entries": {
                "OLD|range": {"until": "2026-01-01", "streak_count": 5},
            }
        }), encoding="utf-8")
        result = blacklist.fit_blacklist("2026-05-11")
        assert result["expired"] == 1
        bl = blacklist.load_blacklist()
        assert bl.is_blocked("OLD", "range") is False

    def test_bear_state_uses_per_state_threshold(self, monkeypatch, tmp_path):
        """bear 状态下 per_state_threshold=2,streak=2 即可拉黑。"""
        from learning import blacklist, market_state
        monkeypatch.setattr(market_state, "load_state_on_date", lambda d: "bear")
        # 2 次 miss
        self._write_streak("BEAR", miss_count=2)
        result = blacklist.fit_blacklist("2026-05-11")
        # bear threshold=2, streak=2 → 满足
        assert result["added"] == 1

    def test_persists_history_snapshot(self, tmp_path):
        from learning import blacklist
        self._write_streak("H", miss_count=4)
        blacklist.fit_blacklist("2026-05-11")
        data = json.loads(blacklist.STATE_FILE.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "H|range" in data["entries"]
        # history 有 add 记录
        adds = [h for h in data["history"] if h.get("action") == "add"]
        assert len(adds) >= 1
        assert adds[0]["code"] == "H"
        assert adds[0]["streak"] == 4
