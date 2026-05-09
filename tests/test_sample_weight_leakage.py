"""tests/test_sample_weight_leakage.py — walk-forward 隔离硬性红绿。

resolve_sample_weight 必须只引用 outcome 日期 ≤ T - WALK_FORWARD_GAP_DAYS 的记录。
任何"用未来信息训练当前样本"的实现都应被这组测试卡死。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── fixture ──────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path):
    from learning import model_learner
    yaml_path = tmp_path / "model_learner.yaml"
    yaml_path.write_text(
        """
sample_weights:
  buy_miss:   2.0
  buy_weak:   1.2
  buy_good:   1.0
  buy_great:  1.5
  non_buy:    0.8
  default:    1.0
calibration:
  lookback_days:          90
  min_samples_per_bucket: 50
  cold_start_fallback:    global
absolute_threshold:
  initial:         0.45
  min:             0.30
  max:             0.60
  step:            0.02
  lookback_days:   30
  min_buy_signals: 10
""",
        encoding="utf-8",
    )
    return model_learner.load_config(yaml_path)


def _write_pred(dir_: Path, date: str, code: str, signal: str) -> None:
    """写一条 pred 记录。"""
    path = dir_ / f"pred_{date}.jsonl"
    record = {"code": code, "name": code, "signal": signal, "rise_prob": 0.5, "scene": "scan"}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_outcome(dir_: Path, date: str, code: str, hit_tier: str, actual_pct: float = 0.0) -> None:
    """写一条 outcome 记录(显式 hit_tier)。"""
    path = dir_ / f"outcome_{date}.jsonl"
    record = {
        "code": code, "actual_open": 10.0, "actual_close": 10.0 * (1 + actual_pct / 100),
        "actual_high": 10.5, "actual_low": 9.5, "actual_pct": actual_pct, "hit_tier": hit_tier,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """把 model_learner.DATA_DIR 重定向到 tmp_path,避免触碰生产 learning/data/。"""
    from learning import model_learner
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(model_learner, "DATA_DIR", d)
    return d


# ── 红绿测试:T-3(违规)与 T-7(合规) ─────────────────────

class TestWalkForwardLeakage:
    """spec §4.3:训练样本 T 只允许引用 outcome 日期 ≤ T - 6。"""

    def test_record_within_gap_is_ignored(self, cfg, isolated_data_dir):
        """T-3 的记录在违规窗口内,resolve_sample_weight 应忽略它。"""
        from learning.model_learner import resolve_sample_weight
        # T = 2026-04-29,T-3 = 2026-04-26(在 6 天 gap 内,违规)
        _write_pred(isolated_data_dir, "2026-04-26", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-26", "000001", "miss")
        weight = resolve_sample_weight("2026-04-29", "000001", cfg)
        # 违规记录被忽略 → default
        assert weight == cfg.sample_weights["default"]

    def test_record_at_gap_boundary_is_used(self, cfg, isolated_data_dir):
        """T-6 是边界,允许引用(spec:日期 ≤ T - 6)。"""
        from learning.model_learner import resolve_sample_weight
        _write_pred(isolated_data_dir, "2026-04-23", "000001", "买入")  # 2026-04-29 - 6
        _write_outcome(isolated_data_dir, "2026-04-23", "000001", "miss")
        weight = resolve_sample_weight("2026-04-29", "000001", cfg)
        assert weight == cfg.sample_weights["buy_miss"]

    def test_record_well_before_gap_is_used(self, cfg, isolated_data_dir):
        """T-7 远在 gap 之外,正常引用。"""
        from learning.model_learner import resolve_sample_weight
        _write_pred(isolated_data_dir, "2026-04-22", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-22", "000001", "great")
        weight = resolve_sample_weight("2026-04-29", "000001", cfg)
        assert weight == cfg.sample_weights["buy_great"]

    def test_only_violation_records_yields_default(self, cfg, isolated_data_dir):
        """只有违规记录时返回 default,不应"凑合"用违规数据。"""
        from learning.model_learner import resolve_sample_weight
        for offset in [1, 2, 3, 4, 5]:
            d = (
                __import__("datetime").datetime.strptime("2026-04-29", "%Y-%m-%d")
                - __import__("datetime").timedelta(days=offset)
            ).strftime("%Y-%m-%d")
            _write_pred(isolated_data_dir, d, "000001", "买入")
            _write_outcome(isolated_data_dir, d, "000001", "miss")
        weight = resolve_sample_weight("2026-04-29", "000001", cfg)
        assert weight == cfg.sample_weights["default"]

    def test_legal_record_overrides_violations(self, cfg, isolated_data_dir):
        """同 code 既有违规又有合规记录时,只用合规。"""
        from learning.model_learner import resolve_sample_weight
        # T-3 违规:miss(weight 2.0,如被错误使用会盖过 great 1.5)
        _write_pred(isolated_data_dir, "2026-04-26", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-26", "000001", "miss")
        # T-7 合规:great
        _write_pred(isolated_data_dir, "2026-04-22", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-22", "000001", "great")
        weight = resolve_sample_weight("2026-04-29", "000001", cfg)
        assert weight == cfg.sample_weights["buy_great"]

    def test_picks_most_recent_legal_record(self, cfg, isolated_data_dir):
        """有多条合规记录时,取最近的一条。"""
        from learning.model_learner import resolve_sample_weight
        # T-30:miss(更老)
        _write_pred(isolated_data_dir, "2026-03-30", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-03-30", "000001", "miss")
        # T-7:great(更新)
        _write_pred(isolated_data_dir, "2026-04-22", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-22", "000001", "great")
        weight = resolve_sample_weight("2026-04-29", "000001", cfg)
        assert weight == cfg.sample_weights["buy_great"]


class TestLookupTableConstruction:
    """build_history_lookup 的核心契约:cutoff 范围 + 多 scene 处理。"""

    def test_cutoff_filters_strict_less_or_equal(self, cfg, isolated_data_dir):
        from learning.model_learner import build_history_lookup
        _write_pred(isolated_data_dir, "2026-04-22", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-22", "000001", "miss")
        _write_pred(isolated_data_dir, "2026-04-23", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-23", "000001", "great")
        _write_pred(isolated_data_dir, "2026-04-24", "000001", "买入")
        _write_outcome(isolated_data_dir, "2026-04-24", "000001", "weak")

        # cutoff = 2026-04-23 → 应包含 22 和 23,排除 24
        lookup = build_history_lookup("2026-04-23", source_dir=isolated_data_dir)
        dates = [r[0] for r in lookup["000001"]]
        assert "2026-04-22" in dates
        assert "2026-04-23" in dates
        assert "2026-04-24" not in dates

    def test_no_pred_record_yields_signal_none(self, cfg, isolated_data_dir):
        """outcome 存在但当日 pred 缺失 → signal=None,_classify_weight 走 non_buy。"""
        from learning.model_learner import build_history_lookup, _classify_weight
        _write_outcome(isolated_data_dir, "2026-04-22", "000099", "great")
        lookup = build_history_lookup("2026-04-29", source_dir=isolated_data_dir)
        assert "000099" in lookup
        date, signal, tier = lookup["000099"][0]
        assert signal is None
        assert tier == "great"
        assert _classify_weight(signal, tier, cfg) == cfg.sample_weights["non_buy"]

    def test_actual_pct_falls_back_to_compute_hit_tier(self, isolated_data_dir):
        """outcome 缺 hit_tier 字段时,从 actual_pct 即时算。"""
        from learning.model_learner import build_history_lookup
        # 写一条不带 hit_tier、只有 actual_pct 的 outcome
        path = isolated_data_dir / "outcome_2026-04-22.jsonl"
        path.write_text(
            json.dumps({"code": "000999", "actual_pct": 3.5}) + "\n",
            encoding="utf-8",
        )
        lookup = build_history_lookup("2026-04-29", source_dir=isolated_data_dir)
        _, _, tier = lookup["000999"][0]
        assert tier == "good"  # 2.0 ≤ 3.5 < 5.0
