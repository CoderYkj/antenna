"""tests/test_tactic_learner_units.py — P2 tactic_learner 纯函数单元测试。

覆盖:
  - load_config:文件缺失 / 字段缺失 / 正常加载
  - get_defaults_for_state:bear override 生效 / 非 bear 不影响
  - adjust_thresholds:
      * tighten_up 收严方向(+step)
      * tighten_down 收严方向(-step)
      * 放宽对称方向
      * 维持区间 / 样本不足 / 无 acc_90d
      * clamp 上下界
      * 布尔字段不受影响
      * drawdown 用专属步长
  - compute_resonance_weights:归一化 / 全 0 兜底 / 上下界夹紧
  - load_params:文件缺失 → defaults / 损坏 → defaults / bear 取 override
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ── 共享 fixture ──────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path):
    from learning import tactic_learner
    yaml_path = tmp_path / "tactic_learner.yaml"
    yaml_path.write_text(
        """
defaults:
  value:
    roe_min:         {value:  8.0,  direction: tighten_up}
    debt_ratio_max:  {value: 50.0,  direction: tighten_down}
    total_score_min: {value:  2.0,  direction: tighten_up}
  growth:
    rev_growth_min:    {value: 15.0, direction: tighten_up}
    profit_growth_min: {value: 15.0, direction: tighten_up}
    roe_min:           {value: 12.0, direction: tighten_up}
  leader:
    roe_min:          {value: 15.0, direction: tighten_up}
    gross_margin_min: {value: 30.0, direction: tighten_up}
    above_ma60_required: true
  contra:
    drawdown_max:   {value: -0.15, direction: tighten_down}
    roe_min:        {value:  3.0,  direction: tighten_up}
    debt_ratio_max: {value: 65.0,  direction: tighten_down}

defaults_bear_override:
  value:
    roe_min:        12.0
    debt_ratio_max: 40.0
  contra:
    drawdown_max:   -0.20

bounds:
  value:
    roe_min:        [5.0, 15.0]
    debt_ratio_max: [40.0, 60.0]
  growth:
    rev_growth_min: [10.0, 25.0]
    profit_growth_min: [10.0, 25.0]
    roe_min: [8.0, 18.0]
  leader:
    roe_min: [12.0, 20.0]
    gross_margin_min: [25.0, 40.0]
  contra:
    drawdown_max: [-0.25, -0.10]
    roe_min: [2.0, 8.0]
    debt_ratio_max: [55.0, 75.0]

step:
  default: 1.0
  drawdown: 0.01

evaluation:
  lookback_days: 90
  min_samples:   30
  acc_low:       0.30
  acc_high:      0.60

weights:
  enable_resonance_rank_boost: true
  rank_boost_per_tactic: 0.02
  rank_boost_max:        0.06
  weight_min:            0.05
  weight_max:            0.40
""",
        encoding="utf-8",
    )
    return tactic_learner.load_config(yaml_path)


# ── load_config ──────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_frozen_dataclass(self, cfg):
        assert cfg.defaults["value"]["roe_min"]["value"] == 8.0
        with pytest.raises(Exception):
            cfg.defaults = {}  # type: ignore[misc]

    def test_missing_file_raises(self, tmp_path):
        from learning import tactic_learner
        with pytest.raises(ValueError, match="config not found"):
            tactic_learner.load_config(tmp_path / "nope.yaml")

    def test_missing_section_raises(self, tmp_path):
        from learning import tactic_learner
        bad = tmp_path / "bad.yaml"
        bad.write_text("defaults: {}\nbounds: {}\nstep: {}\nevaluation: {}\n",
                       encoding="utf-8")
        with pytest.raises(ValueError, match="weights"):
            tactic_learner.load_config(bad)


# ── get_defaults_for_state(bear override) ─────────────────

class TestDefaultsForState:
    def test_range_uses_plain_defaults(self, cfg):
        from learning.tactic_learner import get_defaults_for_state
        d = get_defaults_for_state("range", cfg)
        assert d["value"]["roe_min"] == 8.0
        assert d["contra"]["drawdown_max"] == -0.15

    def test_bull_uses_plain_defaults(self, cfg):
        from learning.tactic_learner import get_defaults_for_state
        d = get_defaults_for_state("bull", cfg)
        assert d["value"]["roe_min"] == 8.0

    def test_bear_applies_override(self, cfg):
        from learning.tactic_learner import get_defaults_for_state
        d = get_defaults_for_state("bear", cfg)
        # override 生效
        assert d["value"]["roe_min"] == 12.0
        assert d["value"]["debt_ratio_max"] == 40.0
        assert d["contra"]["drawdown_max"] == -0.20
        # 未 override 的保持原值
        assert d["value"]["total_score_min"] == 2.0
        assert d["growth"]["rev_growth_min"] == 15.0

    def test_boolean_field_preserved(self, cfg):
        """leader.above_ma60_required 是布尔,应保留不报错。"""
        from learning.tactic_learner import get_defaults_for_state
        d = get_defaults_for_state("range", cfg)
        assert d["leader"]["above_ma60_required"] is True


# ── adjust_thresholds 核心 ───────────────────────────────────

class TestAdjustThresholds:
    """D5 验证:tighten_up vs tighten_down 符号方向。"""

    def test_tighten_up_increases_on_low_acc(self, cfg):
        """roe_min(tighten_up):acc 低 → +step 数值变高 = 更严。"""
        from learning.tactic_learner import adjust_thresholds
        new, reason = adjust_thresholds(
            current={"roe_min": 8.0}, acc_90d=0.25, samples=50, cfg=cfg, tactic="value",
        )
        assert new["roe_min"] == 9.0
        assert "收严" in reason

    def test_tighten_down_decreases_on_low_acc(self, cfg):
        """debt_ratio_max(tighten_down):acc 低 → -step 数值变低 = 更严。

        旧版 "全部 +step" 的 bug 验证:用错方向会把 debt 50 调到 51(允许更高负债=放宽)。
        """
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"debt_ratio_max": 50.0}, acc_90d=0.20, samples=50, cfg=cfg, tactic="value",
        )
        assert new["debt_ratio_max"] == 49.0  # 不应该是 51!

    def test_tighten_up_decreases_on_high_acc(self, cfg):
        """高精准率放宽方向相反:roe_min 应该 -step。"""
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"roe_min": 10.0}, acc_90d=0.70, samples=50, cfg=cfg, tactic="value",
        )
        assert new["roe_min"] == 9.0  # 数值降低 = 放宽

    def test_tighten_down_increases_on_high_acc(self, cfg):
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"debt_ratio_max": 50.0}, acc_90d=0.70, samples=50, cfg=cfg, tactic="value",
        )
        assert new["debt_ratio_max"] == 51.0  # 数值升高 = 放宽

    def test_holds_in_target_range(self, cfg):
        from learning.tactic_learner import adjust_thresholds
        new, reason = adjust_thresholds(
            current={"roe_min": 8.0}, acc_90d=0.45, samples=50, cfg=cfg, tactic="value",
        )
        assert new == {"roe_min": 8.0}
        assert "维持" in reason

    def test_insufficient_samples_holds(self, cfg):
        from learning.tactic_learner import adjust_thresholds
        new, reason = adjust_thresholds(
            current={"roe_min": 8.0}, acc_90d=0.10, samples=5, cfg=cfg, tactic="value",
        )
        assert new == {"roe_min": 8.0}
        assert reason == "insufficient_samples"

    def test_acc_none_holds(self, cfg):
        from learning.tactic_learner import adjust_thresholds
        new, reason = adjust_thresholds(
            current={"roe_min": 8.0}, acc_90d=None, samples=100, cfg=cfg, tactic="value",
        )
        assert new == {"roe_min": 8.0}
        assert reason == "no_acc_90d"

    def test_clamps_to_lower_bound(self, cfg):
        """value.roe_min lower=5.0,试图越界后被夹紧。"""
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"roe_min": 5.0}, acc_90d=0.80, samples=50, cfg=cfg, tactic="value",
        )
        # 应想 -1 = 4.0,但夹到 5.0
        assert new["roe_min"] == 5.0

    def test_clamps_to_upper_bound(self, cfg):
        """value.roe_min upper=15.0。"""
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"roe_min": 15.0}, acc_90d=0.10, samples=50, cfg=cfg, tactic="value",
        )
        # 想 +1 = 16,夹到 15
        assert new["roe_min"] == 15.0

    def test_drawdown_uses_specific_step(self, cfg):
        """drawdown_max 用 0.01 步长,不是 1.0。"""
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"drawdown_max": -0.15}, acc_90d=0.20, samples=50,
            cfg=cfg, tactic="contra",
        )
        # tighten_down 收严 → -step → -0.15 - 0.01 = -0.16
        assert new["drawdown_max"] == -0.16

    def test_boolean_field_preserved_in_adjust(self, cfg):
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"roe_min": 15.0, "above_ma60_required": True},
            acc_90d=0.20, samples=50, cfg=cfg, tactic="leader",
        )
        assert new["above_ma60_required"] is True  # 布尔保持原样
        assert new["roe_min"] == 16.0  # 数值正常调整

    def test_multiple_thresholds_adjusted_together(self, cfg):
        """value 战法 3 个阈值,精准率低时应一起按 direction 收严。"""
        from learning.tactic_learner import adjust_thresholds
        new, _ = adjust_thresholds(
            current={"roe_min": 8.0, "debt_ratio_max": 50.0, "total_score_min": 2.0},
            acc_90d=0.20, samples=50, cfg=cfg, tactic="value",
        )
        assert new["roe_min"] == 9.0           # tighten_up + 收严 → +
        assert new["debt_ratio_max"] == 49.0   # tighten_down + 收严 → -
        assert new["total_score_min"] == 3.0   # tighten_up + 收严 → +


# ── compute_resonance_weights ────────────────────────────────

class TestComputeResonanceWeights:
    def test_normalizes_to_sum_one(self, cfg):
        """归一化后大致 sum = 1(夹紧后可能不严格,但范围合理)。"""
        from learning.tactic_learner import compute_resonance_weights
        accs = {"value": 0.4, "growth": 0.3, "leader": 0.2, "contra": 0.1}
        w = compute_resonance_weights(accs, cfg)
        # 全部在 [0.05, 0.40] 内
        assert all(0.05 <= v <= 0.40 for v in w.values())
        # value 应该最大,contra 最小
        assert w["value"] >= w["growth"]
        assert w["growth"] >= w["leader"]
        assert w["leader"] >= w["contra"]

    def test_all_zero_fallback_to_weight_min(self, cfg):
        from learning.tactic_learner import compute_resonance_weights
        accs = {"value": 0.0, "growth": 0.0, "leader": 0.0, "contra": 0.0}
        w = compute_resonance_weights(accs, cfg)
        assert all(v == 0.05 for v in w.values())

    def test_none_values_treated_as_zero(self, cfg):
        from learning.tactic_learner import compute_resonance_weights
        accs = {"value": None, "growth": 0.6, "leader": None, "contra": None}
        w = compute_resonance_weights(accs, cfg)
        # growth 独占归一化(其他归 0),被夹到 weight_max
        assert w["growth"] == 0.40
        # None 战法兜底 weight_min
        assert w["value"] == 0.05

    def test_clamps_to_max(self, cfg):
        """单战法独大,归一化后超过 weight_max=0.40,被夹紧。"""
        from learning.tactic_learner import compute_resonance_weights
        accs = {"value": 0.9, "growth": 0.05, "leader": 0.05, "contra": 0.0}
        w = compute_resonance_weights(accs, cfg)
        assert w["value"] == 0.40  # 实际 normalized = 0.9 但被夹紧


# ── load_params 兜底 ─────────────────────────────────────────

class TestLoadParams:
    def test_missing_file_returns_defaults(self, tmp_path, monkeypatch, cfg):
        from learning import tactic_learner
        monkeypatch.setattr(tactic_learner, "STATE_FILE", tmp_path / "nope.json")
        params = tactic_learner.load_params("range", cfg)
        assert params["value"]["roe_min"] == 8.0

    def test_bear_missing_file_uses_override(self, tmp_path, monkeypatch, cfg):
        from learning import tactic_learner
        monkeypatch.setattr(tactic_learner, "STATE_FILE", tmp_path / "nope.json")
        params = tactic_learner.load_params("bear", cfg)
        assert params["value"]["roe_min"] == 12.0  # bear override
        assert params["contra"]["drawdown_max"] == -0.20

    def test_corrupt_file_returns_defaults(self, tmp_path, monkeypatch, cfg):
        from learning import tactic_learner
        bad = tmp_path / "bad.json"
        bad.write_text("not-json{", encoding="utf-8")
        monkeypatch.setattr(tactic_learner, "STATE_FILE", bad)
        params = tactic_learner.load_params("range", cfg)
        assert params["value"]["roe_min"] == 8.0  # 回退 defaults

    def test_valid_file_returns_persisted(self, tmp_path, monkeypatch, cfg):
        import json
        from learning import tactic_learner
        f = tmp_path / "params.json"
        f.write_text(json.dumps({
            "params": {
                "range": {
                    "value": {"roe_min": 9.0, "debt_ratio_max": 48.0,
                              "total_score_min": 2.0, "weight": 0.3,
                              "acc_90d": 0.4, "samples": 100, "status": "ok"},
                    "growth": {"rev_growth_min": 16.0, "profit_growth_min": 15.0,
                               "roe_min": 12.0, "weight": 0.25},
                    "leader": {"roe_min": 15.0, "gross_margin_min": 30.0,
                               "above_ma60_required": True, "weight": 0.25},
                    "contra": {"drawdown_max": -0.16, "roe_min": 3.0,
                               "debt_ratio_max": 65.0, "weight": 0.2},
                }
            }
        }), encoding="utf-8")
        monkeypatch.setattr(tactic_learner, "STATE_FILE", f)
        params = tactic_learner.load_params("range", cfg)
        assert params["value"]["roe_min"] == 9.0
        assert params["contra"]["drawdown_max"] == -0.16
        # meta 字段不应出现在阈值结果里
        assert "weight" not in params["value"]
        assert "acc_90d" not in params["value"]
