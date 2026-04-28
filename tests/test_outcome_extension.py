import pytest
from learning.outcome_metrics import compute_hit_tier, compute_5d_metrics


class TestComputeHitTier:
    @pytest.mark.parametrize("pct,expected", [
        (-5.0, "miss"),
        (0.0, "miss"),
        (0.9, "miss"),
        (1.0, "weak"),
        (1.5, "weak"),
        (1.99, "weak"),
        (2.0, "good"),
        (3.5, "good"),
        (4.99, "good"),
        (5.0, "great"),
        (10.0, "great"),
    ])
    def test_hit_tier_boundaries(self, pct, expected):
        assert compute_hit_tier(pct) == expected

    def test_hit_tier_none_returns_none(self):
        assert compute_hit_tier(None) is None


class TestCompute5dMetrics:
    def test_metrics_returns_cumulative_and_drawdown(self):
        """给定 5 日数据:
        Day0 close=100, Day1 close=103, Day2 close=98, Day3 close=105, Day4 close=107
        累计涨幅 = (107/100 - 1)*100 = 7.0
        最大回撤 = (98-103)/103 = -4.85%(从 103 跌到 98)
        """
        closes = [100.0, 103.0, 98.0, 105.0, 107.0]
        metrics = compute_5d_metrics(closes)
        assert metrics["hit_5d"] == pytest.approx(7.0, abs=0.01)
        assert metrics["max_drawdown_5d"] == pytest.approx(-4.85, abs=0.05)

    def test_metrics_all_up(self):
        closes = [100, 101, 102, 103, 104]
        metrics = compute_5d_metrics(closes)
        assert metrics["max_drawdown_5d"] == pytest.approx(0.0, abs=0.01)

    def test_metrics_insufficient_data_returns_none(self):
        assert compute_5d_metrics([100, 101]) is None
        assert compute_5d_metrics([]) is None

    def test_metrics_single_value_returns_none(self):
        assert compute_5d_metrics([100]) is None

    def test_metrics_nan_in_closes_returns_none(self):
        """NaN 或 None 输入应返回 None,避免下游拿到污染数据。"""
        import math
        assert compute_5d_metrics([100.0, float("nan"), 102.0, 103.0, 104.0]) is None
        assert compute_5d_metrics([100.0, None, 102.0, 103.0, 104.0]) is None
        assert compute_5d_metrics([100.0, math.inf, 102.0, 103.0, 104.0]) is None

    def test_metrics_clamps_to_first_5_when_given_more(self):
        """长度 >5 时按前 5 元素计算,使返回稳定为 5 日指标。"""
        # 7 elements: first 5 are [100, 103, 98, 105, 107] → hit_5d=7.0
        closes = [100.0, 103.0, 98.0, 105.0, 107.0, 80.0, 80.0]
        metrics = compute_5d_metrics(closes)
        assert metrics["hit_5d"] == pytest.approx(7.0, abs=0.01)
        assert metrics["max_drawdown_5d"] == pytest.approx(-4.85, abs=0.05)
