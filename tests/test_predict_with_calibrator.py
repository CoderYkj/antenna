"""tests/test_predict_with_calibrator.py — predict / assign_global_signals 接入校准器后的行为。

覆盖:
  - calibrator pkl 缺失 → rise_prob_cal == rise_prob_raw(回退恒等)
  - calibrator 存在 → transform 被应用,返回字段齐全
  - calibration 异常 → 回退不抛,推荐路径不阻断
  - predict 字段完整性(含 P1 新字段)
  - assign_global_signals 双门槛四分支
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


def _make_feature_df(n=80):
    np.random.seed(3)
    from features.technical import add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(5_000_000, 50_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


class _MockModel:
    def __init__(self, prob):
        self._prob = prob

    def predict(self, X):
        return np.full(len(X), self._prob)


class _ScaleCalibrator:
    """测试用:把 prob × 2 并 clip 到 [0, 1]。必须在模块顶层才能被 pickle。"""
    def transform(self, probs):
        return [min(1.0, float(p) * 2) for p in probs]
    def predict(self, probs):
        return self.transform(probs)


# ── predict 字段完整性 ──────────────────────────────────────

class TestPredictFields:
    def test_contains_p1_fields(self):
        from models.predictor import predict
        from features.technical import FEATURE_COLS
        df = _make_feature_df()
        with patch("models.predictor.load_latest_model", return_value=_MockModel(0.55)):
            r = predict(df, FEATURE_COLS)
        for key in ("rise_prob", "rise_prob_raw", "rise_prob_cal",
                    "abs_threshold_snapshot", "self_rank_pct",
                    "fall_prob", "confidence", "signal"):
            assert key in r, f"缺少字段: {key}"

    def test_fallback_identity_when_calibrator_missing(self, tmp_path, monkeypatch):
        """pkl 不存在时 rise_prob_cal 等于 rise_prob_raw。"""
        from learning import model_learner
        from models.predictor import predict
        from features.technical import FEATURE_COLS

        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path / "empty")

        df = _make_feature_df()
        with patch("models.predictor.load_latest_model", return_value=_MockModel(0.52)):
            r = predict(df, FEATURE_COLS)
        assert r["rise_prob_cal"] == r["rise_prob_raw"]
        assert r["rise_prob"] == r["rise_prob_cal"]


# ── calibration 异常不阻断 ────────────────────────────────

class TestCalibrationFailSafe:
    def test_calibrator_load_exception_returns_raw(self, monkeypatch):
        from models.predictor import _apply_calibration

        def boom(*a, **k):
            raise RuntimeError("simulated")

        # 让 load_calibrator 抛异常,验证 _apply_calibration 兜底
        from learning import model_learner
        monkeypatch.setattr(model_learner, "load_calibrator", boom)
        prob_cal, abs_thres = _apply_calibration(0.42)
        assert prob_cal == 0.42
        assert abs_thres == 0.30

    def test_market_state_exception_returns_raw(self, monkeypatch):
        from models.predictor import _apply_calibration
        from learning import market_state
        monkeypatch.setattr(market_state, "load_current_state",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        prob_cal, _ = _apply_calibration(0.88)
        assert prob_cal == 0.88


# ── calibration 真实拟合后 transform ────────────────────────

class TestCalibrationApplied:
    def test_isotonic_transform_shifts_probability(self, tmp_path, monkeypatch):
        """预置一个校准器 pkl,验证 transform 被调用。"""
        import pickle
        from learning import model_learner, market_state

        # 重定向路径
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        # 让 load_current_state 返回 bull
        monkeypatch.setattr(market_state, "load_current_state",
                            lambda: {"current": "bull"})

        # 装一个把概率 × 2 clip 到 [0,1] 的假校准器(顶层类可 pickle)
        with open(tmp_path / "calibrator_bull.pkl", "wb") as f:
            pickle.dump({"calibrator": _ScaleCalibrator(), "model_sha": "ANY"}, f)

        # abs_threshold 也落盘 0.5
        import json
        monkeypatch.setattr(model_learner, "STATE_FILE", tmp_path / "ml.json")
        (tmp_path / "ml.json").write_text(json.dumps({"abs_threshold": 0.50}), encoding="utf-8")

        from models.predictor import _apply_calibration
        prob_cal, abs_thres = _apply_calibration(0.30)
        assert prob_cal == 0.60  # 0.30 × 2 = 0.60
        assert abs_thres == 0.50


# ── assign_global_signals 双门槛四分支 ──────────────────────

class TestAssignGlobalSignalsDualGate:
    @pytest.fixture(autouse=True)
    def stub_abs_threshold(self, monkeypatch):
        """让 abs_threshold = 0.50,便于构造四个分支的样例。"""
        from learning import model_learner
        monkeypatch.setattr(model_learner, "load_abs_threshold", lambda: 0.50)

    def _mk(self, prob_raw, prob_cal=None):
        prob_cal = prob_cal if prob_cal is not None else prob_raw
        return {
            "code": f"code_{prob_raw}",
            "rise_prob":     prob_cal,
            "rise_prob_raw": prob_raw,
            "rise_prob_cal": prob_cal,
        }

    def test_branch_1_buy_when_rank_and_prob_cal_both_pass(self):
        """rank 前 10% 且 prob_cal >= 0.50 → 买入。"""
        from models.predictor import assign_global_signals
        # 构造 10 条,第 1 条 rank=10% 且 prob_cal=0.8 → 买入
        results = [
            self._mk(0.9, 0.8),       # rank 1/10 = 10%, prob_cal 0.8 → 买入
            self._mk(0.5, 0.45),
            self._mk(0.4, 0.4),
            self._mk(0.35, 0.35),
            self._mk(0.3, 0.3),
            self._mk(0.25, 0.25),
            self._mk(0.2, 0.2),
            self._mk(0.15, 0.15),
            self._mk(0.1, 0.1),
            self._mk(0.05, 0.05),
        ]
        assign_global_signals(results, buy_top_pct=0.10)
        top = next(r for r in results if r["rise_prob_raw"] == 0.9)
        assert top["signal"] == "买入"
        assert top["confidence"] == "高"

    def test_branch_2_downgrade_to_watch_when_rank_passes_but_prob_cal_below(self):
        """rank 前 10% 但 prob_cal < 0.50 → 观望(降级)。"""
        from models.predictor import assign_global_signals
        # rank 最高但校准后概率 0.3 < 0.5
        results = [
            self._mk(0.9, 0.3),   # rank 1/10, prob_cal=0.3 → 应为 观望
            *(self._mk(p, p - 0.05) for p in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]),
        ]
        assign_global_signals(results, buy_top_pct=0.10)
        top = next(r for r in results if r["rise_prob_raw"] == 0.9)
        assert top["signal"] == "观望"

    def test_branch_3_watch_zone_is_hold(self):
        """rank 在 watch 区(10%~30%) → 观望,不受 abs_threshold 约束。"""
        from models.predictor import assign_global_signals
        # 10 条,第 2 条 rank=20% 应落在 watch 区
        results = [self._mk(0.9 - i * 0.05, 0.9 - i * 0.05) for i in range(10)]
        assign_global_signals(results, buy_top_pct=0.10)
        second = next(r for r in results if r["rise_prob_raw"] == 0.85)
        assert second["signal"] == "观望"

    def test_branch_4_avoid_when_rank_beyond_watch(self):
        """rank 超过 watch_top_pct(30%) → 回避。"""
        from models.predictor import assign_global_signals
        results = [self._mk(0.9 - i * 0.05, 0.9 - i * 0.05) for i in range(10)]
        assign_global_signals(results, buy_top_pct=0.10)
        tail_ = next(r for r in results if r["rise_prob_raw"] == 0.45)  # rank 10/10
        assert tail_["signal"] == "回避"

    def test_empty_results_noop(self):
        from models.predictor import assign_global_signals
        assert assign_global_signals([], buy_top_pct=0.10) == []

    def test_global_rank_fields_populated(self):
        from models.predictor import assign_global_signals
        results = [self._mk(0.5), self._mk(0.7), self._mk(0.3)]
        assign_global_signals(results, buy_top_pct=0.50)
        # 0.7 应排第 1
        top = next(r for r in results if r["rise_prob_raw"] == 0.7)
        assert top["global_rank"] == 1


# ── abs_threshold 加载失败时的降级行为 ─────────────────────

class TestAssignSignalsAbsThresholdFailSafe:
    def test_abs_threshold_exception_falls_to_zero(self, monkeypatch):
        """load_abs_threshold 抛异常 → abs_threshold=0,相当于单门槛(分位)。"""
        from learning import model_learner
        from models.predictor import assign_global_signals

        def boom():
            raise RuntimeError("xx")
        monkeypatch.setattr(model_learner, "load_abs_threshold", boom)

        results = [
            {"code": "A", "rise_prob": 0.3, "rise_prob_raw": 0.3, "rise_prob_cal": 0.01},
            {"code": "B", "rise_prob": 0.1, "rise_prob_raw": 0.1, "rise_prob_cal": 0.005},
        ]
        assign_global_signals(results, buy_top_pct=0.50)
        # 第一名 rank=50% <= buy_top_pct,且 abs_threshold=0 自动通过 → 买入
        top = next(r for r in results if r["code"] == "A")
        assert top["signal"] == "买入"
