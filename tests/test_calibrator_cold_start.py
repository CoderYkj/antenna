"""tests/test_calibrator_cold_start.py — 冷启动样本不足的兜底策略。

spec §5.1 + §7:
  - 单桶 < min → 按 cold_start_fallback:
      - "global"    用全部桶汇总样本拟合
      - "identity"  直接 IdentityCalibrator(reason='insufficient_data')
  - 全局仍 < min → IdentityCalibrator(reason='insufficient_data')
  - 拟合异常 → IdentityCalibrator(reason='degenerate_labels' 等)
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write_cfg(tmp_path: Path, fallback: str = "global", min_samples: int = 50) -> Path:
    yaml_path = tmp_path / "model_learner.yaml"
    yaml_path.write_text(
        f"""
sample_weights:
  buy_miss:   2.0
  buy_weak:   1.2
  buy_good:   1.0
  buy_great:  1.5
  non_buy:    0.8
  default:    1.0
calibration:
  lookback_days:          90
  min_samples_per_bucket: {min_samples}
  cold_start_fallback:    {fallback}
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
    return yaml_path


def _make_samples(n: int, hit_rate: float = 0.5, seed: int = 0) -> list[tuple[float, int]]:
    """生成 n 条 (prob, hit) 样本,概率与命中正相关。"""
    import random
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        p = rng.random()
        # 与 hit 单调相关:p 大则更可能 hit=1
        threshold = 1.0 - hit_rate
        hit = 1 if p > threshold + rng.gauss(0, 0.05) else 0
        samples.append((p, hit))
    # 保底有 0 也有 1,避免 degenerate
    if all(s[1] == 0 for s in samples):
        samples[0] = (samples[0][0], 1)
    if all(s[1] == 1 for s in samples):
        samples[0] = (samples[0][0], 0)
    return samples


class TestSingleBucketColdStart:
    """对 fit_calibrator_for_bucket 直接测:不经过 fit_calibrators 的 fallback 逻辑。"""

    def test_below_min_returns_identity(self, tmp_path):
        from learning.model_learner import (
            fit_calibrator_for_bucket, IdentityCalibrator, load_config,
        )
        cfg = load_config(_write_cfg(tmp_path, min_samples=50))
        cal = fit_calibrator_for_bucket(_make_samples(30), cfg)
        assert isinstance(cal, IdentityCalibrator)
        assert cal.reason == "insufficient_data"

    def test_at_or_above_min_fits(self, tmp_path):
        from learning.model_learner import (
            fit_calibrator_for_bucket, IdentityCalibrator, load_config,
        )
        cfg = load_config(_write_cfg(tmp_path, min_samples=50))
        cal = fit_calibrator_for_bucket(_make_samples(60, seed=1), cfg)
        assert not isinstance(cal, IdentityCalibrator)


class TestGlobalFallback:
    """fit_calibrators 主入口的 fallback 行为(单桶不足→pooled→仍不足→identity)。"""

    @pytest.fixture(autouse=True)
    def isolate_io(self, tmp_path, monkeypatch):
        """隔离所有 IO 路径到 tmp_path,避免触碰生产 learning/ 与 models/ 目录。"""
        from learning import model_learner, feedback_io
        # 校准器/状态文件路径
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path / "saved")
        monkeypatch.setattr(model_learner, "STATE_FILE", tmp_path / "model_learner.json")
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path / "data")
        # feedback_io 的 history snapshot 路径
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    def test_single_bucket_short_pools_global_when_total_sufficient(self, tmp_path, monkeypatch):
        """bull 仅 30 条,但 bull+bear+range 合计 > min → fallback="global" 应使用 pooled。"""
        from learning import model_learner

        cfg = model_learner.load_config(_write_cfg(tmp_path, fallback="global", min_samples=50))

        # mock 内部数据收集函数,直接给定分桶结果
        # 设计:bull 30(不足触发 fallback);bear/range 各 60(达标 min=50,不触发)
        def fake_collect(date_str, cfg, model, feature_cols):
            return {
                "bull":  _make_samples(30, seed=1),
                "bear":  _make_samples(60, seed=2),
                "range": _make_samples(60, seed=3),
            }

        monkeypatch.setattr(model_learner, "_collect_calibration_samples", fake_collect)
        monkeypatch.setattr(model_learner, "_load_latest_model_path", lambda: tmp_path / "fake_model.pkl")
        # 制造一个假模型 pkl 用于 sha 计算
        (tmp_path / "fake_model.pkl").write_bytes(b"fake_model_bytes")
        # 假的 model 对象
        class FakeModel:
            def feature_name(self): return ["f1"]
        monkeypatch.setattr(model_learner, "_feature_cols_from_model", lambda m: ["f1"])
        monkeypatch.setattr("models.trainer.load_latest_model", lambda d: FakeModel())

        result = model_learner.fit_calibrators("2026-04-29", cfg)

        # bull 应触发 fallback,bear/range 各自达标
        assert result["bull"]["used_fallback_global"] is True
        assert result["bear"]["used_fallback_global"] is False
        assert result["range"]["used_fallback_global"] is False
        # bull 因 fallback 拿到 110 条 pooled,应能拟合成 ok(非 IdentityCalibrator)
        assert result["bull"]["status"] == "ok"

    def test_global_pool_still_insufficient_returns_identity(self, tmp_path, monkeypatch):
        from learning import model_learner

        cfg = model_learner.load_config(_write_cfg(tmp_path, fallback="global", min_samples=50))

        def fake_collect(date_str, cfg, model, feature_cols):
            return {
                "bull":  _make_samples(5, seed=1),
                "bear":  _make_samples(5, seed=2),
                "range": _make_samples(5, seed=3),
            }
        monkeypatch.setattr(model_learner, "_collect_calibration_samples", fake_collect)
        monkeypatch.setattr(model_learner, "_load_latest_model_path", lambda: tmp_path / "fake.pkl")
        (tmp_path / "fake.pkl").write_bytes(b"x")
        class FakeModel:
            def feature_name(self): return ["f1"]
        monkeypatch.setattr(model_learner, "_feature_cols_from_model", lambda m: ["f1"])
        monkeypatch.setattr("models.trainer.load_latest_model", lambda d: FakeModel())

        result = model_learner.fit_calibrators("2026-04-29", cfg)

        # 三个桶都应是 insufficient_data
        for state in ("bull", "bear", "range"):
            assert result[state]["status"] == "insufficient_data"

    def test_no_model_pkl_emits_no_model_status(self, tmp_path, monkeypatch):
        """models/saved/ 下没有 model_*.pkl 时,所有桶 status='no_model',不抛异常。"""
        from learning import model_learner

        cfg = model_learner.load_config(_write_cfg(tmp_path, min_samples=50))
        monkeypatch.setattr(model_learner, "_load_latest_model_path", lambda: None)

        result = model_learner.fit_calibrators("2026-04-29", cfg)
        for state in ("bull", "bear", "range"):
            assert result[state]["status"] == "no_model"
