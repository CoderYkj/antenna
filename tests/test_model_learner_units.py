"""tests/test_model_learner_units.py — P1 model_learner 纯函数单元测试。

覆盖:
  - load_config:文件缺失 / 字段缺失 / 正常加载
  - IdentityCalibrator:transform 各种输入形态
  - _classify_weight:(signal, hit_tier) → weight 全分支
  - compute_abs_threshold_delta:6 个分支(收严/维持/放宽/无样本/无 acc_30d)
  - brier_score:正常 / 空 / 长度不一致
  - fit_calibrator_for_bucket:样本不足 / 退化标签 / 正常拟合
  - load_abs_threshold:文件缺失 / 正常读取 / 损坏文件回退
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ── 共享 fixture ──────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path):
    """构造一份 yaml + 加载为 LearnerConfig。"""
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


# ── load_config ──────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_frozen_dataclass(self, cfg):
        assert cfg.sample_weights["buy_miss"] == 2.0
        with pytest.raises(Exception):
            # frozen=True → 修改抛 FrozenInstanceError 或 AttributeError
            cfg.sample_weights = {}  # type: ignore[misc]

    def test_missing_file_raises(self, tmp_path):
        from learning import model_learner
        with pytest.raises(ValueError, match="config not found"):
            model_learner.load_config(tmp_path / "nonexistent.yaml")

    def test_missing_section_raises(self, tmp_path):
        from learning import model_learner
        bad = tmp_path / "bad.yaml"
        bad.write_text("sample_weights: {default: 1.0}\ncalibration: {min_samples_per_bucket: 50}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="absolute_threshold"):
            model_learner.load_config(bad)

    def test_empty_yaml_raises(self, tmp_path):
        from learning import model_learner
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            model_learner.load_config(empty)


# ── IdentityCalibrator ───────────────────────────────────────

class TestIdentityCalibrator:
    def test_transform_passes_through_list(self):
        from learning.model_learner import IdentityCalibrator
        cal = IdentityCalibrator()
        assert cal.transform([0.1, 0.5, 0.9]) == [0.1, 0.5, 0.9]

    def test_predict_alias(self):
        from learning.model_learner import IdentityCalibrator
        cal = IdentityCalibrator()
        assert cal.predict([0.42]) == [0.42]

    def test_scalar_input_normalized_to_list(self):
        from learning.model_learner import IdentityCalibrator
        cal = IdentityCalibrator()
        assert cal.transform(0.7) == [0.7]

    def test_reason_field_persisted(self):
        from learning.model_learner import IdentityCalibrator
        cal = IdentityCalibrator(reason="missing")
        assert cal.reason == "missing"


# ── _classify_weight ─────────────────────────────────────────

class TestClassifyWeight:
    @pytest.mark.parametrize("tier,expected", [
        ("miss", 2.0),
        ("weak", 1.2),
        ("good", 1.0),
        ("great", 1.5),
    ])
    def test_buy_signal_uses_tier_specific_weight(self, cfg, tier, expected):
        from learning.model_learner import _classify_weight
        assert _classify_weight("买入", tier, cfg) == expected

    def test_buy_signal_unknown_tier_falls_to_default(self, cfg):
        from learning.model_learner import _classify_weight
        assert _classify_weight("买入", None, cfg) == cfg.sample_weights["default"]
        assert _classify_weight("买入", "unexpected", cfg) == cfg.sample_weights["default"]

    @pytest.mark.parametrize("signal", ["观望", "回避", None, ""])
    def test_non_buy_uses_non_buy_weight(self, cfg, signal):
        from learning.model_learner import _classify_weight
        assert _classify_weight(signal, "miss", cfg) == cfg.sample_weights["non_buy"]


# ── compute_abs_threshold_delta ──────────────────────────────

class TestComputeAbsThresholdDelta:
    def test_low_accuracy_tightens(self, cfg):
        from learning.model_learner import compute_abs_threshold_delta
        delta, reason = compute_abs_threshold_delta(0.30, 50, cfg)
        assert delta == +cfg.absolute_threshold["step"]
        assert "收严" in reason

    def test_mid_accuracy_holds(self, cfg):
        from learning.model_learner import compute_abs_threshold_delta
        delta, reason = compute_abs_threshold_delta(0.45, 50, cfg)
        assert delta == 0.0
        assert "维持" in reason

    def test_high_accuracy_loosens(self, cfg):
        from learning.model_learner import compute_abs_threshold_delta
        delta, reason = compute_abs_threshold_delta(0.60, 50, cfg)
        assert delta == -cfg.absolute_threshold["step"]
        assert "放宽" in reason

    def test_insufficient_buy_signals_holds(self, cfg):
        from learning.model_learner import compute_abs_threshold_delta
        delta, reason = compute_abs_threshold_delta(0.10, 5, cfg)
        assert delta == 0.0
        assert reason == "insufficient_samples"

    def test_acc_30d_none_holds(self, cfg):
        from learning.model_learner import compute_abs_threshold_delta
        delta, reason = compute_abs_threshold_delta(None, 50, cfg)
        assert delta == 0.0
        assert reason == "no_acc_30d"

    def test_boundary_35_percent_is_hold(self, cfg):
        """边界 35.0% 不算"低于",维持区间。"""
        from learning.model_learner import compute_abs_threshold_delta
        delta, _ = compute_abs_threshold_delta(0.35, 50, cfg)
        assert delta == 0.0

    def test_boundary_55_percent_is_hold(self, cfg):
        from learning.model_learner import compute_abs_threshold_delta
        delta, _ = compute_abs_threshold_delta(0.55, 50, cfg)
        assert delta == 0.0


# ── brier_score ──────────────────────────────────────────────

class TestBrierScore:
    def test_perfect_prediction(self):
        from learning.model_learner import brier_score
        assert brier_score([1.0, 0.0], [1, 0]) == 0.0

    def test_worst_prediction(self):
        from learning.model_learner import brier_score
        assert brier_score([1.0, 0.0], [0, 1]) == 1.0

    def test_mid_value(self):
        from learning.model_learner import brier_score
        # (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5; mean = 0.25
        assert brier_score([0.5, 0.5], [1, 0]) == 0.25

    def test_empty_returns_none(self):
        from learning.model_learner import brier_score
        assert brier_score([], []) is None

    def test_length_mismatch_returns_none(self):
        from learning.model_learner import brier_score
        assert brier_score([0.1, 0.2], [1]) is None


# ── fit_calibrator_for_bucket ────────────────────────────────

class TestFitCalibratorForBucket:
    def test_insufficient_samples_returns_identity(self, cfg):
        from learning.model_learner import fit_calibrator_for_bucket, IdentityCalibrator
        samples = [(0.5, 1)] * 10  # 远少于 min_samples_per_bucket=50
        cal = fit_calibrator_for_bucket(samples, cfg)
        assert isinstance(cal, IdentityCalibrator)
        assert cal.reason == "insufficient_data"

    def test_degenerate_labels_all_zero_returns_identity(self, cfg):
        from learning.model_learner import fit_calibrator_for_bucket, IdentityCalibrator
        samples = [(0.5, 0)] * 60
        cal = fit_calibrator_for_bucket(samples, cfg)
        assert isinstance(cal, IdentityCalibrator)
        assert cal.reason == "degenerate_labels"

    def test_degenerate_labels_all_one_returns_identity(self, cfg):
        from learning.model_learner import fit_calibrator_for_bucket, IdentityCalibrator
        samples = [(0.5, 1)] * 60
        cal = fit_calibrator_for_bucket(samples, cfg)
        assert isinstance(cal, IdentityCalibrator)
        assert cal.reason == "degenerate_labels"

    def test_normal_fit_returns_isotonic(self, cfg):
        from learning.model_learner import fit_calibrator_for_bucket, IdentityCalibrator
        # 60 个样本,概率与 hit 单调正相关
        import random
        random.seed(42)
        samples = []
        for _ in range(60):
            p = random.random()
            hit = 1 if p > 0.5 + random.gauss(0, 0.1) else 0
            samples.append((p, hit))
        cal = fit_calibrator_for_bucket(samples, cfg)
        assert not isinstance(cal, IdentityCalibrator)
        # 校准后概率 0~1 且单调不减
        out = list(cal.predict([0.1, 0.3, 0.5, 0.7, 0.9]))
        assert all(0.0 <= p <= 1.0 for p in out)
        assert out == sorted(out)


# ── load_abs_threshold ───────────────────────────────────────

class TestLoadAbsThreshold:
    def test_missing_file_returns_default(self, tmp_path, monkeypatch):
        from learning import model_learner
        monkeypatch.setattr(model_learner, "STATE_FILE", tmp_path / "no.json")
        assert model_learner.load_abs_threshold() == 0.30
        assert model_learner.load_abs_threshold(default=0.5) == 0.5

    def test_reads_value_from_state_file(self, tmp_path, monkeypatch):
        import json
        from learning import model_learner
        f = tmp_path / "ml.json"
        f.write_text(json.dumps({"abs_threshold": 0.52}), encoding="utf-8")
        monkeypatch.setattr(model_learner, "STATE_FILE", f)
        assert model_learner.load_abs_threshold() == 0.52

    def test_corrupt_file_falls_back_to_default(self, tmp_path, monkeypatch):
        from learning import model_learner
        f = tmp_path / "ml.json"
        f.write_text("not-json{", encoding="utf-8")
        monkeypatch.setattr(model_learner, "STATE_FILE", f)
        assert model_learner.load_abs_threshold(default=0.45) == 0.45

    def test_state_file_without_threshold_returns_default(self, tmp_path, monkeypatch):
        import json
        from learning import model_learner
        f = tmp_path / "ml.json"
        f.write_text(json.dumps({"calibrators": {}}), encoding="utf-8")
        monkeypatch.setattr(model_learner, "STATE_FILE", f)
        assert model_learner.load_abs_threshold(default=0.40) == 0.40


# ── load_calibrator(无 pkl 时兜底) ─────────────────────────

class TestLoadCalibrator:
    def test_missing_pkl_returns_identity(self, tmp_path, monkeypatch):
        from learning import model_learner
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        cal = model_learner.load_calibrator("bull")
        assert isinstance(cal, model_learner.IdentityCalibrator)
        assert cal.reason == "missing"

    def test_sha_mismatch_returns_identity(self, tmp_path, monkeypatch):
        import pickle
        from learning import model_learner
        from learning.model_learner import IdentityCalibrator
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        # 写入一个带 sha=A 的 pkl
        path = tmp_path / "calibrator_bull.pkl"
        with open(path, "wb") as f:
            pickle.dump({"calibrator": IdentityCalibrator(), "model_sha": "AAAA"}, f)
        cal = model_learner.load_calibrator("bull", current_model_sha="BBBB")
        assert isinstance(cal, IdentityCalibrator)
        assert cal.reason == "sha_mismatch"

    def test_sha_match_returns_calibrator(self, tmp_path, monkeypatch):
        import pickle
        from learning import model_learner
        from learning.model_learner import IdentityCalibrator
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        marker = IdentityCalibrator(reason="from_test")
        path = tmp_path / "calibrator_range.pkl"
        with open(path, "wb") as f:
            pickle.dump({"calibrator": marker, "model_sha": "XYZ"}, f)
        cal = model_learner.load_calibrator("range", current_model_sha="XYZ")
        assert cal.reason == "from_test"

    def test_empty_payload_returns_identity(self, tmp_path, monkeypatch):
        import pickle
        from learning import model_learner
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        path = tmp_path / "calibrator_bear.pkl"
        with open(path, "wb") as f:
            pickle.dump({"calibrator": None, "model_sha": "X"}, f)
        cal = model_learner.load_calibrator("bear")
        assert isinstance(cal, model_learner.IdentityCalibrator)
        assert cal.reason == "empty_payload"


# ── _load_latest_model_path ─────────────────────────────────

class TestLoadLatestModelPath:
    def test_returns_none_when_dir_missing(self, tmp_path, monkeypatch):
        from learning import model_learner
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path / "nonexistent")
        assert model_learner._load_latest_model_path() is None

    def test_returns_none_when_no_pkls(self, tmp_path, monkeypatch):
        from learning import model_learner
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        # 空目录
        assert model_learner._load_latest_model_path() is None

    def test_returns_latest_pkl_lexicographically(self, tmp_path, monkeypatch):
        from learning import model_learner
        monkeypatch.setattr(model_learner, "CALIBRATOR_DIR", tmp_path)
        for name in ("model_20260101.pkl", "model_20260301.pkl", "model_20260201.pkl"):
            (tmp_path / name).write_bytes(b"x")
        latest = model_learner._load_latest_model_path()
        assert latest is not None
        assert latest.name == "model_20260301.pkl"


# ── _compute_recent_buy_accuracy ─────────────────────────────

class TestComputeRecentBuyAccuracy:
    def _write_records(self, dir_, date, code, signal, hit_tier):
        import json
        with open(dir_ / f"pred_{date}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"code": code, "name": code, "signal": signal,
                                "rise_prob": 0.5, "scene": "scan"}) + "\n")
        with open(dir_ / f"outcome_{date}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"code": code, "actual_pct": 3.0, "hit_tier": hit_tier}) + "\n")

    def test_no_data_returns_none(self, tmp_path, monkeypatch):
        from learning import model_learner, tracker
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
        acc, n = model_learner._compute_recent_buy_accuracy("2026-04-29", lookback_days=30)
        assert acc is None
        assert n == 0

    def test_only_buy_signals_counted(self, tmp_path, monkeypatch):
        from learning import model_learner, tracker
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
        # 同一日:1 买入命中 + 1 买入 miss + 1 观望 → 精准率 0.5,n=2
        self._write_records(tmp_path, "2026-04-28", "000001", "买入", "good")
        self._write_records(tmp_path, "2026-04-28", "000002", "买入", "miss")
        self._write_records(tmp_path, "2026-04-28", "000003", "观望", "good")
        acc, n = model_learner._compute_recent_buy_accuracy("2026-04-29", lookback_days=30)
        assert n == 2
        assert acc == 0.5


# ── _persist_state(threshold 写入与 clamp) ────────────────

class TestPersistState:
    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path, monkeypatch):
        from learning import model_learner, feedback_io, tracker
        monkeypatch.setattr(model_learner, "STATE_FILE", tmp_path / "ml.json")
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir()
        monkeypatch.setattr(tracker, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(feedback_io, "HISTORY_DIR", tmp_path / "history")
        monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    def test_initial_threshold_used_when_no_prior_state(self, cfg, tmp_path):
        from learning import model_learner
        model_learner._persist_state("2026-04-29", {"bull": {"status": "ok"}}, cfg)
        import json
        with open(tmp_path / "ml.json", encoding="utf-8") as f:
            data = json.load(f)
        # 无历史样本 → delta=0,abs_threshold = initial=0.45
        assert data["abs_threshold"] == 0.45
        assert "threshold_history" in data

    def test_threshold_clamped_to_max(self, cfg, tmp_path, monkeypatch):
        """前值 0.59 + step 0.02 = 0.61 → clamp 到 max=0.60。"""
        import json
        from learning import model_learner
        # 预置一个高位 prior threshold
        (tmp_path / "ml.json").write_text(
            json.dumps({"abs_threshold": 0.59, "threshold_history": []}),
            encoding="utf-8",
        )
        # 让 _compute_recent_buy_accuracy 返回 acc=0.20 触发收严
        monkeypatch.setattr(
            model_learner, "_compute_recent_buy_accuracy",
            lambda d, lookback_days: (0.20, 50),
        )
        model_learner._persist_state("2026-04-29", {"bull": {}}, cfg)
        with open(tmp_path / "ml.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["abs_threshold"] == 0.60
        assert "clamped" in data["threshold_history"][-1]["change"]


# ── retrain_with_weights 输入校验 ─────────────────────────

class TestRetrainInputValidation:
    def test_empty_df_raises(self, cfg):
        from learning.model_learner import retrain_with_weights
        import pandas as pd
        with pytest.raises(ValueError, match="训练集为空"):
            retrain_with_weights(pd.DataFrame(), [], cfg)

    def test_missing_required_columns_raises(self, cfg):
        from learning.model_learner import retrain_with_weights
        import pandas as pd
        df = pd.DataFrame({"code": ["x"], "label": [1]})  # 缺 date
        with pytest.raises(ValueError, match="date / code / label"):
            retrain_with_weights(df, ["f1"], cfg)
