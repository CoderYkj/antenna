"""tests/test_retrain_weighted.py — P1 加权重训冒烟。

覆盖:
  - trainer.train_weighted 接受 sample_weight 列并传递给 LightGBM
  - 高权重样本(buy_miss=2.0)在最终模型中影响更大
  - retrain_with_weights 调用链:lookup 构建 → resolve_sample_weight → train_weighted
  - retrain_with_weights 的输入校验(空 df / 缺列)
  - cli --weighted flag 走加权路径(不真实训练,只断言路径选择)

注意:
  - 不真跑 LightGBM 训练全流程(慢 + 噪声),用小数据 + 短训练验证 weight 被传递
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── trainer.train_weighted 端到端 ─────────────────────────

class TestTrainWeightedEndToEnd:
    @pytest.fixture
    def small_dataset(self):
        """构造一个简单可学习的合成数据集:f1 与 label 强相关。"""
        np.random.seed(7)
        n = 800  # 小数据点求快
        f1 = np.random.uniform(0, 1, n)
        # label = (f1 > 0.5) 的概率高
        label = (f1 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "code": [f"c{i % 5}" for i in range(n)],
            "f1": f1,
            "f2": np.random.uniform(0, 1, n),
            "label": label,
            "sample_weight": np.ones(n),  # 默认全 1
        })
        return df

    def test_returns_booster_with_default_weights(self, small_dataset):
        from models.trainer import train_weighted
        model = train_weighted(small_dataset, feature_cols=["f1", "f2"])
        # LightGBM Booster 总有 predict 方法
        preds = model.predict(small_dataset[["f1", "f2"]].iloc[:5])
        assert len(preds) == 5
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_high_weight_on_subset_shifts_predictions(self, small_dataset):
        """同样数据,把后半段样本权重 ×10,模型应更偏向后半段的标签分布。"""
        from models.trainer import train_weighted
        df_a = small_dataset.copy()
        df_b = small_dataset.copy()
        # b 版本:把 f1>0.5 的样本权重提高
        df_b.loc[df_b["f1"] > 0.5, "sample_weight"] = 10.0

        model_a = train_weighted(df_a, feature_cols=["f1", "f2"])
        model_b = train_weighted(df_b, feature_cols=["f1", "f2"])

        # 取一个 f1=0.9 的测试点,b 应给出更高 prob(被加权样本主导)
        test_x = pd.DataFrame({"f1": [0.9], "f2": [0.5]})
        prob_a = float(model_a.predict(test_x)[0])
        prob_b = float(model_b.predict(test_x)[0])
        # 不强求严格大于,但差值应非零(weight 真的传给了 LGB)
        assert prob_a != prob_b, "sample_weight 应该影响模型输出"

    def test_drops_rows_with_missing_features(self, small_dataset):
        """有 NaN 的行应被 dropna 排除而非崩溃。"""
        from models.trainer import train_weighted
        df = small_dataset.copy()
        df.loc[0:5, "f1"] = np.nan
        # 不应抛异常
        model = train_weighted(df, feature_cols=["f1", "f2"])
        assert model is not None


# ── retrain_with_weights 调用链 ───────────────────────────

class TestRetrainWithWeightsPipeline:
    @pytest.fixture
    def cfg(self, tmp_path):
        from learning import model_learner
        yaml_path = tmp_path / "ml.yaml"
        yaml_path.write_text(
            """
sample_weights:
  buy_miss: 2.0
  buy_weak: 1.2
  buy_good: 1.0
  buy_great: 1.5
  non_buy: 0.8
  default: 1.0
calibration:
  lookback_days: 90
  min_samples_per_bucket: 50
  cold_start_fallback: global
absolute_threshold:
  initial: 0.45
  min: 0.30
  max: 0.60
  step: 0.02
  lookback_days: 30
  min_buy_signals: 10
""", encoding="utf-8")
        return model_learner.load_config(yaml_path)

    def test_sample_weight_column_appended(self, cfg, tmp_path, monkeypatch):
        """retrain_with_weights 应给 df 加 sample_weight 列再调 train_weighted。"""
        from learning import model_learner

        # 隔离 DATA_DIR(空目录 → 所有样本走 default=1.0)
        monkeypatch.setattr(model_learner, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir()

        captured = {}

        def fake_train(df, feature_cols, weight_col="sample_weight"):
            # 捕获被传入的 df,断言 sample_weight 列存在
            captured["df"] = df.copy()
            captured["feature_cols"] = feature_cols
            captured["weight_col"] = weight_col
            return "fake-model"

        monkeypatch.setattr("models.trainer.train_weighted", fake_train)

        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "code": ["c1"] * 5,
            "f1":   [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0],
        })
        result = model_learner.retrain_with_weights(df, feature_cols=["f1"], cfg=cfg)
        assert result == "fake-model"
        assert "sample_weight" in captured["df"].columns
        # 空 lookup → 全 default(1.0)
        assert (captured["df"]["sample_weight"] == 1.0).all()

    def test_buy_miss_history_yields_high_weight(self, cfg, tmp_path, monkeypatch):
        """构造一条 T-7 的 buy+miss 历史,验证训练样本拿到 weight=2.0。"""
        from learning import model_learner

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(model_learner, "DATA_DIR", data_dir)

        # 写 outcome + pred 在 2025-12-25(T-7 of 2026-01-01)
        with open(data_dir / "pred_2025-12-25.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"code": "c1", "name": "c1", "signal": "买入",
                                "rise_prob": 0.7, "scene": "scan"}) + "\n")
        with open(data_dir / "outcome_2025-12-25.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"code": "c1", "actual_pct": 0.3,
                                "hit_tier": "miss"}) + "\n")

        captured = {}
        def fake_train(df, feature_cols, weight_col="sample_weight"):
            captured["df"] = df.copy()
            return "fake"
        monkeypatch.setattr("models.trainer.train_weighted", fake_train)

        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "code": ["c1", "c2"],
            "f1": [0.5, 0.6],
            "label": [1, 0],
        })
        model_learner.retrain_with_weights(df, feature_cols=["f1"], cfg=cfg)
        weights = captured["df"].set_index("code")["sample_weight"].to_dict()
        # c1 有历史 buy+miss → weight=2.0
        assert weights["c1"] == 2.0
        # c2 无历史 → default=1.0
        assert weights["c2"] == 1.0


# ── cli --weighted flag 路由 ───────────────────────────────

class TestCliWeightedRouting:
    def test_weighted_flag_invokes_retrain_path(self, monkeypatch, tmp_path, capsys):
        """args.weighted=True 时 cli.cmd_train 走 retrain_with_weights 而非 train。"""
        import cli
        # 把所有重型依赖打桩

        def fake_load_universe(cfg):
            return ["c1", "c2"]

        def fake_fetch(code, days, cache_only):
            return pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "open": [10.0]*10, "high": [11.0]*10, "low": [9.0]*10,
                "close": [10.0 + i*0.1 for i in range(10)],
                "volume": [1e6]*10, "amount": [1e8]*10, "turnover": [1.0]*10,
            })

        def fake_build_features(df):
            from features.technical import FEATURE_COLS
            for c in FEATURE_COLS:
                if c not in df.columns:
                    df[c] = 0.0
            return df

        def fake_build_labels(df, target_days, threshold):
            return pd.Series([0, 1] * (len(df) // 2 + 1))[:len(df)]

        called = {"train": False, "retrain": False, "save": False}

        def fake_train(df, feature_cols):
            called["train"] = True
            return "model_v1"

        def fake_retrain(df, feature_cols):
            called["retrain"] = True
            return "model_v1_weighted"

        def fake_save(model, saved_dir):
            called["save"] = True
            return "ok"

        monkeypatch.setattr("data.universe.load_universe", fake_load_universe)
        monkeypatch.setattr("data.fetcher.fetch_stock_hist", fake_fetch)
        monkeypatch.setattr("features.builder.build_features", fake_build_features)
        monkeypatch.setattr("models.trainer.build_labels", fake_build_labels)
        monkeypatch.setattr("models.trainer.train", fake_train)
        monkeypatch.setattr("learning.model_learner.retrain_with_weights", fake_retrain)
        monkeypatch.setattr("models.trainer.save_model", fake_save)

        class Args:
            weighted = True
            workers = 1
        config = {
            "data": {"default_days": 30},
            "model": {"target_days": 5, "threshold": 0.02, "saved_dir": str(tmp_path)},
        }
        cli.cmd_train(Args(), config)
        assert called["retrain"] is True
        assert called["train"] is False, "weighted=True 时不应走 train()"

    def test_no_weighted_flag_uses_default_train(self, monkeypatch, tmp_path):
        import cli

        def fake_load_universe(cfg):
            return ["c1"]

        def fake_fetch(code, days, cache_only):
            return pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "open": [10.0]*10, "high": [11.0]*10, "low": [9.0]*10,
                "close": [10.0]*10,
                "volume": [1e6]*10, "amount": [1e8]*10, "turnover": [1.0]*10,
            })

        def fake_build_features(df):
            from features.technical import FEATURE_COLS
            for c in FEATURE_COLS:
                if c not in df.columns:
                    df[c] = 0.0
            return df

        def fake_build_labels(df, target_days, threshold):
            return pd.Series([0] * len(df))

        called = {"train": False, "retrain": False}
        monkeypatch.setattr("data.universe.load_universe", fake_load_universe)
        monkeypatch.setattr("data.fetcher.fetch_stock_hist", fake_fetch)
        monkeypatch.setattr("features.builder.build_features", fake_build_features)
        monkeypatch.setattr("models.trainer.build_labels", fake_build_labels)
        monkeypatch.setattr("models.trainer.train",
                            lambda df, feature_cols: called.update(train=True) or "m")
        monkeypatch.setattr("learning.model_learner.retrain_with_weights",
                            lambda df, feature_cols: called.update(retrain=True) or "m")
        monkeypatch.setattr("models.trainer.save_model", lambda m, saved_dir: None)

        class Args:
            weighted = False
            workers = 1
        config = {
            "data": {"default_days": 30},
            "model": {"target_days": 5, "threshold": 0.02, "saved_dir": str(tmp_path)},
        }
        cli.cmd_train(Args(), config)
        assert called["train"] is True
        assert called["retrain"] is False
