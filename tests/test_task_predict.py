"""tests/test_task_predict.py — task_predict.run() 特征列来源测试。

覆盖点：
- run() 调用 predict() 时传入的特征列来自 get_active_feature_cols()，而非硬编码 FEATURE_COLS
- get_active_feature_cols() 抛异常时回退到 FEATURE_COLS

设计说明：
  task_predict.run() 在函数体内用 import 动态引入 data/features/models 等模块，
  再用 importlib.reload 重新执行 run()。为了不触碰真实文件，
  用 patch("scripts.task_predict.yaml.safe_load") 替换配置读取，
  用 patch("scripts.task_predict.open", ...) 跳过 open("config.yaml")，
  其余依赖全部 mock。
"""
import io
from contextlib import ExitStack
from unittest.mock import patch, MagicMock, mock_open

import pandas as pd
import numpy as np
import pytest


# ── 常量 ────────────────────────────────────────────────────

_FEATURE_COLS_PRUNED = ["ma5", "ma10", "rsi6", "macd_dif"]


# ── 辅助 ──────────────────────────────────────────────────

def _make_minimal_df(n=65):
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date":   dates.strftime("%Y-%m-%d"),
        "open":   np.linspace(10, 12, n),
        "high":   np.linspace(10.5, 12.5, n),
        "low":    np.linspace(9.5, 11.5, n),
        "close":  np.linspace(10, 12, n),
        "volume": [1_000_000] * n,
    })


def _fake_config():
    return {
        "universe": {"watchlist": ["000001"]},
        "model":    {"saved_dir": "models/saved"},
        "feishu":   {"webhook_url": "http://fake-webhook"},
    }


def _fake_predict_result():
    return {
        "rise_prob":  0.65,
        "signal":     "买入",
        "confidence": "高",
    }


def _fake_price_info():
    return {
        "pred_high": 12.5,
        "pred_low":  10.5,
        "price":     11.0,
        "pct":       0.5,
        "high":      11.5,
        "low":       10.5,
        "open":      10.8,
        "close":     11.0,
    }


def _build_patches(fake_active_cols_or_exc, fake_predict_fn):
    """返回 run() 所需的全套 patch 列表。

    fake_active_cols_or_exc:
      - list  → get_active_feature_cols 返回该列表
      - Exception 实例 → get_active_feature_cols 抛出该异常
    """
    minimal_df = _make_minimal_df()

    if isinstance(fake_active_cols_or_exc, Exception):
        gafc_patch = patch(
            "features.technical.get_active_feature_cols",
            side_effect=fake_active_cols_or_exc,
        )
    else:
        gafc_patch = patch(
            "features.technical.get_active_feature_cols",
            return_value=fake_active_cols_or_exc,
        )

    # task_predict.run() 用 `with open("config.yaml") as f: yaml.safe_load(f)`
    # 需要 open 返回可作为 context manager 的对象，yaml.safe_load 收到文件对象。
    # 最简单：让 yaml.safe_load 直接 return 配置，open 给一个假 file-like。
    fake_open = mock_open(read_data="")  # yaml 读取文件对象，返回值由 yaml.safe_load mock 决定

    return [
        patch("scripts.task_predict.is_trading_day",   return_value=True),
        patch("scripts.task_predict.is_trading_time",   return_value=True),
        # 在 task_predict 模块里 open() 被调用时直接绕过磁盘
        patch("scripts.task_predict.open",              fake_open, create=True),
        # yaml.safe_load 无论收到什么都返回我们的配置
        patch("scripts.task_predict.yaml.safe_load",    return_value=_fake_config()),
        patch("data.fetcher.fetch_stock_hist",           return_value=minimal_df),
        patch("data.fetcher.fetch_realtime_prices",      return_value={}),
        patch("data.fetcher.fetch_intraday_kline",       return_value=minimal_df),
        patch("data.fetcher._load_name_map",             return_value={}),
        patch("features.builder.build_features",         side_effect=lambda df, **kw: df),
        gafc_patch,
        patch("features.analyser.analyse",               return_value="分析文本"),
        patch("features.analyser.predict_range",         return_value=_fake_price_info()),
        patch("features.analyser.text_intraday_kline",   return_value="分时文本"),
        patch("models.predictor.predict",                side_effect=fake_predict_fn),
        patch("models.predictor.load_model",             return_value=MagicMock()),
        patch("notify.feishu.send_predict_results",      return_value=True),
        patch("data.alt_fetcher.fetch_alt_features",     return_value={}),
    ]


# ── 核心测试：特征列来自 get_active_feature_cols() ──────────

def test_run_uses_get_active_feature_cols():
    """run() 调用 predict() 时，特征列必须来自 get_active_feature_cols()，而非硬编码 FEATURE_COLS。"""
    captured_feature_cols = []

    def _fake_predict(df, feature_cols, model=None):
        captured_feature_cols.append(list(feature_cols))
        return _fake_predict_result()

    import importlib
    import scripts.task_predict as tp
    importlib.reload(tp)  # 确保模块干净，reload 必须在 patch 之前

    with ExitStack() as stack:
        for p in _build_patches(_FEATURE_COLS_PRUNED[:], _fake_predict):
            stack.enter_context(p)

        tp.run()

    assert len(captured_feature_cols) == 1, (
        f"predict 应被调用 1 次，实际 {len(captured_feature_cols)} 次"
    )
    assert captured_feature_cols[0] == _FEATURE_COLS_PRUNED, (
        f"期望特征列={_FEATURE_COLS_PRUNED!r}, 实际={captured_feature_cols[0]!r}"
    )


# ── 回退测试：get_active_feature_cols() 抛异常时使用 FEATURE_COLS ──

def test_run_falls_back_to_FEATURE_COLS_on_exception():
    """get_active_feature_cols() 抛异常时，run() 回退到完整的 FEATURE_COLS。"""
    captured_feature_cols = []

    def _fake_predict(df, feature_cols, model=None):
        captured_feature_cols.append(list(feature_cols))
        return _fake_predict_result()

    import importlib
    import scripts.task_predict as tp
    importlib.reload(tp)  # reload 必须在 patch 之前

    with ExitStack() as stack:
        for p in _build_patches(RuntimeError("feature_weights.json missing"), _fake_predict):
            stack.enter_context(p)

        tp.run()

    assert len(captured_feature_cols) == 1

    from features.technical import FEATURE_COLS
    assert captured_feature_cols[0] == FEATURE_COLS, (
        f"回退路径应用 FEATURE_COLS，实际={captured_feature_cols[0]!r}"
    )


# ── alt 注入测试 ─────────────────────────────────────────────

def _build_patches_with_alt(fake_active_cols_or_exc, fake_predict_fn, fake_alt_map=None, alt_side_effect=None):
    """扩展 _build_patches，额外注入 fetch_alt_features mock。"""
    base = _build_patches(fake_active_cols_or_exc, fake_predict_fn)

    if alt_side_effect is not None:
        alt_patch = patch(
            "data.alt_fetcher.fetch_alt_features",
            side_effect=alt_side_effect,
        )
    else:
        alt_patch = patch(
            "data.alt_fetcher.fetch_alt_features",
            return_value=fake_alt_map or {},
        )

    return base + [alt_patch]


def test_run_passes_alt_to_build_features():
    """run() 在调用 build_features 时应传 alt kwarg，且该 alt 来自 fetch_alt_features。"""
    code = "000001"
    fake_alt_dict = {
        "main_net_in_1d":     0.3,
        "main_net_in_5d":     0.1,
        "dragon_top_cnt_10d": 0.0,
        "sector_heat_rank":   0.4,
        "north_hold_chg_5d":  0.2,
    }
    fake_alt_map = {code: fake_alt_dict}

    captured_alt: list = []

    def _fake_build_features(df, **kwargs):
        captured_alt.append(kwargs.get("alt"))
        return df

    def _fake_predict(df, feature_cols, model=None):
        return _fake_predict_result()

    import importlib
    import scripts.task_predict as tp
    importlib.reload(tp)

    patches = _build_patches_with_alt(
        _FEATURE_COLS_PRUNED[:],
        _fake_predict,
        fake_alt_map=fake_alt_map,
    )
    # 替换 build_features patch 为能捕获 alt 的版本
    patches = [
        p for p in patches
        if not (hasattr(p, "attribute") and p.attribute == "build_features")
    ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(
            patch("features.builder.build_features", side_effect=_fake_build_features)
        )

        tp.run()

    assert len(captured_alt) >= 1, "build_features 应被调用至少一次"
    alt_received = captured_alt[0]
    assert alt_received is not None, "alt kwarg 不应为 None"
    assert alt_received != {}, "alt kwarg 不应为空 dict"
    assert alt_received == fake_alt_dict, (
        f"期望 alt={fake_alt_dict!r}, 实际={alt_received!r}"
    )


def test_run_alt_fetch_failure_does_not_crash():
    """fetch_alt_features 失败时 run() 不崩溃，build_features 收到空 dict。"""
    captured_alt: list = []

    def _fake_build_features(df, **kwargs):
        captured_alt.append(kwargs.get("alt"))
        return df

    def _fake_predict(df, feature_cols, model=None):
        return _fake_predict_result()

    import importlib
    import scripts.task_predict as tp
    importlib.reload(tp)

    patches = _build_patches_with_alt(
        _FEATURE_COLS_PRUNED[:],
        _fake_predict,
        alt_side_effect=RuntimeError("network error"),
    )
    patches = [
        p for p in patches
        if not (hasattr(p, "attribute") and p.attribute == "build_features")
    ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(
            patch("features.builder.build_features", side_effect=_fake_build_features)
        )

        tp.run()  # should not raise

    assert len(captured_alt) >= 1, "build_features 应被调用"
    assert captured_alt[0] == {}, (
        f"alt fetch 失败时应回退到空 dict, 实际={captured_alt[0]!r}"
    )
