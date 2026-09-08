"""tests/test_predict_cmd.py — cmd_scan_bot alt_data 集成测试。

覆盖点：
- cmd_scan_bot 在线程池启动前调用 fetch_alt_features
- per-code alt dict 通过 alt_cache.get(code, {}) 传入 _scan_one
- fetch_alt_features 失败时 scan 不崩溃，降级为空 alt_cache
- _scan_one 新增 alt 参数有默认值 None，向后兼容
"""
import ast
import pathlib
from concurrent.futures import Future
from contextlib import ExitStack
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest


# ── 常量路径 ────────────────────────────────────────────────
_CMD_FILE = pathlib.Path(__file__).parent.parent / "server" / "predict_cmd.py"


# ── 辅助 ──────────────────────────────────────────────────

def _make_minimal_df():
    """返回满足 build_features 最低要求的 DataFrame（65 行）。"""
    n = 65
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date":   dates.strftime("%Y-%m-%d"),
        "open":   np.linspace(10, 12, n),
        "high":   np.linspace(10.5, 12.5, n),
        "low":    np.linspace(9.5, 11.5, n),
        "close":  np.linspace(10, 12, n),
        "volume": [1_000_000] * n,
    })


def _fake_result(code, minimal_df):
    return {
        "code":          code,
        "rise_prob":     0.6,
        "signal":        "买入",
        "confidence":    "高",
        "momentum":      1.0,
        "last":          minimal_df.iloc[-1].to_dict(),
        "price_info":    {"pred_high": 12.5, "pred_low": 10.5},
        "drawdown":      -0.05,
        "rise_prob_raw": 0.6,
        "rise_prob_cal": 0.6,
        "self_rank_pct": 0.05,
        "global_rank":   1,
        "global_rank_pct": 0.05,
    }


class _FakePool:
    """替换 ThreadPoolExecutor，拦截 submit 并记录调用参数。"""

    def __init__(self, submitted_calls, scan_one_fn):
        self._submitted_calls = submitted_calls
        self._fn = scan_one_fn

    def __call__(self, max_workers=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def submit(self, fn, *args, **kwargs):
        self._submitted_calls.append((args, kwargs))
        result = self._fn(*args, **kwargs)
        f = Future()
        f.set_result(result)
        return f


def _build_patches(codes, alt_patch):
    """返回所有重量级依赖的 patch list，alt_patch 是对 fetch_alt_features 的 MagicMock。"""
    fake_cfg = {
        "model":    {"saved_dir": "models/saved"},
        "universe": {"scan_pool": "watchlist", "watchlist": []},
        "scan":     {"workers": 2},
    }
    fake_strategy = {
        "buy_top_pct": 0.10,
        "last_thresh_buy": 0,
        "last_thresh_watch": 0,
        "last_scan_date": "",
        "last_scan_total": 0,
    }
    return [
        patch("server.predict_cmd._load_cfg",         return_value=fake_cfg),
        patch("learning.optimizer.load_strategy",      return_value=fake_strategy),
        patch("data.universe.load_universe",           return_value=codes),
        patch("models.predictor.load_model",           return_value=MagicMock()),
        patch("models.predictor.assign_global_signals"),
        patch("learning.optimizer.save_strategy"),
        patch("learning.tracker.log_predictions"),
        patch("data.fetcher.fetch_realtime_prices",    return_value={}),
        patch("data.fetcher._load_name_map",           return_value={}),
        patch("features.analyser.suggest_trade_levels",        return_value={}),
        patch("features.analyser.suggest_dual_period_trades",  return_value={}),
        patch("server.predict_cmd._enrich_tactic_scores",
              side_effect=lambda r, **kw: r),
        patch("server.predict_cmd._apply_resonance_boost",
              side_effect=lambda r: r),
        patch("data.fetcher.fetch_hot_sectors",        return_value=[]),
        patch("data.fetcher.get_stock_sector",         return_value=""),
        patch("features.analyser.fetch_cls_news_batch"),
        patch("features.analyser.fetch_cls_news_for", return_value=([], [])),
        patch("features.analyser.analyse",             return_value=""),
        patch("features.analyser.build_bull_reasons",  return_value=([], [])),
        # 让 cmd_scan_bot 的后台线程同步执行，避免断言与线程的竞态条件
        patch(
            "threading.Thread",
            side_effect=lambda target, args=(), daemon=False, **kwargs: type(
                "_SyncThread",
                (),
                {"start": staticmethod(lambda: target(*args))},
            )(),
        ),
        patch("server.predict_cmd._predict_date",      return_value=("2026-05-21", True)),
        patch("data.alt_fetcher.fetch_alt_features",   alt_patch),
    ]


# ── _scan_one 签名向后兼容 ────────────────────────────────────

def test_scan_one_has_alt_param_with_default():
    """_scan_one 闭包中应声明 alt 参数，且有默认值 None。"""
    src = _CMD_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src)

    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_scan_one":
            args_names = [a.arg for a in node.args.args]
            defaults_count = len(node.args.defaults)
            found = (args_names, defaults_count)
            break

    assert found is not None, "_scan_one 未在 predict_cmd.py 中找到"
    args_names, defaults_count = found
    assert "alt" in args_names, "_scan_one 应有 alt 参数"
    assert defaults_count >= 1, "alt 应有默认值"


# ── alt_cache 传入 _scan_one ─────────────────────────────────

def test_cmd_scan_bot_passes_alt_per_code():
    """cmd_scan_bot 把 alt_cache.get(code, {}) 正确传给每个 _scan_one 调用。"""
    minimal_df = _make_minimal_df()
    codes = ["000001", "000002"]
    fake_alt = {
        "000001": {"news_sentiment": 0.5, "margin_balance_chg": 0.1,
                   "north_flow_5d": 0.2, "sector_momentum_5d": 0.3,
                   "limit_up_ratio_5d": 0.4},
        "000002": {"news_sentiment": -0.1, "margin_balance_chg": None,
                   "north_flow_5d": None, "sector_momentum_5d": None,
                   "limit_up_ratio_5d": None},
    }

    submitted_calls: list = []

    def _fake_scan_one(code, alt=None):
        return _fake_result(code, minimal_df)

    pool_instance = _FakePool(submitted_calls, _fake_scan_one)
    mock_fetch_alt = MagicMock(return_value=fake_alt)

    with ExitStack() as stack:
        for p in _build_patches(codes, mock_fetch_alt):
            stack.enter_context(p)
        stack.enter_context(
            patch("concurrent.futures.ThreadPoolExecutor", pool_instance)
        )
        stack.enter_context(
            patch("concurrent.futures.as_completed",
                  side_effect=lambda futs: iter(futs))
        )

        from server.predict_cmd import cmd_scan_bot
        cmd_scan_bot(top_n=2)

    # fetch_alt_features 应被调用一次
    mock_fetch_alt.assert_called_once()
    import re
    call_codes, call_date = mock_fetch_alt.call_args[0]
    assert set(call_codes) == set(codes)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", call_date), (
        f"call_date {call_date!r} does not match YYYY-MM-DD"
    )

    # 每个股票提交时都带了对应的 alt dict
    assert len(submitted_calls) == len(codes)
    for args, kwargs in submitted_calls:
        code = args[0]
        alt_passed = args[1] if len(args) > 1 else kwargs.get("alt")
        expected = fake_alt.get(code, {})
        assert alt_passed == expected, (
            f"code={code} expected alt={expected!r}, got {alt_passed!r}"
        )


# ── alt_data fetch 失败降级 ───────────────────────────────────

def test_cmd_scan_bot_alt_fetch_failure_does_not_crash():
    """fetch_alt_features 抛异常时，scan 正常执行，alt 降级为空 dict。"""
    minimal_df = _make_minimal_df()
    codes = ["000001"]

    submitted_calls: list = []

    def _fake_scan_one(code, alt=None):
        return {
            "code":          code,
            "rise_prob":     0.55,
            "signal":        "观望",
            "confidence":    "中",
            "momentum":      0.8,
            "last":          minimal_df.iloc[-1].to_dict(),
            "price_info":    {"pred_high": None, "pred_low": None},
            "drawdown":      -0.03,
            "rise_prob_raw": 0.55,
            "rise_prob_cal": 0.55,
            "self_rank_pct": 0.2,
            "global_rank":   5,
            "global_rank_pct": 0.2,
        }

    pool_instance = _FakePool(submitted_calls, _fake_scan_one)
    mock_fetch_alt = MagicMock(side_effect=RuntimeError("network error"))

    with ExitStack() as stack:
        for p in _build_patches(codes, mock_fetch_alt):
            stack.enter_context(p)
        stack.enter_context(
            patch("concurrent.futures.ThreadPoolExecutor", pool_instance)
        )
        stack.enter_context(
            patch("concurrent.futures.as_completed",
                  side_effect=lambda futs: iter(futs))
        )

        from server.predict_cmd import cmd_scan_bot
        result = cmd_scan_bot(top_n=1)

    # scan 不崩溃，返回正常卡片
    assert isinstance(result, dict)
    assert result.get("msg_type") == "interactive"

    # alt_data fetch 被调用（尽管失败）
    mock_fetch_alt.assert_called_once()

    # 降级后，alt 传入为空 dict（因为 alt_cache = {} → .get(code, {}) = {}）
    assert len(submitted_calls) == 1
    args, kwargs = submitted_calls[0]
    alt_passed = args[1] if len(args) > 1 else kwargs.get("alt")
    assert alt_passed == {}, f"Expected empty dict on alt fetch failure, got {alt_passed!r}"


def test_cmd_scan_bot_rejects_overlapping_scan():
    """同一进程内已有扫描时，不应再次启动后台扫描并广播重复结果。"""
    from server.predict_cmd import _scan_lock, cmd_scan_bot

    assert _scan_lock.acquire(blocking=False)
    try:
        result = cmd_scan_bot(top_n=1)
    finally:
        _scan_lock.release()

    assert result["msg_type"] == "text"
    assert "正在进行" in result["content"]


# ── cmd_predict 单股 alt 注入 ────────────────────────────────

def _build_predict_patches(code: str, alt_data: dict, alt_side_effect=None):
    """返回 cmd_predict() 所需的 patch 列表。

    alt_data: fetch_alt_features 正常返回时的值
    alt_side_effect: 若非 None，fetch_alt_features 抛该异常（测试 fallback）
    """
    minimal_df = _make_minimal_df()
    fake_cfg = {
        "model": {"saved_dir": "models/saved"},
    }
    fake_strategy = {
        "buy_top_pct": 0.10,
        "last_thresh_buy": 0,
        "last_thresh_watch": 0,
        "last_scan_date": "",
        "last_scan_total": 0,
    }

    if alt_side_effect is not None:
        alt_patch = patch(
            "data.alt_fetcher.fetch_alt_features",
            side_effect=alt_side_effect,
        )
    else:
        alt_patch = patch(
            "data.alt_fetcher.fetch_alt_features",
            return_value=alt_data,
        )

    return [
        patch("server.predict_cmd._load_cfg",          return_value=fake_cfg),
        patch("learning.optimizer.load_strategy",       return_value=fake_strategy),
        patch("data.fetcher.fetch_stock_hist",          return_value=minimal_df),
        patch("features.builder.build_features",        side_effect=lambda df, **kw: df),
        patch("features.technical.get_active_feature_cols", return_value=["ma5", "rsi6"]),
        patch("models.predictor.load_model",            return_value=MagicMock()),
        patch("models.predictor.predict",               return_value={
            "rise_prob": 0.65,
            "signal": "买入",
            "confidence": "高",
            "self_rank_pct": 0.05,
        }),
        patch("features.analyser.analyse",              return_value="技术分析文本"),
        patch("features.analyser.predict_range",        return_value={
            "pred_high": 12.5, "pred_low": 10.5, "price": 11.0,
            "pct": 0.5, "high": 11.5, "low": 10.5, "open": 10.8,
        }),
        patch("features.analyser.build_commentary",     return_value=("点评", "依据")),
        patch("features.analyser.suggest_holding",      return_value=("短线", "理由")),
        patch("features.analyser.suggest_trade_levels", return_value={}),
        patch("features.analyser.text_intraday_kline",  return_value=""),
        patch("data.fetcher.fetch_realtime_prices",     return_value={}),
        patch("data.fetcher.fetch_intraday_kline",      return_value=None),
        patch("learning.tracker.load_predictions",      return_value=[]),
        alt_patch,
    ]


def test_cmd_predict_passes_non_empty_alt_to_build_features():
    """cmd_predict 调用 build_features 时，alt kwarg 非空（非 None、非 {}）。"""
    code = "000001"
    fake_alt_dict = {"main_net_in_1d": 0.3, "main_net_in_5d": 0.1,
                     "dragon_top_cnt_10d": 0.0, "sector_heat_rank": 0.4,
                     "north_hold_chg_5d": 0.2}
    alt_data = {code: fake_alt_dict}

    captured_alt = []

    def _fake_build_features(df, **kwargs):
        captured_alt.append(kwargs.get("alt"))
        return df

    patches = _build_predict_patches(code, alt_data)
    # override build_features patch to capture alt
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

        from server import predict_cmd
        import importlib
        importlib.reload(predict_cmd)
        predict_cmd.cmd_predict(code)

    assert len(captured_alt) >= 1, "build_features should have been called"
    alt_received = captured_alt[0]
    assert alt_received is not None, "alt kwarg should not be None"
    assert alt_received != {}, "alt kwarg should not be empty dict"
    assert alt_received == fake_alt_dict, (
        f"Expected alt={fake_alt_dict!r}, got {alt_received!r}"
    )


def test_cmd_predict_alt_fetch_failure_fallback():
    """fetch_alt_features 失败时，cmd_predict 不崩溃，build_features 收到空 dict。"""
    code = "000001"

    captured_alt = []

    def _fake_build_features(df, **kwargs):
        captured_alt.append(kwargs.get("alt"))
        return df

    patches = _build_predict_patches(code, {}, alt_side_effect=RuntimeError("network"))
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

        from server import predict_cmd
        import importlib
        importlib.reload(predict_cmd)
        result = predict_cmd.cmd_predict(code)

    # cmd_predict 不崩溃，返回正常结果
    assert isinstance(result, (dict, str)), "cmd_predict should return dict or str"
    # alt fallback 为空 dict
    assert len(captured_alt) >= 1, "build_features should have been called"
    assert captured_alt[0] == {}, (
        f"Expected empty dict on alt failure, got {captured_alt[0]!r}"
    )
