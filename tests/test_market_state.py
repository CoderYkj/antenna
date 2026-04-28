import pandas as pd
import pytest
from pathlib import Path
from learning import market_state


def _make_hs300_df(close_series: list[float], start: str = "2026-01-02") -> pd.DataFrame:
    """构造连续交易日的沪深 300 DataFrame(open/high/low 简化为 close ±0.5%)。"""
    dates = pd.bdate_range(start=start, periods=len(close_series))
    return pd.DataFrame({
        "date":   dates,
        "open":   [c * 0.995 for c in close_series],
        "close":  close_series,
        "high":   [c * 1.005 for c in close_series],
        "low":    [c * 0.995 for c in close_series],
        "volume": [1e8] * len(close_series),
    })


class TestComputeState:
    def test_bull_when_above_ma60_and_60d_return_positive(self):
        """60 日匀速上涨 10%:close > ma60 且 ret_60d > 5% → bull。"""
        closes = [3000 + i * 5 for i in range(90)]  # 3000 → 3445,涨 ~14%
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        assert state["state"] == "bull"
        assert state["ret_60d"] > 0.05

    def test_bear_when_below_ma60_and_60d_return_negative(self):
        closes = [3500 - i * 5 for i in range(90)]  # 3500 → 3055,跌 ~13%
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        assert state["state"] == "bear"
        assert state["ret_60d"] < -0.05

    def test_range_when_flat(self):
        closes = [3500 + (i % 10 - 5) * 2 for i in range(90)]
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        assert state["state"] == "range"

    def test_high_volatility_forces_range(self):
        """20 日 ATR/close > 2.5% 强制 range,即使趋势向上。"""
        import numpy as np
        np.random.seed(42)
        closes = [3000 + i * 3 + np.random.uniform(-150, 150) for i in range(90)]
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        # atr_pct 高波动应触发 range
        if state["atr_pct"] > 0.025:
            assert state["state"] == "range"

    def test_insufficient_data_returns_range(self):
        df = _make_hs300_df([3500] * 30)  # 只有 30 天,不足 60
        state = market_state.compute_state(df)
        assert state["state"] == "range"
        assert state.get("reason") == "insufficient_data"


class TestDebounceSwitch:
    def test_switch_requires_3_consecutive_days(self, tmp_path, monkeypatch):
        monkeypatch.setattr(market_state, "STATE_FILE", tmp_path / "market_state.json")

        # 第 1 日触发 bull(从无到 bull,直接切 OK,无需防抖)
        closes_bull = [3000 + i * 5 for i in range(90)]
        df_bull = _make_hs300_df(closes_bull)
        market_state.update_state(df_bull, date_str="2026-04-01")
        assert market_state.load_current_state()["current"] == "bull"

        # 第 2 日触发 bear 单次 — 不应立即切,存入 pending
        closes_bear = [3500 - i * 5 for i in range(90)]
        df_bear = _make_hs300_df(closes_bear)
        market_state.update_state(df_bear, date_str="2026-04-02")
        state = market_state.load_current_state()
        assert state["current"] == "bull"  # 仍是 bull
        assert state["pending"] == "bear"

        # 第 3 日仍 bear
        market_state.update_state(df_bear, date_str="2026-04-03")
        assert market_state.load_current_state()["current"] == "bull"

        # 第 4 日仍 bear,连续 3 日达成,切换
        market_state.update_state(df_bear, date_str="2026-04-04")
        assert market_state.load_current_state()["current"] == "bear"


class TestStateOnDate:
    def test_load_state_on_date_uses_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(market_state, "STATE_FILE", tmp_path / "market_state.json")
        # 构造带 history 的 state 文件
        import json
        data = {
            "current": "range",
            "since":   "2026-04-20",
            "pending": None,
            "hs300":   {"close": 3500, "ma60": 3490, "ret_60d": 0.01, "atr_pct": 0.018},
            "history": [
                {"date": "2026-04-15", "state": "bull"},
                {"date": "2026-04-16", "state": "bull"},
                {"date": "2026-04-17", "state": "range"},
                {"date": "2026-04-20", "state": "range"},
            ],
        }
        (tmp_path / "market_state.json").write_text(json.dumps(data), encoding="utf-8")
        assert market_state.load_state_on_date("2026-04-15") == "bull"
        assert market_state.load_state_on_date("2026-04-17") == "range"
        # 无记录的日期回退默认
        assert market_state.load_state_on_date("2025-01-01") == "range"
