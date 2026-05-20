"""tests/test_analyser_params.py - integration tests for suggest_dual_period_trades
reading price params via load_price_params.
"""
import pytest
from unittest.mock import patch


def _make_last(ma5=10.0, ma20=9.5, ma60=9.0, bb_upper=11.0, bb_lower=9.0, atr=0.2):
    return {
        "ma5": ma5, "ma20": ma20, "ma60": ma60,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "atr": atr,
    }


def _make_price_info(cur=10.5):
    return {"price": cur, "up_ratio": 2.0}


def test_atr_stop_uses_custom_mult():
    """short ATR stop uses _SHORT_ATR_MULT instead of hardcoded 1.5."""
    from features.analyser import suggest_dual_period_trades

    custom_pp = {"short_atr_mult": 0.5, "short_gain_mult": 2.2,
                 "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    last = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.3)
    price_info = _make_price_info(cur=10.0)

    with patch("learning.price_learner.load_price_params", return_value=custom_pp),          patch("learning.market_state.load_current_state", return_value={"current": "bull"}):
        result = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    s = result["short"]
    # s_buy = 10.0 (cur fallback, no ma5/bb)
    # s_stop_atr = round(10.0 - 0.3 * 0.5, 2) = 9.85
    # s_stop_pct = round(10.0 * 0.98, 2) = 9.8
    # atr_stop(9.85) > pct_stop(9.8) so atr branch wins
    assert s["stop_price"] == pytest.approx(9.85)
    assert "ATR" in s["stop_desc"]


def test_atr_stop_hardcoded_1_5_would_not_match():
    """Confirm that with mult=0.5 the stop differs from what mult=1.5 would produce."""
    from features.analyser import suggest_dual_period_trades

    mult_05 = {"short_atr_mult": 0.5, "short_gain_mult": 2.2,
               "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    mult_15 = {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
               "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    last = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=1.0)
    price_info = _make_price_info(cur=10.0)

    with patch("learning.price_learner.load_price_params", return_value=mult_05),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        r05 = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    with patch("learning.price_learner.load_price_params", return_value=mult_15),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        r15 = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    # With atr=1.0: stop_atr_05 = 10.0 - 1.0*0.5 = 9.5; stop_atr_15 = 10.0 - 1.0*1.5 = 8.5
    # pct_stop = 9.8; 9.5 > 9.8? No. 8.5 > 9.8? No. Both fall to pct stop 9.8 actually
    # Use atr=0.3 to get atr > pct:
    # atr_05 = 10.0 - 0.3*0.5 = 9.85 > 9.8 → atr branch
    # atr_15 = 10.0 - 0.3*1.5 = 9.55 < 9.8 → pct branch (9.8)
    # so with atr=1.0, both use pct stop 9.8 and will be equal.
    # This test checks the atr stop desc when atr is dominant, as in test_atr_stop_uses_custom_mult.
    # Instead just check stop prices are different when atr=0.3
    last2 = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.3)
    with patch("learning.price_learner.load_price_params", return_value=mult_05),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        r05b = suggest_dual_period_trades(last2, price_info, rise_prob=0.5)
    with patch("learning.price_learner.load_price_params", return_value=mult_15),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        r15b = suggest_dual_period_trades(last2, price_info, rise_prob=0.5)

    assert r05b["short"]["stop_price"] != r15b["short"]["stop_price"]


def test_short_sell_uses_gain_mult():
    """short sell fallback uses up_ratio * _SHORT_GAIN_MULT."""
    from features.analyser import suggest_dual_period_trades

    custom_pp = {"short_atr_mult": 1.5, "short_gain_mult": 3.0,
                 "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    last = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.0)
    price_info = {"price": 10.0, "up_ratio": 2.0}

    with patch("learning.price_learner.load_price_params", return_value=custom_pp),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        result = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    s = result["short"]
    # up_ratio = 0.02; s_buy = 10.0
    # s_sell = round(10.0 * (1 + 0.02 * 3.0), 2) = round(10.6, 2) = 10.6
    # s_sell(10.6) > s_buy * 1.005(10.05) so no guard triggered
    assert s["sell_price"] == pytest.approx(10.6)


def test_long_sell_uses_amp_mult():
    """long sell fallback uses up_ratio * _LONG_AMP_MULT."""
    from features.analyser import suggest_dual_period_trades

    custom_pp = {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
                 "long_amp_mult": 4.0, "long_ma60_buffer": 0.97}
    last = _make_last(ma5=0.0, ma20=9.5, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.0)
    price_info = {"price": 10.0, "up_ratio": 2.0}

    with patch("learning.price_learner.load_price_params", return_value=custom_pp),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        result = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    long = result["long"]
    l_buy = round(9.5 * 0.997, 2)
    expected_sell = round(l_buy * (1 + 0.02 * 4.0), 2)
    assert long["sell_price"] == pytest.approx(expected_sell)


def test_market_state_passed_to_load_price_params():
    """load_current_state() result is forwarded to load_price_params."""
    from features.analyser import suggest_dual_period_trades

    calls = []

    def fake_load(state):
        calls.append(state)
        return {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
                "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}

    with patch("learning.price_learner.load_price_params", side_effect=fake_load),          patch("learning.market_state.load_current_state", return_value={"current": "bear"}):
        suggest_dual_period_trades(_make_last(), _make_price_info(), rise_prob=0.5)

    assert calls == ["bear"]


def test_default_params_fallback_when_dict_empty():
    """Empty pp dict triggers .get() defaults; function still returns valid result."""
    from features.analyser import suggest_dual_period_trades

    with patch("learning.price_learner.load_price_params", return_value={}),          patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        result = suggest_dual_period_trades(_make_last(), _make_price_info(), rise_prob=0.5)

    assert "short" in result and "long" in result
    assert result["short"]["buy_price"] > 0
    assert result["short"]["sell_price"] > result["short"]["buy_price"]
    assert result["long"]["buy_price"] > 0
