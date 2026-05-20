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

    with patch("learning.price_learner.load_price_params", return_value=custom_pp), \
         patch("learning.market_state.load_current_state", return_value={"current": "bull"}):
        result = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    s = result["short"]
    # s_buy = 10.0 (cur fallback, no ma5/bb)
    # s_stop_atr = round(10.0 - 0.3 * 0.5, 2) = 9.85
    # s_stop_pct = round(10.0 * 0.98, 2) = 9.8
    # atr_stop(9.85) > pct_stop(9.8) so atr branch wins
    assert s["stop_price"] == pytest.approx(9.85)
    assert "ATR" in s["stop_desc"]


def test_lower_atr_mult_produces_tighter_stop():
    """A lower short_atr_mult results in a higher (tighter) ATR stop price."""
    from features.analyser import suggest_dual_period_trades

    mult_lo = {"short_atr_mult": 0.5, "short_gain_mult": 2.2,
               "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    mult_hi = {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
               "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    # atr=0.3: atr_stop_lo = 10.0 - 0.15 = 9.85 (> pct_stop 9.8 → atr branch)
    #          atr_stop_hi = 10.0 - 0.45 = 9.55 (< pct_stop 9.8 → pct branch)
    last = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.3)
    price_info = _make_price_info(cur=10.0)

    with patch("learning.price_learner.load_price_params", return_value=mult_lo), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        r_lo = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    with patch("learning.price_learner.load_price_params", return_value=mult_hi), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        r_hi = suggest_dual_period_trades(last, price_info, rise_prob=0.5)

    assert r_lo["short"]["stop_price"] != r_hi["short"]["stop_price"]
    assert r_lo["short"]["stop_price"] > r_hi["short"]["stop_price"]  # lower mult → tighter stop


def test_short_sell_uses_gain_mult():
    """short sell fallback uses up_ratio * _SHORT_GAIN_MULT."""
    from features.analyser import suggest_dual_period_trades

    custom_pp = {"short_atr_mult": 1.5, "short_gain_mult": 3.0,
                 "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    last = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.0)
    price_info = {"price": 10.0, "up_ratio": 2.0}

    with patch("learning.price_learner.load_price_params", return_value=custom_pp), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}):
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

    with patch("learning.price_learner.load_price_params", return_value=custom_pp), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}):
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

    # Patch the module attribute — works because analyser uses deferred import,
    # so `from learning.price_learner import name` re-binds from the patched module each call.
    with patch("learning.price_learner.load_price_params", side_effect=fake_load), \
         patch("learning.market_state.load_current_state", return_value={"current": "bear"}):
        suggest_dual_period_trades(_make_last(), _make_price_info(), rise_prob=0.5)

    assert calls == ["bear"]


def test_default_params_fallback_when_dict_empty():
    """Empty pp dict triggers .get() defaults; stop_desc confirms short_atr_mult=1.5 applied."""
    from features.analyser import suggest_dual_period_trades

    # atr=0.3, cur=10.0: atr_stop = 10.0 - 0.3*1.5(default) = 9.55 < pct_stop(9.8)
    # → pct branch; but with atr=0.05: atr_stop = 10.0 - 0.05*1.5 = 9.925 > 9.8 → ATR branch
    last = _make_last(ma5=0.0, ma20=0.0, ma60=0.0, bb_upper=0.0, bb_lower=0.0, atr=0.05)
    with patch("learning.price_learner.load_price_params", return_value={}), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}):
        result = suggest_dual_period_trades(last, _make_price_info(cur=10.0), rise_prob=0.5)

    s = result["short"]
    assert "short" in result and "long" in result
    assert s["buy_price"] > 0
    assert s["sell_price"] > s["buy_price"]
    # ATR stop should apply with default short_atr_mult=1.5: stop = round(10.0 - 0.05*1.5, 2) = 9.93
    assert s["stop_price"] == pytest.approx(9.93)
    assert "ATR" in s["stop_desc"]
