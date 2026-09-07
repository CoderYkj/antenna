"""tests/test_cmd_scan_filter.py — cmd_scan_bot 战法过滤逻辑单元测试。"""


def test_tier_split_basic():
    from server.predict_cmd import _tier_split
    enriched = [
        {"rise_prob": 0.70, "tactic_resonance": 3, "global_rank_pct": 0.01},
        {"rise_prob": 0.65, "tactic_resonance": 2, "global_rank_pct": 0.02},
        {"rise_prob": 0.60, "tactic_resonance": 1, "global_rank_pct": 0.05},
        {"rise_prob": 0.55, "tactic_resonance": 0, "global_rank_pct": 0.08},
    ]
    tier1, tier2 = _tier_split(enriched)
    assert len(tier1) == 2
    assert len(tier2) == 1
    assert all(r["tactic_resonance"] >= 2 for r in tier1)
    assert all(r["tactic_resonance"] == 1 for r in tier2)


def test_tier_split_empty():
    from server.predict_cmd import _tier_split
    tier1, tier2 = _tier_split([])
    assert tier1 == []
    assert tier2 == []


def test_tier_split_all_excluded():
    from server.predict_cmd import _tier_split
    enriched = [{"tactic_resonance": 0}] * 5
    tier1, tier2 = _tier_split(enriched)
    assert tier1 == [] and tier2 == []


def test_final_top_order_tier1_before_tier2():
    """final_top = (tier1 + tier2)[:top_n] 应保证 tier1 在前，即使 tier2 的 rank 更好。"""
    from server.predict_cmd import _tier_split
    enriched = [
        {"tactic_resonance": 1, "global_rank_pct": 0.01},  # tier2, better rank
        {"tactic_resonance": 2, "global_rank_pct": 0.05},  # tier1, worse rank
    ]
    tier1, tier2 = _tier_split(enriched)
    final = (tier1 + tier2)[:5]
    assert final[0]["tactic_resonance"] == 2
    assert final[1]["tactic_resonance"] == 1


def test_tier_split_high_resonance_all_tier1():
    from server.predict_cmd import _tier_split
    enriched = [{"tactic_resonance": i} for i in range(4, -1, -1)]  # 4,3,2,1,0
    tier1, tier2 = _tier_split(enriched)
    assert len(tier1) == 3  # 4, 3, 2
    assert len(tier2) == 1  # 1


def test_precision_rerank_penalizes_high_volatility_and_drawdown(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(predict_cmd, "_load_recommend_scoring_cfg", lambda: {})
    monkeypatch.setattr(predict_cmd, "_load_current_tactic_precisions", lambda: {})

    safer = {
        "code": "A",
        "rise_prob": 0.66,
        "global_rank_pct": 0.06,
        "momentum": 1.1,
        "confidence": "中",
        "tactic_resonance": 1,
        "tactic_tags": {"价值": True},
        "drawdown": -0.10,
        "last": {"close": 10.0, "atr14": 0.3},
    }
    riskier = {
        "code": "B",
        "rise_prob": 0.66,
        "global_rank_pct": 0.06,
        "momentum": 1.1,
        "confidence": "中",
        "tactic_resonance": 1,
        "tactic_tags": {"价值": True},
        "drawdown": -0.50,
        "last": {"close": 10.0, "atr14": 1.2},
    }
    ranked = predict_cmd._apply_precision_rerank([riskier, safer])
    assert ranked[0]["code"] == "A"
    assert ranked[0]["recommend_score"] > ranked[1]["recommend_score"]


def test_precision_rerank_uses_tactic_precision_bonus(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(predict_cmd, "_load_recommend_scoring_cfg", lambda: {})
    monkeypatch.setattr(
        predict_cmd,
        "_load_current_tactic_precisions",
        lambda: {"价值": 0.70, "成长": 0.40},
    )

    value_pick = {
        "code": "V",
        "rise_prob": 0.60,
        "global_rank_pct": 0.10,
        "momentum": 1.0,
        "confidence": "高",
        "tactic_resonance": 1,
        "tactic_tags": {"价值": True},
        "drawdown": -0.12,
        "last": {"close": 10.0, "atr14": 0.25},
    }
    growth_pick = {
        "code": "G",
        "rise_prob": 0.60,
        "global_rank_pct": 0.10,
        "momentum": 1.0,
        "confidence": "高",
        "tactic_resonance": 1,
        "tactic_tags": {"成长": True},
        "drawdown": -0.12,
        "last": {"close": 10.0, "atr14": 0.25},
    }

    ranked = predict_cmd._apply_precision_rerank([growth_pick, value_pick])
    assert ranked[0]["code"] == "V"


def test_precision_gate_weak_mode_demotes_low_quality_buy(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 20,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.50,
            "min_rise_prob_weak": 0.58,
            "min_resonance_weak": 1,
            "protect_top_buy": 1,
            "max_demote_ratio": 1.0,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.48, "samples_7d": 20, "samples_30d": 80},
    )

    rows = [
        {"code": "A", "signal": "买入", "recommend_score": 0.62, "rise_prob": 0.64, "tactic_resonance": 2},
        {"code": "B", "signal": "买入", "recommend_score": 0.42, "rise_prob": 0.56, "tactic_resonance": 0},
    ]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert gated[1]["signal"] == "观望"
    assert summary["applied"] is True
    assert summary["demoted"] == 1
    assert summary["mode"] == "weak"


def test_precision_gate_skips_when_samples_insufficient(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {"enabled": True, "min_samples_7d": 20, "min_samples_30d": 100},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.30, "acc_30d": 0.30, "samples_7d": 5, "samples_30d": 10},
    )
    rows = [{"code": "A", "signal": "买入", "recommend_score": 0.20, "rise_prob": 0.40, "tactic_resonance": 0}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["applied"] is False


def test_precision_gate_demotes_by_prob_bin_reliability(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": True,
            "prob_bin_min_samples": 10,
            "prob_bin_min_hit_rate_weak": 0.48,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_prob_bin_precision_stats",
        lambda: {"[0.55,0.60)": {"hit_rate": 0.40, "samples": 25}},
    )

    rows = [{"code": "B1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.58, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_bin"] == 1


def test_precision_gate_demotes_by_confidence_reliability(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": True,
            "confidence_min_samples": 10,
            "confidence_min_hit_rate_weak": {"高": 0.52, "中": 0.48, "低": 0.44},
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_confidence_precision_stats",
        lambda: {"高": {"hit_rate": 0.40, "samples": 25}, "中": {"hit_rate": 0.55, "samples": 25}, "低": {"hit_rate": 0.55, "samples": 25}},
    )

    rows = [{"code": "C1", "signal": "买入", "confidence": "高", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_confidence"] == 1


def test_precision_gate_demotes_by_tactic_reliability(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": True,
            "tactic_min_samples": 20,
            "tactic_min_precision_weak": 0.50,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_current_tactic_stats",
        lambda: {"成长": {"precision": 0.42, "samples": 50}},
    )

    rows = [{
        "code": "T1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66,
        "tactic_resonance": 1, "tactic_tags": {"成长": True},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_tactic"] == 1


def test_precision_gate_keeps_buy_when_tactic_reliability_healthy(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": True,
            "tactic_min_samples": 20,
            "tactic_min_precision_weak": 0.50,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_current_tactic_stats",
        lambda: {"价值": {"precision": 0.62, "samples": 80}},
    )

    rows = [{
        "code": "T2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66,
        "tactic_resonance": 1, "tactic_tags": {"价值": True},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_tactic"] == 0


def test_precision_gate_demotes_by_symbol_memory(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": True,
            "symbol_memory_min_samples": 3,
            "symbol_memory_min_hit_rate_weak": 0.40,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_symbol_buy_stats",
        lambda _codes, _strategy, window_days=30: {"S1": {"hit_rate": 0.20, "samples": 5}},
    )

    rows = [{"code": "S1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_symbol"] == 1


def test_precision_gate_keeps_buy_when_symbol_memory_healthy(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": True,
            "symbol_memory_min_samples": 3,
            "symbol_memory_min_hit_rate_weak": 0.40,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_symbol_buy_stats",
        lambda _codes, _strategy, window_days=30: {"S2": {"hit_rate": 0.60, "samples": 5}},
    )

    rows = [{"code": "S2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_symbol"] == 0


def test_precision_gate_demotes_by_payoff_quality(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": True,
            "payoff_min_gain_weak": 2.0,
            "payoff_min_ratio_weak": 0.45,
            "payoff_min_risk_pct": 6.0,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "P1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "drawdown": -0.30, "dual_trade": {"short": {"gain_pct": 1.2}, "long": {"gain_pct": 1.8}},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_payoff"] == 1


def test_precision_gate_keeps_buy_when_payoff_quality_healthy(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": True,
            "payoff_min_gain_weak": 2.0,
            "payoff_min_ratio_weak": 0.45,
            "payoff_min_risk_pct": 6.0,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "P2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "drawdown": -0.08, "dual_trade": {"short": {"gain_pct": 3.0}, "long": {"gain_pct": 5.0}},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_payoff"] == 0


def test_precision_gate_demotes_by_technical_confirmation(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": True,
            "tech_min_confirmations_weak": 2,
            "tech_min_available_checks": 3,
            "tech_rsi_floor_weak": 50,
            "tech_vol_ratio_floor_weak": 1.0,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "K1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"rsi6": 48, "macd_hist": -0.1, "vol_ratio": 0.8, "close": 10.0, "ma20": 10.5},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_technical"] == 1


def test_precision_gate_keeps_buy_when_technical_confirmation_sufficient(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": True,
            "tech_min_confirmations_weak": 2,
            "tech_min_available_checks": 3,
            "tech_rsi_floor_weak": 50,
            "tech_vol_ratio_floor_weak": 1.0,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "K2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"rsi6": 55, "macd_hist": 0.2, "vol_ratio": 1.2, "close": 10.6, "ma20": 10.4},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_technical"] == 0


def test_precision_gate_demotes_by_symbol_risk_memory(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": True,
            "symbol_risk_min_samples": 3,
            "symbol_risk_dd_threshold": -0.10,
            "symbol_risk_max_rate_weak": 0.55,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_symbol_risk_stats",
        lambda _codes, window_days=45, severe_dd_threshold=-0.10: {"R1": {"severe_rate": 0.75, "samples": 4}},
    )

    rows = [{"code": "R1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_symbol_risk"] == 1


def test_precision_gate_keeps_buy_when_symbol_risk_memory_healthy(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": True,
            "symbol_risk_min_samples": 3,
            "symbol_risk_dd_threshold": -0.10,
            "symbol_risk_max_rate_weak": 0.55,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_symbol_risk_stats",
        lambda _codes, window_days=45, severe_dd_threshold=-0.10: {"R2": {"severe_rate": 0.20, "samples": 5}},
    )

    rows = [{"code": "R2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_symbol_risk"] == 0


def test_precision_gate_demotes_by_liquidity_quality(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": True,
            "liquidity_min_turnover_weak": 1.2,
            "liquidity_max_turnover_weak": 25.0,
            "liquidity_min_vol_ratio_weak": 0.9,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "L1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"turnover": 0.6, "vol_ratio": 0.7},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_liquidity"] == 1


def test_precision_gate_keeps_buy_when_liquidity_quality_healthy(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": True,
            "liquidity_min_turnover_weak": 1.2,
            "liquidity_max_turnover_weak": 25.0,
            "liquidity_min_vol_ratio_weak": 0.9,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "L2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"turnover": 3.2, "vol_ratio": 1.4},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_liquidity"] == 0


def test_precision_gate_demotes_by_trend_alignment(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": True,
            "trend_min_hits_weak": 2,
            "trend_min_available_checks": 3,
            "trend_close_buffer_weak": -0.01,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "TND1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 9.8, "ma5": 10.1, "ma10": 10.3, "ma20": 10.5, "ma60": 10.8},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_trend"] == 1


def test_precision_gate_keeps_buy_when_trend_alignment_good(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": True,
            "trend_min_hits_weak": 2,
            "trend_min_available_checks": 3,
            "trend_close_buffer_weak": -0.01,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "TND2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 10.9, "ma5": 10.6, "ma10": 10.4, "ma20": 10.1, "ma60": 9.8},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_trend"] == 0


def test_precision_gate_demotes_by_trend_extension(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": True,
            "max_close_ma20_dev_weak": 0.12,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "EXT1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 12.0, "ma20": 10.0},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_extension"] == 1


def test_precision_gate_keeps_buy_when_trend_extension_normal(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": True,
            "max_close_ma20_dev_weak": 0.12,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{
        "code": "EXT2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 10.8, "ma20": 10.0},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_extension"] == 0


def test_precision_gate_demotes_by_extreme_volatility(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": True,
            "atr_pct_max_weak": 0.09,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{
        "code": "V1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 10.0, "atr14": 1.5},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_volatility"] == 1


def test_precision_gate_keeps_buy_when_volatility_normal(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": True,
            "atr_pct_max_weak": 0.09,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{
        "code": "V2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 10.0, "atr14": 0.5},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_volatility"] == 0


def test_precision_gate_demotes_by_probability_edge(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_edge_guard": True,
            "edge_min_prob_weak": 0.01,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{"code": "E1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.605, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={"last_thresh_buy": 0.60})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_edge"] == 1


def test_precision_gate_keeps_buy_when_probability_edge_sufficient(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_edge_guard": True,
            "edge_min_prob_weak": 0.01,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{"code": "E2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.625, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={"last_thresh_buy": 0.60})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_edge"] == 0


def test_precision_gate_demotes_by_dual_target_consistency(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_edge_guard": False,
            "use_dual_target_guard": True,
            "dual_min_short_gain_weak": 1.5,
            "dual_min_long_gain_weak": 2.2,
            "dual_min_long_short_ratio_weak": 0.8,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{
        "code": "D1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "dual_trade": {"short": {"gain_pct": 1.2}, "long": {"gain_pct": 1.0}},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_dual_target"] == 1


def test_precision_gate_keeps_buy_when_dual_target_consistent(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_edge_guard": False,
            "use_dual_target_guard": True,
            "dual_min_short_gain_weak": 1.5,
            "dual_min_long_gain_weak": 2.2,
            "dual_min_long_short_ratio_weak": 0.8,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{
        "code": "D2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "dual_trade": {"short": {"gain_pct": 2.0}, "long": {"gain_pct": 3.0}},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_dual_target"] == 0


def test_precision_gate_demotes_by_atr_reward_ratio(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_edge_guard": False,
            "use_dual_target_guard": False,
            "use_atr_reward_guard": True,
            "min_gain_atr_ratio_weak": 1.05,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{
        "code": "AR1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 10.0, "atr14": 1.0},
        "dual_trade": {"short": {"gain_pct": 0.9}, "long": {"gain_pct": 0.8}},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "观望"
    assert summary["demoted"] == 1
    assert summary["demoted_by_atr_reward"] == 1


def test_precision_gate_keeps_buy_when_atr_reward_ratio_good(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_edge_guard": False,
            "use_dual_target_guard": False,
            "use_atr_reward_guard": True,
            "min_gain_atr_ratio_weak": 1.05,
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.50, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{
        "code": "AR2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.66, "tactic_resonance": 2,
        "last": {"close": 10.0, "atr14": 0.1},
        "dual_trade": {"short": {"gain_pct": 1.2}, "long": {"gain_pct": 1.6}},
    }]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={})
    assert gated[0]["signal"] == "买入"
    assert summary["demoted"] == 0
    assert summary["demoted_by_atr_reward"] == 0


def test_precision_gate_adaptive_soft_relaxes_threshold(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_dual_target_guard": False,
            "use_atr_reward_guard": False,
            "use_edge_guard": True,
            "edge_min_prob_weak": 0.010,
            "adaptive_tuning": {"enabled": True, "soft_min_factor": 0.8},
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.54, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{"code": "AS1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.609, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={"last_thresh_buy": 0.60})
    assert gated[0]["signal"] == "买入"
    assert summary["tuning_level"] == "soft"


def test_precision_gate_adaptive_strict_tightens_threshold(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_hard": 0.20,
            "min_rise_prob_hard": 0.55,
            "min_resonance_hard": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_dual_target_guard": False,
            "use_atr_reward_guard": False,
            "use_edge_guard": True,
            "edge_min_prob_hard": 0.015,
            "adaptive_tuning": {"enabled": True, "strict_min_factor": 1.2, "hard_strict_margin_30d": 0.05},
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.30, "acc_30d": 0.30, "samples_7d": 20, "samples_30d": 90},
    )

    rows = [{"code": "AT1", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.616, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={"last_thresh_buy": 0.60})
    assert gated[0]["signal"] == "观望"
    assert summary["tuning_level"] == "strict"


def test_precision_gate_adaptive_strength_reduces_with_low_samples(monkeypatch):
    from server import predict_cmd

    monkeypatch.setattr(
        predict_cmd,
        "_load_recommend_precision_gate_cfg",
        lambda: {
            "enabled": True,
            "min_samples_7d": 10,
            "min_samples_30d": 30,
            "weak_acc_7d": 0.55,
            "weak_acc_30d": 0.50,
            "hard_acc_30d": 0.40,
            "min_recommend_score_weak": 0.20,
            "min_rise_prob_weak": 0.55,
            "min_resonance_weak": 0,
            "protect_top_buy": 0,
            "max_demote_ratio": 1.0,
            "use_prob_bin_guard": False,
            "use_confidence_guard": False,
            "use_tactic_guard": False,
            "use_symbol_memory_guard": False,
            "use_payoff_guard": False,
            "use_technical_guard": False,
            "use_symbol_risk_guard": False,
            "use_liquidity_guard": False,
            "use_trend_guard": False,
            "use_extension_guard": False,
            "use_volatility_guard": False,
            "use_dual_target_guard": False,
            "use_atr_reward_guard": False,
            "use_edge_guard": True,
            "edge_min_prob_weak": 0.010,
            "adaptive_tuning": {
                "enabled": True,
                "sample_target_7d": 40,
                "sample_target_30d": 200,
                "soft_min_factor": 0.8,
            },
        },
    )
    monkeypatch.setattr(
        predict_cmd,
        "_load_recent_accuracy_context",
        lambda _s: {"acc_7d": 0.54, "acc_30d": 0.49, "samples_7d": 20, "samples_30d": 90},
    )
    rows = [{"code": "AS2", "signal": "买入", "recommend_score": 0.80, "rise_prob": 0.609, "tactic_resonance": 2}]
    gated, summary = predict_cmd._apply_precision_gate(rows, strategy={"last_thresh_buy": 0.60})
    assert gated[0]["signal"] == "买入"
    assert summary["tuning_level"] == "soft"
    assert 0.0 < summary["tuning_strength"] < 1.0


def test_precision_gate_line_contains_rate_and_top_contributors():
    from server.predict_cmd import _precision_gate_line

    line = _precision_gate_line({
        "applied": True,
        "mode": "weak",
        "tuning_level": "soft",
        "tuning_strength": 0.5,
        "acc_7d": 0.53,
        "acc_30d": 0.49,
        "buy_before": 10,
        "buy_after": 7,
        "demoted": 3,
        "demoted_by_bin": 2,
        "demoted_by_confidence": 1,
        "demoted_by_tactic": 0,
        "demoted_by_symbol": 0,
        "demoted_by_payoff": 0,
        "demoted_by_technical": 0,
        "demoted_by_symbol_risk": 0,
        "demoted_by_liquidity": 0,
        "demoted_by_trend": 0,
        "demoted_by_extension": 0,
        "demoted_by_volatility": 0,
        "demoted_by_edge": 0,
        "demoted_by_dual_target": 0,
        "demoted_by_atr_reward": 0,
    })
    assert "降级率" in line
    assert "10→7" in line
    assert "分桶×2" in line
