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
