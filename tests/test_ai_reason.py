"""tests/test_ai_reason.py — P2 ai_reason 路由与缓存测试。

覆盖:
  - build_prompt 含必要字段(code / state / rise_prob / tactic_hits / 财务/技术行)
  - _parse_response schema 校验(valid / invalid verdict / invalid confidence / 非 JSON)
  - _cache_key 桶化一致性(同桶同 key,跨桶不同 key)
  - generate 三级降级:
      * 缓存命中直接返回
      * Qwen 成功 → 落缓存
      * Qwen 失败 → Claude 兜底 → 成功
      * 两者都失败 → None
      * 日封顶超限 → None
  - render_card_section 输出结构
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ── 共享 fixture ──────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    from server import ai_reason
    # 隔离缓存文件 + 重置 in-memory 状态
    monkeypatch.setattr(ai_reason, "CACHE_FILE", tmp_path / "cache.json")
    ai_reason._reset_state_for_tests()
    # 默认所有测试:config 返回有 qwen + anthropic api_key 的配置
    monkeypatch.setattr(
        ai_reason, "_load_cfg",
        lambda: {"qwen": {"api_key": "qw-fake", "model": "qwen3-8b"},
                 "anthropic": {"api_key": "cl-fake",
                               "model": "claude-haiku-4-5-20251001"}},
    )
    yield
    ai_reason._reset_state_for_tests()


def _make_ctx(**kw):
    from server.ai_reason import StockContext
    defaults = dict(
        code="600519", name="贵州茅台",
        rise_prob_raw=0.55, rise_prob_cal=0.32,
        market_state="range", acc_30d=0.30,
        tactic_hits=["价值", "龙头"],
        drawdown=-0.10,
        fin={"roe": 29.4, "gross_margin": 91.5, "debt_ratio": 18.2,
             "rev_growth": 12.5, "profit_growth": 15.1},
        tech={"rsi6": 58.2, "macd_hist": 0.12},
        news_summary="近期公告:Q3 净利同比 +15%",
        global_rank=2, scan_total=5193, sector="食品饮料",
    )
    defaults.update(kw)
    return StockContext(**defaults)


# ── build_prompt ─────────────────────────────────────

class TestBuildPrompt:
    def test_contains_required_fields(self):
        from server.ai_reason import build_prompt
        ctx = _make_ctx()
        p = build_prompt(ctx)
        assert "600519" in p
        assert "贵州茅台" in p
        assert "range" in p
        assert "rise_prob_raw" in p
        assert "价值" in p and "龙头" in p
        assert "ROE" in p and "29.4" in p

    def test_handles_missing_fin_data(self):
        from server.ai_reason import build_prompt
        ctx = _make_ctx(fin={})
        p = build_prompt(ctx)
        assert "财务数据暂缺" in p

    def test_empty_tactic_hits_shows_technical_driven(self):
        from server.ai_reason import build_prompt
        ctx = _make_ctx(tactic_hits=[])
        p = build_prompt(ctx)
        assert "纯技术驱动" in p

    def test_acc_30d_none_shows_dash(self):
        from server.ai_reason import build_prompt
        ctx = _make_ctx(acc_30d=None)
        p = build_prompt(ctx)
        # 近 30 日精准率位置显示占位
        assert "— " in p or "—)" in p


# ── _parse_response schema 校验 ──────────────────────

class TestParseResponse:
    def test_valid_json_parses(self):
        from server.ai_reason import _parse_response
        text = json.dumps({
            "bull_reasons": ["a", "b", "c"],
            "risks": ["x", "y"],
            "verdict": "稳健加仓",
            "confidence": "高",
        }, ensure_ascii=False)
        out = _parse_response(text)
        assert out is not None
        assert out["bull_reasons"] == ["a", "b", "c"]
        assert out["verdict"] == "稳健加仓"

    def test_json_embedded_in_prose(self):
        """LLM 经常在 JSON 前后多余文字,提取应容错。"""
        from server.ai_reason import _parse_response
        text = '好的,以下是分析:\n\n{"bull_reasons":["a","b","c"],"risks":["x","y"],"verdict":"观望","confidence":"中"}\n\n希望有帮助。'
        out = _parse_response(text)
        assert out is not None
        assert out["verdict"] == "观望"

    def test_invalid_verdict_rejected(self):
        from server.ai_reason import _parse_response
        text = json.dumps({
            "bull_reasons": ["a"], "risks": ["x"],
            "verdict": "强买",   # 非法
            "confidence": "高",
        }, ensure_ascii=False)
        assert _parse_response(text) is None

    def test_invalid_confidence_rejected(self):
        from server.ai_reason import _parse_response
        text = json.dumps({
            "bull_reasons": ["a"], "risks": ["x"],
            "verdict": "稳健加仓",
            "confidence": "极高",   # 非法
        }, ensure_ascii=False)
        assert _parse_response(text) is None

    def test_non_json_rejected(self):
        from server.ai_reason import _parse_response
        assert _parse_response("这完全不是 JSON 输出") is None
        assert _parse_response("") is None
        assert _parse_response(None) is None

    def test_truncates_long_reasons(self):
        """超过 40 字的理由被裁剪。"""
        from server.ai_reason import _parse_response
        long_str = "a" * 100
        text = json.dumps({
            "bull_reasons": [long_str, "b", "c"],
            "risks": ["x", "y"],
            "verdict": "稳健加仓", "confidence": "高",
        }, ensure_ascii=False)
        out = _parse_response(text)
        assert out is not None
        assert len(out["bull_reasons"][0]) == 40


# ── _cache_key 桶化 ──────────────────────────────────

class TestCacheKey:
    def test_same_prob_bucket_same_key(self):
        from server.ai_reason import _cache_key
        k1 = _cache_key(_make_ctx(rise_prob_cal=0.31))
        k2 = _cache_key(_make_ctx(rise_prob_cal=0.33))
        # 0.31 和 0.33 都 round 到 0.30(bucket=0.05) 或 0.35;取决于实现
        # 我们只校验"同桶一致"的契约:0.31 与 0.32 都在 [0.30, 0.35) 桶,key 应一致
        assert _cache_key(_make_ctx(rise_prob_cal=0.31)) == \
               _cache_key(_make_ctx(rise_prob_cal=0.32))

    def test_different_state_different_key(self):
        from server.ai_reason import _cache_key
        k1 = _cache_key(_make_ctx(market_state="bull"))
        k2 = _cache_key(_make_ctx(market_state="bear"))
        assert k1 != k2

    def test_different_code_different_key(self):
        from server.ai_reason import _cache_key
        k1 = _cache_key(_make_ctx(code="600519"))
        k2 = _cache_key(_make_ctx(code="000858"))
        assert k1 != k2

    def test_far_prob_different_key(self):
        """rise_prob_cal 差 > 0.05 应不同桶。"""
        from server.ai_reason import _cache_key
        k1 = _cache_key(_make_ctx(rise_prob_cal=0.10))
        k2 = _cache_key(_make_ctx(rise_prob_cal=0.50))
        assert k1 != k2


# ── generate 三级降级 ────────────────────────────────

class TestGenerateFallback:
    def test_cache_hit_returns_immediately(self, monkeypatch):
        """缓存命中时不调 LLM。"""
        from server import ai_reason
        ctx = _make_ctx()
        cached_data = {
            "bull_reasons": ["a", "b", "c"], "risks": ["x", "y"],
            "verdict": "稳健加仓", "confidence": "高",
        }
        ai_reason._cache_set(ai_reason._cache_key(ctx), cached_data)

        called = {"qwen": False, "claude": False}
        monkeypatch.setattr(ai_reason, "_call_qwen",
                            lambda p, c: called.update(qwen=True))
        monkeypatch.setattr(ai_reason, "_call_claude",
                            lambda p, c: called.update(claude=True))

        result = ai_reason.generate(ctx)
        assert result == cached_data
        assert not called["qwen"]
        assert not called["claude"]

    def test_qwen_success_caches_and_returns(self, monkeypatch):
        from server import ai_reason
        response = json.dumps({
            "bull_reasons": ["a", "b", "c"], "risks": ["x", "y"],
            "verdict": "少量试仓", "confidence": "中",
        }, ensure_ascii=False)
        monkeypatch.setattr(ai_reason, "_call_qwen", lambda p, c: response)
        # Claude 设置成会抛,确认没被调
        def _never(): raise RuntimeError("should not be called")
        monkeypatch.setattr(ai_reason, "_call_claude", lambda p, c: _never())

        ctx = _make_ctx()
        result = ai_reason.generate(ctx)
        assert result is not None
        assert result["verdict"] == "少量试仓"
        # 再次调用应命中缓存
        assert ai_reason.generate(ctx) == result

    def test_qwen_fail_claude_fallback(self, monkeypatch):
        from server import ai_reason
        monkeypatch.setattr(ai_reason, "_call_qwen", lambda p, c: None)
        claude_resp = json.dumps({
            "bull_reasons": ["a", "b", "c"], "risks": ["x", "y"],
            "verdict": "观望", "confidence": "中",
        }, ensure_ascii=False)
        monkeypatch.setattr(ai_reason, "_call_claude", lambda p, c: claude_resp)

        result = ai_reason.generate(_make_ctx())
        assert result is not None
        assert result["verdict"] == "观望"

    def test_both_fail_returns_none(self, monkeypatch):
        from server import ai_reason
        monkeypatch.setattr(ai_reason, "_call_qwen", lambda p, c: None)
        monkeypatch.setattr(ai_reason, "_call_claude", lambda p, c: None)
        assert ai_reason.generate(_make_ctx()) is None

    def test_qwen_returns_invalid_json_falls_to_claude(self, monkeypatch):
        """Qwen 返回不合法 JSON 时走 Claude。"""
        from server import ai_reason
        monkeypatch.setattr(ai_reason, "_call_qwen", lambda p, c: "抱歉,我不会分析股票")
        claude_resp = json.dumps({
            "bull_reasons": ["a", "b", "c"], "risks": ["x", "y"],
            "verdict": "规避", "confidence": "低",
        }, ensure_ascii=False)
        monkeypatch.setattr(ai_reason, "_call_claude", lambda p, c: claude_resp)

        result = ai_reason.generate(_make_ctx())
        assert result is not None
        assert result["verdict"] == "规避"

    def test_daily_limit_blocks_calls(self, monkeypatch):
        """超过 MAX_DAILY_CALLS 返回 None 不调用 LLM。"""
        from server import ai_reason
        monkeypatch.setattr(ai_reason, "MAX_DAILY_CALLS", 1)
        response = json.dumps({
            "bull_reasons": ["a", "b", "c"], "risks": ["x", "y"],
            "verdict": "稳健加仓", "confidence": "高",
        }, ensure_ascii=False)
        monkeypatch.setattr(ai_reason, "_call_qwen", lambda p, c: response)
        monkeypatch.setattr(ai_reason, "_call_claude", lambda p, c: None)

        # 第一次调用,消耗限额
        result1 = ai_reason.generate(_make_ctx(code="AAA"))
        assert result1 is not None

        # 第二次调用不同股(避免缓存),超限 → None
        called = {"qwen": False}
        monkeypatch.setattr(ai_reason, "_call_qwen",
                            lambda p, c: called.update(qwen=True) or response)
        result2 = ai_reason.generate(_make_ctx(code="BBB"))
        assert result2 is None
        assert not called["qwen"]  # 超限 → 根本不调

    def test_config_load_failure_returns_none(self, monkeypatch):
        from server import ai_reason

        def boom(): raise RuntimeError("no config")
        monkeypatch.setattr(ai_reason, "_load_cfg", boom)
        assert ai_reason.generate(_make_ctx()) is None


# ── render_card_section ─────────────────────────────

class TestRenderCard:
    def test_none_returns_empty_list(self):
        from server.ai_reason import render_card_section
        assert render_card_section(None) == []

    def test_valid_reason_returns_one_markdown_element(self):
        from server.ai_reason import render_card_section
        reason = {
            "bull_reasons": ["财务稳健", "行业龙头", "技术突破"],
            "risks":        ["估值偏高", "资金面承压"],
            "verdict":      "少量试仓",
            "confidence":   "中",
        }
        elements = render_card_section(reason)
        assert len(elements) == 1
        assert elements[0]["tag"] == "markdown"
        content = elements[0]["content"]
        assert "少量试仓" in content
        assert "财务稳健" in content
        assert "估值偏高" in content
        assert "✅ 看多" in content
        assert "⚠️ 风险" in content


# ── 缓存 TTL ────────────────────────────────────────

class TestCacheTTL:
    def test_expired_cache_ignored(self, monkeypatch):
        from server import ai_reason
        ctx = _make_ctx()
        # 手动塞过期缓存
        ai_reason._cache_set(ai_reason._cache_key(ctx), {"stale": "data"})
        import time as _time
        # 把 _cache 里的 ts 调回 TTL + 100 秒之前
        with ai_reason._cache_lock:
            cache = ai_reason._load_cache_unlocked()
            key = ai_reason._cache_key(ctx)
            cache[key]["ts"] = _time.time() - ai_reason.CACHE_TTL_SECONDS - 100

        # 现在缓存应失效,generate 会走 LLM
        response = json.dumps({
            "bull_reasons": ["a", "b", "c"], "risks": ["x", "y"],
            "verdict": "稳健加仓", "confidence": "高",
        }, ensure_ascii=False)
        monkeypatch.setattr(ai_reason, "_call_qwen", lambda p, c: response)
        result = ai_reason.generate(ctx)
        assert result is not None
        assert result["verdict"] == "稳健加仓"  # 新鲜数据,非 "stale"
