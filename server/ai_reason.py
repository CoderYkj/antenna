"""
ai_reason.py - P2 LLM 深度推荐理由生成。

职责:
  - 对 scan_bot / tactic / predict 推送的 Top-N 股票调 LLM 生成结构化理由
  - 输出 JSON: {bull_reasons[3], risks[2], verdict, confidence}
  - 三级降级: Qwen3 → Claude Haiku → None(调用方用 build_commentary 兜底)
  - 24h 缓存: 按 (code, market_state, prob_cal_bucket=0.05)
  - 日封顶: 默认 50 次硬保护,超限回退 None

成本目标: 0.5 元/月(预期缓存命中 ≥ 50%)

产物: 不落盘学习系统(learning/),缓存落 learning/ai_reason_cache.json 仅为省钱
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 常量 ───────────────────────────────────────────────
CACHE_FILE = Path("learning/ai_reason_cache.json")
CACHE_TTL_SECONDS = 24 * 3600
MAX_DAILY_CALLS = 50        # 硬封顶,超限返回 None
QWEN_TIMEOUT = 20           # 秒
CLAUDE_TIMEOUT = 20


# ── 上下文结构 ─────────────────────────────────────────

@dataclass(frozen=True)
class StockContext:
    """generate() 入参 schema。字段全部可选(缺省走 prompt 默认值)。"""
    code:           str
    name:           str       = ""
    rise_prob_raw:  float     = 0.0
    rise_prob_cal:  float     = 0.0
    market_state:   str       = "range"
    acc_30d:        float | None = None
    tactic_hits:    list[str] = field(default_factory=list)
    drawdown:       float     = 0.0
    fin:            dict      = field(default_factory=dict)
    tech:           dict      = field(default_factory=dict)
    news_summary:   str       = ""
    global_rank:    int       = 0
    scan_total:     int       = 0
    sector:         str       = ""


# ── 缓存(thread-safe + 进程内 + 落盘) ─────────────────

_cache_lock = threading.Lock()
_cache: dict | None = None
_daily_counter: dict = {"date": "", "count": 0}


def _load_cache_unlocked() -> dict:
    """已持 _cache_lock 时调用。"""
    global _cache
    if _cache is not None:
        return _cache
    if not CACHE_FILE.exists():
        _cache = {}
        return _cache
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            _cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        _cache = {}
    return _cache


def _persist_cache_unlocked() -> None:
    """已持 _cache_lock 时调用,原子写。"""
    if _cache is None:
        return
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_name(CACHE_FILE.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CACHE_FILE)
    except BaseException:
        tmp.unlink(missing_ok=True)


def _cache_key(stock: StockContext) -> str:
    """(code, market_state, prob_cal_bucket=0.05) → 字符串 key。"""
    prob_bucket = round(stock.rise_prob_cal / 0.05) * 0.05
    return f"{stock.code}|{stock.market_state}|{prob_bucket:.2f}"


def _cache_get(key: str) -> dict | None:
    with _cache_lock:
        cache = _load_cache_unlocked()
        entry = cache.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        return entry.get("data")


def _cache_set(key: str, data: dict) -> None:
    with _cache_lock:
        cache = _load_cache_unlocked()
        cache[key] = {"ts": time.time(), "data": data}
        _persist_cache_unlocked()


def _check_daily_limit() -> bool:
    """返回 True = 可调用 + 计数已 +1;False = 超限拒绝。"""
    today = datetime.now().strftime("%Y-%m-%d")
    with _cache_lock:
        if _daily_counter["date"] != today:
            _daily_counter["date"] = today
            _daily_counter["count"] = 0
        if _daily_counter["count"] >= MAX_DAILY_CALLS:
            return False
        _daily_counter["count"] += 1
        return True


# ── 配置加载 ───────────────────────────────────────────

def _load_cfg() -> dict:
    """读 config.yaml。为测试时可 monkeypatch 替换。"""
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── prompt 拼装 ────────────────────────────────────────

def build_prompt(stock: StockContext) -> str:
    """按 spec §4.2 拼装 prompt。目标 ≤ 1000 tokens。"""
    rank_str = f"第 {stock.global_rank}/{stock.scan_total}" if stock.scan_total else "—"
    acc_str = f"{stock.acc_30d:.1%}" if stock.acc_30d is not None else "—"
    tactic_str = "、".join(stock.tactic_hits) if stock.tactic_hits else "纯技术驱动"
    news = (stock.news_summary or "无最近资讯")[:300]

    fin = stock.fin or {}
    tech = stock.tech or {}

    fin_line = "、".join(
        f"{label}={fin[k]:.1f}%"
        for label, k in [("ROE", "roe"), ("毛利", "gross_margin"),
                         ("负债率", "debt_ratio"), ("营收增", "rev_growth"),
                         ("利润增", "profit_growth")]
        if fin.get(k) is not None
    ) or "财务数据暂缺"

    tech_line = (
        f"RSI6={tech.get('rsi6', 50):.1f}  "
        f"MACD hist={tech.get('macd_hist', 0):+.3f}  "
        f"距 52 周高 {stock.drawdown*100:+.1f}%"
    )

    return f"""你是 A 股分析师,基于下述信息判断买入机会,严格输出 JSON。

【股票】{stock.name or stock.code}({stock.code}) {stock.sector}
【市场状态】{stock.market_state}(近 30 日"买入"精准率 {acc_str})
【AI 评分】rise_prob_raw={stock.rise_prob_raw:.3f} → prob_cal={stock.rise_prob_cal:.3f}
  全市场排名 {rank_str}
【技术】{tech_line}
【财务】{fin_line}
【命中战法】{tactic_str}
【最近资讯】{news}

输出严格 JSON(不要多余文字):
{{
  "bull_reasons": ["...", "...", "..."],   # 3 条买入理由,每条 ≤ 30 字
  "risks":        ["...", "..."],           # 2 条风险,每条 ≤ 30 字
  "verdict":      "稳健加仓",               # 或: 少量试仓 / 观望 / 规避
  "confidence":   "高"                      # 或: 中 / 低
}}"""


# ── LLM 路由(复用 predict_cmd.py cmd_chat 模式) ────────

def _call_qwen(prompt: str, cfg: dict) -> str | None:
    """Qwen3 via DashScope OpenAI-compatible API。失败返回 None。"""
    qw_cfg = cfg.get("qwen") or {}
    api_key = os.environ.get("DASHSCOPE_API_KEY") or qw_cfg.get("api_key") or ""
    model = qw_cfg.get("model", "qwen3-8b")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
            stream=False,
            extra_body={"enable_thinking": False},
            timeout=QWEN_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[ai_reason] Qwen 调用失败: {e}")
        return None


def _call_claude(prompt: str, cfg: dict) -> str | None:
    """Claude Haiku 兜底。失败返回 None。"""
    ac_cfg = cfg.get("anthropic") or {}
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ac_cfg.get("api_key") or ""
    model = ac_cfg.get("model", "claude-haiku-4-5-20251001")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, timeout=CLAUDE_TIMEOUT)
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"[ai_reason] Claude 调用失败: {e}")
        return None


# ── JSON 解析(strict schema) ───────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_VALID_VERDICTS = {"稳健加仓", "少量试仓", "观望", "规避"}
_VALID_CONFIDENCE = {"高", "中", "低"}


def _parse_response(text: str) -> dict | None:
    """从 LLM 输出提取 JSON 并验证 schema。失败返回 None。"""
    if not text:
        return None
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    bull = obj.get("bull_reasons")
    risks = obj.get("risks")
    verdict = obj.get("verdict")
    confidence = obj.get("confidence")

    if not isinstance(bull, list) or not isinstance(risks, list):
        return None
    if verdict not in _VALID_VERDICTS or confidence not in _VALID_CONFIDENCE:
        return None

    # 规范化:每条 ≤ 40 字(放宽 spec 的 30 字硬限,避免频繁失败)
    bull = [str(x)[:40] for x in bull[:3]]
    risks = [str(x)[:40] for x in risks[:2]]

    return {
        "bull_reasons": bull,
        "risks":        risks,
        "verdict":      verdict,
        "confidence":   confidence,
    }


# ── 主入口 ─────────────────────────────────────────────

def generate(stock: StockContext) -> dict | None:
    """三级降级入口。返回 None 表示全失败,调用方应回退 build_commentary。

    流程:
      1. 缓存命中 → 直接返回
      2. 日封顶检查 → 超限 None
      3. Qwen3 → 失败则 Claude Haiku → 失败则 None
      4. 解析 JSON + schema 校验 → 失败继续下一级
      5. 成功 → 缓存落盘 24h
    """
    key = _cache_key(stock)
    if cached := _cache_get(key):
        return cached

    if not _check_daily_limit():
        logger.warning(f"[ai_reason] 日调用超限(>{MAX_DAILY_CALLS}),跳过")
        return None

    try:
        cfg = _load_cfg()
    except Exception as e:
        logger.warning(f"[ai_reason] config 加载失败: {e}")
        return None

    prompt = build_prompt(stock)

    # 三级降级:Qwen → Claude → None
    for caller in (_call_qwen, _call_claude):
        raw = caller(prompt, cfg)
        if not raw:
            continue
        parsed = _parse_response(raw)
        if parsed:
            _cache_set(key, parsed)
            return parsed

    return None


# ── 飞书卡片渲染 ───────────────────────────────────────

_VERDICT_EMOJI = {"稳健加仓": "🎯", "少量试仓": "🧭", "观望": "👀", "规避": "🚫"}


def render_card_section(reason: dict | None) -> list[dict]:
    """渲染为飞书 markdown elements 列表。reason=None 时返回空列表(不占卡片位)。"""
    if not reason:
        return []
    emoji = _VERDICT_EMOJI.get(reason.get("verdict", ""), "🎯")
    bulls = "\n".join(f"  · {b}" for b in reason.get("bull_reasons", []))
    risks = "\n".join(f"  · {r}" for r in reason.get("risks", []))

    md = (
        f"{emoji} **{reason.get('verdict', '—')}** ({reason.get('confidence', '—')})\n"
        f"**✅ 看多**\n{bulls}\n"
        f"**⚠️ 风险**\n{risks}"
    )
    return [{"tag": "markdown", "content": md}]


# ── 测试辅助 ───────────────────────────────────────────

def _reset_state_for_tests() -> None:
    """仅供测试:清空 in-memory 缓存与日计数。"""
    global _cache, _daily_counter
    with _cache_lock:
        _cache = None
        _daily_counter = {"date": "", "count": 0}
