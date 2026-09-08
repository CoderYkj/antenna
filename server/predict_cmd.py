"""
predict_cmd.py - 飞书指令「预测 <代码>」和「行情 <代码>」的执行逻辑。
返回飞书 interactive 卡片 payload（dict）。
"""
import json
import logging
import threading
import yaml
from datetime import datetime, time as dtime, timedelta

log = logging.getLogger(__name__)

# A full recommendation scan is expensive and its completion is broadcast to
# every configured chat. Prevent duplicate scans from producing multiple cards.
_scan_lock = threading.Lock()


def _log_critical_error(tag: str, exc: Exception) -> None:
    """记录后台线程异常:既写常规日志(log.exception),也追加一条不受
    日志轮转/重启影响的持久化记录到 logs/critical_errors.jsonl,
    方便事后排查(含完整 traceback,而不只是 str(e) 摘要)。
    """
    import json as _json
    import os as _os
    import time as _time
    import traceback as _traceback

    log.exception(tag)
    try:
        _os.makedirs("logs", exist_ok=True)
        record = {
            "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag,
            "error": repr(exc),
            "traceback": _traceback.format_exc(),
        }
        with open("logs/critical_errors.jsonl", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 持久化失败不应影响原有的失败通知流程


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _next_trading_day(dt: datetime) -> str:
    """返回下一个交易日日期字符串（优先用 exchange_calendars，否则跳过周末）。"""
    try:
        import exchange_calendars as xcals
        import pandas as pd
        cal = xcals.get_calendar("XSHG")
        ts = pd.Timestamp(dt.date()) + pd.Timedelta(days=1)
        while not cal.is_session(ts):
            ts += pd.Timedelta(days=1)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        d = dt.date() + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        return d.strftime("%Y-%m-%d")


def _predict_date() -> tuple[str, bool]:
    """
    返回 (预测日期字符串, 是否为下一交易日)。
    交易时段结束（15:00）后或非交易日，预测目标切换到下一交易日。
    """
    now = datetime.now()
    market_close = dtime(15, 0)
    is_today_trading = True
    try:
        import exchange_calendars as xcals
        import pandas as pd
        cal = xcals.get_calendar("XSHG")
        is_today_trading = cal.is_session(pd.Timestamp(now.date()))
    except Exception:
        is_today_trading = now.weekday() < 5
    if is_today_trading and now.time() < market_close:
        return now.strftime("%Y-%m-%d"), False
    else:
        return _next_trading_day(now), True


def _escape_md(text: str) -> str:
    """转义飞书 Markdown 中的特殊字符（避免股票名如 *ST 触发斜体）。"""
    return text.replace("*", "＊").replace("_", "\\_")


def _stock_label(code: str, name: str = "") -> str:
    """返回 '名称（代码）' 标准格式。name 为空时尝试从 name_map 查找，再查不到降级为纯代码。"""
    if not name:
        try:
            from data.fetcher import _load_name_map
            name = _load_name_map().get(code, "")
        except Exception:
            name = ""
    safe = _escape_md(name) if name else ""
    return f"{safe}（{code}）" if safe else code


def _fmt_rank(pct: float, rank: int = 0, total: int = 0) -> str:
    """将 global_rank_pct 格式化，避免极小值显示为 0%。
    rank/total 可选，有值时对小排名直接显示序号。"""
    if rank and total and rank <= max(10, int(total * 0.005)):
        # 前 0.5% 或前10名内，直接显示序号
        return f"第{rank}/{total}"
    if pct <= 0:
        return "第1"
    if pct < 0.001:
        return f"前{pct * 100:.2f}%"
    if pct < 0.01:
        return f"前{pct * 100:.1f}%"
    return f"前{pct:.0%}"


def _load_cfg() -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 行情指令 ───────────────────────────────────────────────────────────────

def cmd_quote(code: str) -> dict:
    """
    行情 <代码>：实时价格 + 日内温度计 + 分时走势图，不跑模型，速度快。
    """
    from data.fetcher import fetch_realtime_prices, fetch_intraday_kline
    from features.analyser import text_intraday_kline

    rt = fetch_realtime_prices([code])
    info = rt.get(code)
    if not info:
        return f"「{_stock_label(code)}」暂无实时行情（非交易时段或代码有误）"

    raw_name = info.get("name", code)
    name     = _escape_md(raw_name)

    price  = info["price"]
    pct    = info.get("pct", 0)
    high   = info.get("high", price)
    low    = info.get("low", price)
    d_open = info.get("open", price)
    arrow  = "🔴 ▲" if pct >= 0 else "🟢 ▼"
    color  = "red" if pct >= 0 else "green"

    BAR = 20
    is_st   = "ST" in raw_name.upper()
    pct_lim = 20.0 if is_st else 10.0

    # 日内价格区间温度计（低~高，◆标记现价）
    if high > low:
        fill = max(0, min(BAR, round((price - low) / (high - low) * BAR)))
        day_bar = "█" * fill + "░" * (BAR - fill)
        range_line = f"`{low:.2f} ▕{day_bar}▏ {high:.2f}`  ◆{price}"
    else:
        range_line = f"现价 **{price}**"

    # 涨跌幅温度计（-pct_lim% ~ +pct_lim%，◆标记当前）
    pct_clamped = max(-pct_lim, min(pct_lim, pct))
    pct_pos = (pct_clamped + pct_lim) / (2 * pct_lim)
    mid_idx = BAR // 2
    cur_idx = max(0, min(BAR - 1, round(pct_pos * (BAR - 1))))

    pct_chars = ["░"] * BAR
    pct_chars[mid_idx] = "|"    # 0% 基准线
    pct_chars[cur_idx] = "◆"   # 当前涨跌幅位置
    if pct >= 0 and cur_idx > mid_idx:
        for i in range(mid_idx + 1, cur_idx):
            pct_chars[i] = "█"
    elif pct < 0 and cur_idx < mid_idx:
        for i in range(cur_idx + 1, mid_idx):
            pct_chars[i] = "█"
    pct_bar  = "".join(pct_chars)
    lim_str  = f"{pct_lim:.0f}%"
    pct_line = f"`-{lim_str} ▕{pct_bar}▏ +{lim_str}`  {abs(pct):.2f}%"

    price_info = {"price": price, "pct": pct, "open": d_open, "high": high, "low": low}

    # 分时走势图（有数据时完全替代 summary，无数据时降级到静态摘要）
    intraday_text = None
    try:
        df_intra = fetch_intraday_kline(code)
        if df_intra is not None and not df_intra.empty:
            intraday_text = text_intraday_kline(df_intra, code, raw_name, price_info)
    except Exception:
        pass

    if intraday_text:
        # 有分时数据：text_intraday_kline 已含标题/涨跌/日内/开高低，不再重复
        elements = [{"tag": "markdown", "content": intraday_text}]
    else:
        # 非交易时段或获取失败：仅显示静态摘要
        summary = (
            f"**{name}（{code}）**  {price}  {arrow} {abs(pct):.2f}%\n"
            f"涨跌  {pct_line}\n"
            f"日内  {range_line}\n"
            f"开 {d_open}  高 {high}  低 {low}"
        )
        elements = [{"tag": "markdown", "content": summary}]

    # 黑名单警告（如果该股在当前状态下被拉黑）
    try:
        from learning.blacklist import load_blacklist
        from learning.market_state import load_current_state
        _bl = load_blacklist()
        _cs = load_current_state().get("current", "range")
        if _bl.is_blocked(code, _cs):
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown",
                "content": (f"⚠️ **黑名单提示**　{_stock_label(code)} 在当前 {_cs} 市场下"
                            f"因近期连续错误预测已被系统拉黑，建议谨慎操作。")
            })
    except Exception:
        pass

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"{raw_name}（{code}）实时行情"},
            "template": color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 新闻指令 ───────────────────────────────────────────────────────────────

def cmd_news(code: str) -> dict:
    """
    新闻 <代码>：展示指定股票近期正面与负面消息（东财个股新闻，按情绪分类）。
    只保留明确提及股票名称或代码的条目，过滤无关的全市场通用资讯。
    """
    from data.fetcher import _load_name_map
    from features.analyser import _fetch_em_news_raw, _classify_news

    name_map   = _load_name_map()
    stock_name = name_map.get(code, code)
    safe_name  = _escape_md(stock_name)

    # 股票名称的关键匹配词：全名 + 去掉通用后缀的简称
    # 例：「游族网络」→ ["游族网络", "游族"]
    name_keywords = {stock_name}
    for suffix in ("股份", "集团", "网络", "科技", "电子", "医药", "能源",
                   "银行", "证券", "保险", "地产", "控股", "实业", "化工"):
        if stock_name.endswith(suffix) and len(stock_name) > len(suffix) + 1:
            name_keywords.add(stock_name[: -len(suffix)])
            break
    name_keywords.add(code)   # 6 位数字代码也算明确提及

    def _is_relevant(title: str, snippet: str) -> bool:
        """标题或摘要中明确出现股票名/代码，视为相关新闻。"""
        text = title + snippet
        return any(kw in text for kw in name_keywords)

    # 拉取最新新闻（取 20 条保证覆盖）
    try:
        raw_items  = _fetch_em_news_raw(code, page_size=20)
        classified = _classify_news(raw_items)
    except Exception as e:
        return f"「{_stock_label(code)}」新闻获取失败：{e}"

    # 只保留与该股直接相关的条目，按日期降序（最新在前）
    relevant = [(t, s, ip, ig, d, u) for t, s, ip, ig, d, u in classified if _is_relevant(t, s)]
    relevant.sort(key=lambda x: x[4] or "", reverse=True)

    total_raw = len(classified)
    total_rel = len(relevant)

    if not relevant:
        return (
            f"「{safe_name}（{code}）」暂无直接相关新闻。\n"
            f"（共拉取 {total_raw} 条，均为通用市场资讯）"
        )

    pos_items = [(t, d, u) for t, s, ip, ig, d, u in relevant if ip and not ig]
    neg_items = [(t, d, u) for t, s, ip, ig, d, u in relevant if ig]   # 含混合情绪
    neu_items = [(t, d, u) for t, s, ip, ig, d, u in relevant if not ip and not ig]

    # 去重：Jaccard 相似度 ≥ 70% 视为同一条
    from features.analyser import _dedup_news
    pos_items = _dedup_news(pos_items)
    neg_items = _dedup_news(neg_items)
    neu_items = _dedup_news(neu_items)

    _this_year = str(datetime.now().year)

    import re as _re
    _news_nm = name_map  # 复用已加载的 name_map

    def _replace_codes_in_title(text: str) -> str:
        """把标题里的 6 位数字代码替换为 名称（代码） 格式。跳过 (XXXXXX.SH/SZ) 格式。"""
        def _sub(m):
            c = m.group(0)
            n = _news_nm.get(c)
            return f"{n}（{c}）" if n else c
        return _re.sub(r'\b\d{6}\b(?!\.\s*[A-Z]{2})', _sub, text)

    def _fmt_item(title: str, date_str: str, url: str = "") -> str:
        clean = _escape_md(_replace_codes_in_title(title))
        prefix = f"[{date_str}]  " if date_str else ""
        if url:
            return f"· {prefix}[{clean}]({url})"
        return f"· {prefix}{clean}"

    lines = [f"**{safe_name}（{code}）近期消息**　相关 {total_rel} 条 / 共 {total_raw} 条"]

    if pos_items:
        lines.append(f"\n**📰 正面消息**（{len(pos_items)} 条）")
        for title, date_str, url in pos_items[:10]:
            lines.append(_fmt_item(title, date_str, url))

    if neg_items:
        lines.append(f"\n**🔴 负面消息**（{len(neg_items)} 条）")
        for title, date_str, url in neg_items[:10]:
            lines.append(_fmt_item(title, date_str, url))

    if neu_items:
        lines.append(f"\n**📋 中性资讯**（{len(neu_items)} 条）")
        for title, date_str, url in neu_items[:10]:
            lines.append(_fmt_item(title, date_str, url))

    try:
        from learning.market_state import load_current_state as _lcs_news
        lines.append(f"\n📍 当前市场状态：**{_lcs_news().get('current', 'range')}**")
    except Exception:
        pass

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"{stock_name}（{code}）近期消息"},
            "template": "blue",
        },
        "elements": [{"tag": "markdown", "content": "\n".join(lines)}],
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}



def cmd_predict(code: str) -> dict:
    """
    预测 <代码>：完整预测流程，AI 信号 + 技术分析 + 价格区间预测。
    """
    from data.fetcher import fetch_stock_hist, fetch_realtime_prices, fetch_intraday_kline
    from features.builder import build_features
    from features.technical import get_active_feature_cols
    from features.analyser import analyse, predict_range, build_commentary, suggest_holding, suggest_trade_levels, text_intraday_kline
    from models.predictor import load_model, predict
    from learning.optimizer import load_strategy

    cfg      = _load_cfg()
    strategy = load_strategy()
    buy_top_pct = strategy.get("buy_top_pct", 0.10)

    # 拉取历史数据 & 计算特征
    try:
        df = fetch_stock_hist(code, days=365)
    except Exception as e:
        return f"「{_stock_label(code)}」数据获取失败：{e}"

    # 拉取单股 alt_data（失败降级为空 dict，不崩溃）
    today_str = _today()
    alt_dict: dict = {}
    try:
        from data.alt_fetcher import fetch_alt_features
        alt_cache = fetch_alt_features([code], today_str)
        alt_dict = alt_cache.get(code, {})
    except Exception as _alt_err:
        log.warning(f"[cmd_predict] alt_data fetch failed for {code}, proceeding without alt features: {_alt_err}")

    try:
        df = build_features(df, alt=alt_dict)
    except Exception as e:
        return f"「{_stock_label(code)}」特征计算失败：{e}"

    # 模型预测
    try:
        result = predict(df, get_active_feature_cols(),
                         model_dir=cfg["model"]["saved_dir"],
                         buy_top_pct=buy_top_pct)
    except Exception as e:
        return f"「{_stock_label(code)}」模型预测失败：{e}"

    rise_prob  = result["rise_prob"]
    signal     = result["signal"]
    confidence = result["confidence"]
    self_rank_pct = result.get("self_rank_pct", 0.5)

    # 最后一行技术指标
    last_row = df.iloc[-1].to_dict()

    # 技术分析文字
    tech_text = analyse(last_row)

    # 实时行情（补充现价、高低开盘）
    price_info = predict_range(df, rise_prob)
    try:
        rt = fetch_realtime_prices([code])
        info = rt.get(code, {})
        raw_name = info.get("name", code)
        if info:
            price_info.update({k: info[k] for k in ("price", "pct", "high", "low", "open") if k in info})
    except Exception:
        raw_name = code
        info = {}

    name = _escape_md(raw_name)

    # 点评 & 依据
    commentary, basis = build_commentary(df, result, price_info)

    # 持仓建议
    holding_label, holding_reason = suggest_holding(last_row, rise_prob)

    # 买卖建议
    trade = suggest_trade_levels(last_row, price_info, rise_prob)

    # 预测日期
    pred_date, is_next = _predict_date()
    date_hint = f"预测目标：{pred_date}（{'下一交易日' if is_next else '今日'}）"

    # ── 信号判定（三层优先级）────────────────────────────────
    # 1. 今日推荐股（Top-N）→ 锁定"买入"，全天有效，同时展示实时 rise_prob 变化
    # 2. 今日有扫描 & 非推荐股 → 用全市场门槛判断实时 rise_prob
    # 3. 今日无扫描 → 退回自身 60 日排名信号
    from learning.tracker import load_predictions
    snap_preds = {p["code"]: p for p in load_predictions(pred_date)}
    stored     = snap_preds.get(code)

    scan_rise_prob  = None   # 扫描时的涨概率（用于展示 delta）
    global_rank     = 0
    global_rank_pct = None
    scan_total      = 0

    thresh_buy_saved   = strategy.get("last_thresh_buy", 0)
    thresh_watch_saved = strategy.get("last_thresh_watch", 0)
    scan_date_saved    = strategy.get("last_scan_date", "")
    scan_total         = strategy.get("last_scan_total", 0)

    if stored and stored.get("recommended"):
        # 今日推荐股：默认锁定扫描信号，但实时 rise_prob 大幅下滑时降级
        scan_rise_prob  = stored.get("rise_prob", rise_prob)
        global_rank     = stored.get("global_rank", 0)
        global_rank_pct = stored.get("global_rank_pct")
        scan_total      = stored.get("scan_total", scan_total)

        drop = scan_rise_prob - rise_prob   # 下滑幅度
        if drop > 0.08 or (thresh_watch_saved and rise_prob < thresh_watch_saved):
            # 实时涨概率大幅回落（>8%）或跌破观望门槛 → 降级为观望，提示复核
            signal, confidence = "观望", "中"
            rank_source = "今日推荐⚠️已回落"
        else:
            signal     = stored["signal"]
            confidence = stored.get("confidence", confidence)
            rank_source = "今日推荐"
    else:
        scan_rise_prob = None
        global_rank    = 0
        global_rank_pct = None

        if thresh_buy_saved and scan_date_saved == _today():
            # 今日有扫描，用全市场门槛判断实时值
            if rise_prob >= thresh_buy_saved:
                signal, confidence = "买入", "高"
            elif thresh_watch_saved and rise_prob >= thresh_watch_saved:
                signal, confidence = "观望", "中"
            else:
                signal, confidence = "回避", "低"
            rank_source = f"全市场{scan_total}只"
        else:
            # 无扫描记录，退回自身 60 日
            rank_source = "自身60日"

    # 信号颜色
    color_map = {"买入": "red", "观望": "yellow", "回避": "grey"}
    color = color_map.get(signal, "blue")

    # 信号行
    if global_rank_pct is not None and scan_total:
        rank_label = _fmt_rank(global_rank_pct, global_rank, scan_total)
        rank_str   = f"全市场{rank_label}"
    else:
        rank_label = _fmt_rank(self_rank_pct)
        rank_str   = f"{rank_source} {rank_label}"

    # 实时 rise_prob 与扫描时的 delta（仅推荐股显示）
    prob_delta_str = ""
    if scan_rise_prob is not None:
        delta = rise_prob - scan_rise_prob
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
        prob_delta_str = f"　{arrow}扫描 {scan_rise_prob:.1%}"

    signal_line = (
        f"**{signal}**　涨概率 **{rise_prob:.1%}**{prob_delta_str}"
        f"　{rank_str}　置信度 **{confidence}**"
    )

    # 技术分析行
    tech_line = f"技术：{tech_text}"

    # 预测区间行
    pred_high = price_info.get("pred_high")
    pred_low  = price_info.get("pred_low")
    range_line = ""
    if pred_high and pred_low:
        range_line = f"预测区间：{pred_low} ～ {pred_high}"

    # 买卖建议行
    trade_lines = []
    if trade:
        trade_lines.append(
            f"买入参考 **{trade.get('buy_price')}**（{trade.get('buy_desc', '')}）　"
            f"止盈 **{trade.get('sell_price')}**（{trade.get('sell_desc', '')}）　"
            f"止损 **{trade.get('stop_price')}**（{trade.get('stop_desc', '')}）"
        )
        rr = trade.get("rr_ratio", 0)
        trade_lines.append(f"风险收益比 **{rr}:1**")

    # 持仓建议行
    holding_line = f"持仓周期：**{holding_label}**　{holding_reason}"

    # 分时图（可选，失败不影响卡片）
    intraday_section = None
    try:
        df_intra = fetch_intraday_kline(code)
        if df_intra is not None and not df_intra.empty:
            intraday_text = text_intraday_kline(df_intra, code, raw_name, price_info)
            intraday_section = intraday_text
    except Exception:
        pass

    # 组装 elements
    elements = []

    # 主信号区
    main_content = "\n".join(filter(None, [
        signal_line,
        date_hint,
        tech_line,
        range_line,
    ]))
    elements.append({"tag": "markdown", "content": main_content})
    elements.append({"tag": "hr"})

    # 点评与依据
    elements.append({"tag": "markdown", "content": f"**点评**\n{commentary}\n\n**依据**\n{basis}"})

    # 分时图
    if intraday_section:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": intraday_section})

    # 买卖建议
    if trade_lines:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**买卖建议**\n" + "\n".join(trade_lines)})

    # 学习上下文
    _lc = _learning_context_line()
    if _lc:
        elements.append({"tag": "markdown", "content": _lc})

    # 持仓建议
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": holding_line})

    # 历史回测依据
    try:
        from learning.backtest_history import format_backtest_context_md
        bt_md = format_backtest_context_md()
        if bt_md:
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown", "content": bt_md})
    except Exception:
        pass

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"{raw_name}（{code}）预测分析"},
            "template": color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 复盘指令 ───────────────────────────────────────────────────────────────

def cmd_review(date_str: str = None) -> dict:
    """
    复盘 <日期>：查看历史预测命中情况及策略自适应调整记录。
    只统计「买入」信号的精准率，「观望/回避」不计入。
    """
    from learning.tracker import (
        load_predictions, log_outcomes, list_prediction_dates, load_outcomes
    )
    from learning.optimizer import build_review_report, build_guardrail_trace_text, RISE_THRESHOLD
    from data.fetcher import fetch_realtime_prices, fetch_stock_hist

    # 确定复盘日期
    all_dates = list_prediction_dates()
    if not all_dates:
        return "暂无预测记录，请先运行「推荐」指令生成预测快照。"

    if date_str:
        # 容错：去掉前后空格
        date_str = date_str.strip()
        if date_str not in all_dates:
            return f"找不到 {date_str} 的预测记录，可用日期：{', '.join(all_dates[-5:])}"
    else:
        date_str = all_dates[-1]

    preds = load_predictions(date_str)
    if not preds:
        return f"{date_str} 无预测记录。"

    # 从 preds 取扫描日期（scan_date 字段，兼容旧数据）
    scan_date = preds[0].get("scan_date", "")
    if scan_date:
        scan_hint = f"扫描日 {scan_date} → 预测日 {date_str}"
    else:
        scan_hint = f"预测日 {date_str}"

    # 补全缺失 outcome
    existing_outcomes = load_outcomes(date_str)
    missing_codes = [p["code"] for p in preds if p["code"] not in existing_outcomes]

    if missing_codes:
        # 先尝试实时行情
        try:
            rt = fetch_realtime_prices(missing_codes)
        except Exception:
            rt = {}

        new_outcomes = {}
        for code in missing_codes:
            info = rt.get(code)
            if info and info.get("price"):
                op = info.get("open", info["price"])
                cl = info["price"]
                new_outcomes[code] = {
                    "actual_open":  op,
                    "actual_close": cl,
                    "actual_high":  info.get("high", cl),
                    "actual_low":   info.get("low", cl),
                    "actual_pct":   round((cl / op - 1) * 100, 2) if op else None,  # op=0 表示停牌/缺数据
                }
            else:
                # 回退到缓存日线数据
                try:
                    import pandas as pd
                    df_hist = fetch_stock_hist(code, days=30, cache_only=True)
                    df_hist["date"] = pd.to_datetime(df_hist["date"])
                    row = df_hist[df_hist["date"].dt.strftime("%Y-%m-%d") == date_str]
                    if not row.empty:
                        rx = row.iloc[0]
                        op = float(rx["open"])
                        cl = float(rx["close"])
                        new_outcomes[code] = {
                            "actual_open":  op,
                            "actual_close": cl,
                            "actual_high":  float(rx["high"]),
                            "actual_low":   float(rx["low"]),
                            "actual_pct":   round((cl / op - 1) * 100, 2) if op else None,
                        }
                except Exception:
                    pass

        if new_outcomes:
            merged = {**existing_outcomes, **new_outcomes}
            log_outcomes(date_str, merged)

    # 生成复盘报告
    report = build_review_report(date_str)
    day_result  = report.get("day_result")
    strategy    = report.get("strategy", {})
    guardrail_trace = report.get("guardrail_trace", {})
    guardrail_trace_text = report.get("guardrail_trace_text", "")
    guardrail_summary = report.get("guardrail_summary", {})
    change_desc = report.get("change_desc", "")
    acc_7d      = report.get("acc_7d", 0)
    acc_30d     = report.get("acc_30d", 0)
    samples_7   = report.get("samples_7", 0)
    samples_30  = report.get("samples_30", 0)

    buy_top_pct = strategy.get("buy_top_pct", 0.10)

    guardrail_line = guardrail_trace_text or build_guardrail_trace_text(guardrail_trace)
    guardrail_line = guardrail_line.replace("，", "　")
    summary_line = ""
    if guardrail_summary.get("total_days", 0) > 0:
        top_items = guardrail_summary.get("top_reasons") or []
        top_str = "、".join(f"{x['label']}×{x['count']}" for x in top_items[:3]) if top_items else "无"
        summary_line = (
            f"30日护栏：触发 **{guardrail_summary['trigger_days']} / {guardrail_summary['total_days']}**"
            f"（{guardrail_summary['trigger_rate']:.0%}）　主要原因 {top_str}"
        )

    # ── 构建卡片 elements ──────────────────────────────────
    elements = []

    # 规则说明
    rule_text = (
        f"**复盘规则**：买入命中=涨幅≥{RISE_THRESHOLD}%，观望/回避命中=涨幅<{RISE_THRESHOLD}%（正确回避），全部计入准确率。"
    )
    elements.append({"tag": "markdown", "content": rule_text})
    elements.append({"tag": "hr"})

    # 加载名称映射（复盘快照可能缺 name 字段）
    try:
        from data.fetcher import _load_name_map
        _review_name_map = _load_name_map()
    except Exception:
        _review_name_map = {}

    def _rname(d: dict) -> str:
        code = d["code"]
        name = d.get("name", "")
        # 快照回退值是 code 本身，需额外判断
        if name and name != code:
            return _escape_md(name)
        return _escape_md(_review_name_map.get(code, code))

    # 逐股明细：推荐股 / 自选股 分开统计
    BAR = 20

    def _acc_bar_line(hits: int, total: int, label: str) -> str:
        if total == 0:
            return f"{label}：待结算"
        acc = hits / total
        fill = max(0, min(BAR, round(acc * BAR)))
        bar = "█" * fill + "░" * (BAR - fill)
        icon = "✅" if acc >= 0.85 else ("⚠️" if acc >= 0.70 else "❌")
        return f"{label} {icon}  `{bar}`  **{acc:.0%}**（{hits}/{total} 命中）"

    if day_result and day_result.get("details"):
        details = day_result["details"]

        # 分离：自选股 vs 推荐股
        wl_details  = [d for d in details if d.get("is_watchlist")]
        wl_codes    = {d["code"] for d in wl_details}
        rec_details = [d for d in details if d["code"] not in wl_codes]

        # ── 推荐股区段 ──────────────────────────────────
        rec_buy   = [d for d in rec_details if d.get("is_buy")]
        rec_watch = [d for d in rec_details if not d.get("is_buy")]

        rec_lines = ["**📊 推荐股复盘**"]

        if rec_buy:
            rec_lines.append("买入信号（计入准确率）")
            for d in rec_buy:
                code  = d["code"]
                dname = _rname(d)
                prob  = d.get("rise_prob", 0)
                apct  = d.get("actual_pct")
                hit   = d.get("hit")
                hit_icon = "✅" if hit is True else ("❌" if hit is False else "⬜")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                rec_lines.append(
                    f"· {hit_icon} **{dname}**（{code}）  "
                    f"涨概率 {prob:.1%}  实际 {apct_str}"
                )

        if rec_watch:
            rec_lines.append("观望/回避（计入准确率）")
            for d in rec_watch:
                code  = d["code"]
                dname = _rname(d)
                prob  = d.get("rise_prob", 0)
                apct  = d.get("actual_pct")
                hit   = d.get("hit")
                hit_icon = "✅" if hit is True else ("❌" if hit is False else "⬜")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                rec_lines.append(
                    f"· {hit_icon} {dname}（{code}）  "
                    f"涨概率 {prob:.1%}  实际 {apct_str}"
                )

        rec_hits  = sum(1 for d in rec_details if d.get("hit") is True)
        rec_total = sum(1 for d in rec_details if d.get("hit") is not None)
        rec_lines.append("")
        rec_lines.append(_acc_bar_line(rec_hits, rec_total, "推荐股准确率"))

        elements.append({"tag": "markdown", "content": "\n".join(rec_lines)})

        # ── 自选股区段 ──────────────────────────────────
        if wl_details:
            elements.append({"tag": "hr"})
            wl_hits  = sum(1 for d in wl_details if d.get("hit") is True)
            wl_total = sum(1 for d in wl_details if d.get("hit") is not None)

            wl_lines = ["**⭐ 自选股复盘**"]
            for d in wl_details:
                code  = d["code"]
                dname = _rname(d)
                apct  = d.get("actual_pct")
                hit   = d.get("hit")
                sig   = d.get("signal", "观望")
                hit_icon = "✅" if hit is True else ("❌" if hit is False else "⬜")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                wl_lines.append(
                    f"· {hit_icon} **{dname}**（{code}）  "
                    f"信号「{sig}」  实际 {apct_str}"
                )

            wl_lines.append("")
            wl_lines.append(_acc_bar_line(wl_hits, wl_total, "自选股准确率"))
            elements.append({"tag": "markdown", "content": "\n".join(wl_lines)})

        elements.append({"tag": "hr"})

    # 精准率汇总（综合 推荐+自选 = day_result 原始统计）
    if day_result and day_result.get("total", 0) > 0:
        hits    = day_result["hits"]
        total   = day_result["total"]
        acc_day = day_result["accuracy"]
        acc_color = "✅" if acc_day >= 0.85 else ("⚠️" if acc_day >= 0.70 else "❌")
        acc_text = (
            f"综合准确率（推荐+自选）{acc_color} **{acc_day:.0%}**（{hits}/{total} 命中）\n"
            f"7日准确率：**{acc_7d:.0%}**（{samples_7}条样本）　"
            f"30日准确率：**{acc_30d:.0%}**（{samples_30}条样本）"
        )
    else:
        acc_text = f"当日无信号数据，7日准确率：**{acc_7d:.0%}**　30日准确率：**{acc_30d:.0%}**"

    elements.append({"tag": "markdown", "content": acc_text})
    elements.append({"tag": "hr"})

    # 策略自优化说明
    strategy_text = (
        f"**策略自优化** — 当前门槛：前 **{buy_top_pct:.0%}**\n"
        f"{change_desc}\n"
        f"{guardrail_line}"
        + (f"\n{summary_line}" if summary_line else "")
    )
    elements.append({"tag": "markdown", "content": strategy_text})

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"复盘报告　{scan_hint}"},
            "template": "blue",
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 推荐指令 ───────────────────────────────────────────────────────────────

def _enrich_tactic_scores(results: list, workers: int = 8) -> list:
    """
    对已筛选结果并行拉取财务数据，输出四维战法标签与共振度。
    不改变入参列表顺序，仅追加字段：tactic_tags / tactic_reasons / tactic_resonance / fin_res。

    P2 改造:阈值从 server/predict_cmd.py 硬编码 → 读 learning/tactic_params.json[当前 state]。
    文件缺失/损坏自动回退到 yaml.defaults(含 bear override)。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import fetch_financial_data
    from features.fundamental import analyse_financials

    # P2:加载当前 market_state 对应的战法阈值
    try:
        from learning import market_state, tactic_learner
        state = market_state.load_current_state().get("current", "range")
        params = tactic_learner.load_params(state)
    except Exception:
        params = {}

    # 各战法参数(回退到原硬编码值,保证 P2 关闭时行为不变)
    p_value  = params.get("value",  {}) or {}
    p_growth = params.get("growth", {}) or {}
    p_leader = params.get("leader", {}) or {}
    p_contra = params.get("contra", {}) or {}

    v_roe_min        = float(p_value.get("roe_min",         8.0))
    v_debt_max       = float(p_value.get("debt_ratio_max",  50.0))
    v_total_min      = float(p_value.get("total_score_min", 2.0))
    g_rev_min        = float(p_growth.get("rev_growth_min",    15.0))
    g_pft_min        = float(p_growth.get("profit_growth_min", 15.0))
    g_roe_min        = float(p_growth.get("roe_min",           12.0))
    l_roe_min        = float(p_leader.get("roe_min",          15.0))
    l_gross_min      = float(p_leader.get("gross_margin_min", 30.0))
    l_above_required = bool(p_leader.get("above_ma60_required", True))
    c_drawdown_max   = float(p_contra.get("drawdown_max",  -0.15))
    c_roe_min        = float(p_contra.get("roe_min",        3.0))
    c_debt_max       = float(p_contra.get("debt_ratio_max", 65.0))

    def _score_one(item):
        code = item["code"]
        try:
            fin_res = analyse_financials(fetch_financial_data(code))
        except Exception:
            fin_res = {}

        last   = item.get("last", {})
        def _f(k, d=0.0):
            v = last.get(k, d)
            return float(v) if v is not None and v == v else d

        close    = _f("close")
        ma60     = _f("ma60")
        above_60 = (close > ma60) if ma60 else False
        drawdown = item.get("drawdown", 0.0)

        roe     = fin_res.get("roe")
        gross_m = fin_res.get("gross_margin")
        debt_r  = fin_res.get("debt_ratio")
        rev_gr  = fin_res.get("rev_growth")
        pft_gr  = fin_res.get("profit_growth")
        total_s = fin_res.get("total_score", 0) or 0

        tags    = {}   # tactic_key → True
        reasons = {}   # tactic_key → 简短说明

        # ── 价值：低负债 + ROE健康 + 财务总分正向 ──────────────
        v = 0
        if roe and roe > v_roe_min:        v += 1
        if debt_r and debt_r < v_debt_max: v += 1
        if total_s >= v_total_min:         v += 1
        if v >= 2:
            tags["价值"] = True
            parts = []
            if roe     is not None: parts.append(f"ROE {roe:.1f}%")
            if debt_r  is not None: parts.append(f"负债率 {debt_r:.1f}%")
            reasons["价值"] = "　".join(parts) or "财务稳健"

        # ── 成长：营收/利润增速领先 ─────────────────────────────
        g = 0
        if rev_gr and rev_gr > g_rev_min: g += 1
        if pft_gr and pft_gr > g_pft_min: g += 1
        if roe and roe > g_roe_min:       g += 1
        if g >= 2:
            tags["成长"] = True
            parts = []
            if rev_gr is not None: parts.append(f"营收 {rev_gr:+.1f}%")
            if pft_gr is not None: parts.append(f"利润 {pft_gr:+.1f}%")
            reasons["成长"] = "　".join(parts) or "增速突出"

        # ── 龙头：高毛利 + ROE优秀 + 技术强势 ─────────────────
        l = 0
        if roe and roe > l_roe_min:           l += 1
        if gross_m and gross_m > l_gross_min: l += 1
        if (not l_above_required) or above_60: l += 1
        if l >= 2:
            tags["龙头"] = True
            parts = []
            if gross_m is not None: parts.append(f"毛利率 {gross_m:.1f}%")
            if roe     is not None: parts.append(f"ROE {roe:.1f}%")
            reasons["龙头"] = "　".join(parts) or "行业优势明显"

        # ── 逆向：技术超跌 + 基本面仍稳健 ─────────────────────
        c = 0
        if drawdown < c_drawdown_max:    c += 1
        if roe and roe > c_roe_min:      c += 1
        if debt_r and debt_r < c_debt_max: c += 1
        if c >= 2:
            tags["逆向"] = True
            reasons["逆向"] = f"距高点 {drawdown*100:.1f}%，基本面支撑"

        return {
            **item,
            "tactic_tags":      tags,
            "tactic_reasons":   reasons,
            "tactic_resonance": len(tags),
            "fin_res":          fin_res,
        }

    enriched_map = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_score_one, r) for r in results]):
            try:
                r = fut.result()
                enriched_map[r["code"]] = r
            except Exception:
                pass

    # 保持原始顺序，失败项保留原始数据
    return [enriched_map.get(r["code"], r) for r in results]


# ── P2 D3:共振股 rank_pct 加权(不动 rise_prob/prob_cal) ──

# 中文战法名 → yaml 英文 key
_TACTIC_CN_TO_EN = {"价值": "value", "成长": "growth", "龙头": "leader", "逆向": "contra"}


def _apply_resonance_boost(results: list) -> list:
    """spec §3.3 D3:对共振股(≥2 战法命中)向前提名次,不动 rise_prob/prob_cal。

    - 读当前 market_state 对应的战法权重
    - 命中 N 个战法,rank_pct -= sum(weights[hit]) × rank_boost_per_tactic
    - 总提升不超过 rank_boost_max(默认 6%)
    - 全部 rise_prob/rise_prob_raw/rise_prob_cal 字段保持原值
    - 任何异常 → 无操作返回原列表
    """
    if not results:
        return results
    try:
        from learning import market_state, tactic_learner
        state = market_state.load_current_state().get("current", "range")
        cfg = tactic_learner.load_config()
        if not cfg.weights.get("enable_resonance_rank_boost", True):
            return results
        weights = tactic_learner.load_resonance_weights(state)
        boost_per_tactic = float(cfg.weights.get("rank_boost_per_tactic", 0.02))
        boost_max        = float(cfg.weights.get("rank_boost_max",        0.06))
    except Exception:
        return results

    for r in results:
        tags = r.get("tactic_tags", {}) or {}
        hits = [t for t in tags if tags.get(t)]
        if len(hits) < 2:
            continue
        weighted = sum(weights.get(_TACTIC_CN_TO_EN.get(t, ""), 0.0) for t in hits)
        rank_delta = min(weighted * boost_per_tactic, boost_max)
        if rank_delta <= 0:
            continue
        old_rank_pct = float(r.get("global_rank_pct", 1.0))
        r["global_rank_pct"] = round(max(0.0, old_rank_pct - rank_delta), 4)
        r["resonance_rank_boost"] = round(rank_delta, 4)
        # rise_prob / rise_prob_raw / rise_prob_cal 一概不动

    return results


def _tier_split(enriched: list) -> tuple[list, list]:
    """按战法认可程度分层。
    tier1: tactic_resonance >= 2（共振）
    tier2: tactic_resonance == 1（单战法）
    0 战法命中的不返回。
    """
    tier1 = [r for r in enriched if r.get("tactic_resonance", 0) >= 2]
    tier2 = [r for r in enriched if r.get("tactic_resonance", 0) == 1]
    return tier1, tier2


def _load_recommend_scoring_cfg() -> dict:
    """读取推荐重排参数，缺省返回保守默认值。"""
    cfg = _load_cfg()
    scan_cfg = cfg.get("scan", {}) if isinstance(cfg, dict) else {}
    score_cfg = scan_cfg.get("recommend_scoring", {}) if isinstance(scan_cfg, dict) else {}
    return score_cfg if isinstance(score_cfg, dict) else {}


def _load_recommend_precision_gate_cfg() -> dict:
    """读取推荐提精闸门参数。"""
    cfg = _load_cfg()
    scan_cfg = cfg.get("scan", {}) if isinstance(cfg, dict) else {}
    gate_cfg = scan_cfg.get("recommend_precision_gate", {}) if isinstance(scan_cfg, dict) else {}
    return gate_cfg if isinstance(gate_cfg, dict) else {}


def _load_recent_accuracy_context(strategy: dict) -> dict:
    """获取近期精准率上下文，优先实时重算，失败回退 strategy 缓存值。"""
    try:
        from learning.optimizer import rolling_accuracy
        acc_7d, _, samples_7d = rolling_accuracy(7)
        acc_30d, _, samples_30d = rolling_accuracy(30)
        return {
            "acc_7d": float(acc_7d),
            "acc_30d": float(acc_30d),
            "samples_7d": int(samples_7d),
            "samples_30d": int(samples_30d),
        }
    except Exception as err:
        log.warning(f"[scan_bot] load rolling_accuracy failed, fallback strategy cache: {err}")
        return {
            "acc_7d": float(strategy.get("accuracy_7d", 0.0) or 0.0),
            "acc_30d": float(strategy.get("accuracy_30d", 0.0) or 0.0),
            "samples_7d": int((strategy.get("history") or [{}])[-1].get("samples_7", 0) or 0),
            "samples_30d": 0,
        }


def _prob_bin_label(rise_prob: float) -> str:
    p = float(rise_prob or 0.0)
    if p < 0.60:
        return "[0.55,0.60)"
    if p < 0.65:
        return "[0.60,0.65)"
    return "[0.65,1.00]"


def _load_prob_bin_precision_stats() -> dict:
    """
    读取 30d 概率分桶命中率与样本，用于提精闸门。
    返回: label -> {hit_rate, samples}
    """
    try:
        from learning.optimizer import build_monitor_dashboard_metrics

        metrics = build_monitor_dashboard_metrics()
        windows = metrics.get("windows", {}) if isinstance(metrics, dict) else {}
        w30 = windows.get("30d", {}) if isinstance(windows, dict) else {}
        rates = w30.get("prob_bin_hit_rate", {}) if isinstance(w30, dict) else {}
        samples = w30.get("prob_bin_samples", {}) if isinstance(w30, dict) else {}
        out = {}
        for label, rate in rates.items():
            out[str(label)] = {
                "hit_rate": float(rate or 0.0),
                "samples": int((samples or {}).get(label, 0) or 0),
            }
        return out
    except Exception as err:
        log.warning(f"[scan_bot] load prob-bin precision stats failed: {err}")
        return {}


def _normalize_confidence_label(confidence: str) -> str:
    c = str(confidence or "").strip().lower()
    if c in ("高", "high", "h"):
        return "高"
    if c in ("低", "low", "l"):
        return "低"
    return "中"


def _load_confidence_precision_stats() -> dict:
    """
    读取 30d 置信度分层命中率与样本，用于提精闸门。
    返回: 高/中/低 -> {hit_rate, samples}
    """
    try:
        from learning.optimizer import build_monitor_dashboard_metrics

        metrics = build_monitor_dashboard_metrics()
        windows = metrics.get("windows", {}) if isinstance(metrics, dict) else {}
        w30 = windows.get("30d", {}) if isinstance(windows, dict) else {}
        rates = w30.get("confidence_hit_rate", {}) if isinstance(w30, dict) else {}
        samples = w30.get("confidence_samples", {}) if isinstance(w30, dict) else {}
        out = {}
        for label in ("高", "中", "低"):
            out[label] = {
                "hit_rate": float((rates or {}).get(label, 0.0) or 0.0),
                "samples": int((samples or {}).get(label, 0) or 0),
            }
        return out
    except Exception as err:
        log.warning(f"[scan_bot] load confidence precision stats failed: {err}")
        return {}


def _confidence_threshold_map(cfg: dict, mode: str) -> dict:
    default = {
        "weak": {"高": 0.52, "中": 0.48, "低": 0.44},
        "hard": {"高": 0.55, "中": 0.50, "低": 0.46},
    }
    key = "confidence_min_hit_rate_hard" if mode == "hard" else "confidence_min_hit_rate_weak"
    raw = cfg.get(key)
    if isinstance(raw, dict):
        merged = dict(default["hard" if mode == "hard" else "weak"])
        for k in ("高", "中", "低"):
            if k in raw:
                merged[k] = float(raw[k])
        return merged
    return dict(default["hard" if mode == "hard" else "weak"])


def _resolve_adaptive_tuning_profile(
    cfg: dict,
    mode: str,
    summary: dict,
    *,
    weak_acc_7d: float,
    weak_acc_30d: float,
    hard_acc_30d: float,
) -> dict:
    """
    根据当前精度压力返回阈值调优档位：
      - soft: 接近阈值边界时，轻微放松，减少误杀
      - neutral: 默认
      - strict: 明显失压时，进一步收紧
    """
    base = {
        "level": "neutral",
        "min_factor": 1.0,
        "max_factor": 1.0,
        "count_delta": 0,
        "demote_ratio_factor": 1.0,
        "strength": 0.0,
    }
    tuning = cfg.get("adaptive_tuning", {}) if isinstance(cfg, dict) else {}
    if not isinstance(tuning, dict) or not bool(tuning.get("enabled", True)):
        return base

    acc7 = float(summary.get("acc_7d", 0.0) or 0.0)
    acc30 = float(summary.get("acc_30d", 0.0) or 0.0)
    s7 = int(summary.get("samples_7d", 0) or 0)
    s30 = int(summary.get("samples_30d", 0) or 0)
    target7 = max(1, int(tuning.get("sample_target_7d", 20)))
    target30 = max(1, int(tuning.get("sample_target_30d", 90)))
    strength = min(1.0, max(0.0, min(s7 / target7, s30 / target30)))

    def _blend_factor(raw: float) -> float:
        # 样本不足时向 1.0 回归，避免过度调参
        return 1.0 + (float(raw) - 1.0) * strength

    def _blend_count(raw: int) -> int:
        if raw == 0:
            return 0
        return int(round(int(raw) * strength))

    soft_margin_7d = float(tuning.get("weak_soft_margin_7d", 0.02))
    soft_margin_30d = float(tuning.get("weak_soft_margin_30d", 0.02))
    strict_margin_30d = float(tuning.get("hard_strict_margin_30d", 0.05))

    if mode == "weak" and acc7 >= weak_acc_7d - soft_margin_7d and acc30 >= weak_acc_30d - soft_margin_30d:
        return {
            "level": "soft",
            "min_factor": _blend_factor(float(tuning.get("soft_min_factor", 0.90))),
            "max_factor": _blend_factor(float(tuning.get("soft_max_factor", 1.10))),
            "count_delta": _blend_count(int(tuning.get("soft_count_delta", -1))),
            "demote_ratio_factor": _blend_factor(float(tuning.get("soft_demote_ratio_factor", 0.85))),
            "strength": round(strength, 3),
        }
    if mode == "hard" and acc30 <= hard_acc_30d - strict_margin_30d:
        return {
            "level": "strict",
            "min_factor": _blend_factor(float(tuning.get("strict_min_factor", 1.10))),
            "max_factor": _blend_factor(float(tuning.get("strict_max_factor", 0.90))),
            "count_delta": _blend_count(int(tuning.get("strict_count_delta", 1))),
            "demote_ratio_factor": _blend_factor(float(tuning.get("strict_demote_ratio_factor", 1.10))),
            "strength": round(strength, 3),
        }
    return base


def _load_recent_symbol_buy_stats(codes: list[str], strategy: dict, window_days: int = 30) -> dict:
    """
    统计近 window_days 内个股买入命中情况（仅 scene=scan，排除 watchlist）。
    返回: code -> {hit_rate, samples}
    """
    if not codes:
        return {}
    try:
        from learning.tracker import list_prediction_dates, load_predictions, load_outcomes
    except Exception as err:
        log.warning(f"[scan_bot] load tracker helpers failed: {err}")
        return {}

    tracked = {str(c) for c in codes if c}
    if not tracked:
        return {}
    rise_target = float(strategy.get("rise_target_pct", 1.0) or 1.0)
    dates = list_prediction_dates()
    recent = dates[-window_days:] if len(dates) >= window_days else dates

    stats = {code: {"hits": 0, "samples": 0} for code in tracked}
    for d in recent:
        preds = load_predictions(d) or []
        outs = load_outcomes(d) or {}
        if not preds or not outs:
            continue
        for p in preds:
            code = str(p.get("code") or "")
            if code not in tracked:
                continue
            if p.get("scene") not in (None, "", "scan"):
                continue
            if p.get("watchlist"):
                continue
            if p.get("signal") != "买入":
                continue
            actual_pct = (outs.get(code) or {}).get("actual_pct")
            if actual_pct is None:
                continue
            stats[code]["samples"] += 1
            stats[code]["hits"] += int(float(actual_pct) >= rise_target)

    out = {}
    for code, s in stats.items():
        n = int(s.get("samples", 0))
        if n <= 0:
            continue
        hit_rate = float(s.get("hits", 0)) / n
        out[code] = {"hit_rate": hit_rate, "samples": n}
    return out


def _load_recent_symbol_risk_stats(codes: list[str], window_days: int = 30, severe_dd_threshold: float = -0.10) -> dict:
    """
    统计近 window_days 内个股 5 日回撤风险。
    返回: code -> {severe_rate, samples}
    severe_rate = max_drawdown_5d 触发严重回撤阈值的比例（阈值由调用侧控制）。
    """
    if not codes:
        return {}
    try:
        from learning.tracker import list_prediction_dates, load_predictions, load_outcomes
    except Exception as err:
        log.warning(f"[scan_bot] load tracker helpers for risk stats failed: {err}")
        return {}

    tracked = {str(c) for c in codes if c}
    if not tracked:
        return {}
    dates = list_prediction_dates()
    recent = dates[-window_days:] if len(dates) >= window_days else dates

    stats = {code: {"severe": 0, "samples": 0} for code in tracked}
    for d in recent:
        preds = load_predictions(d) or []
        outs = load_outcomes(d) or {}
        if not preds or not outs:
            continue
        for p in preds:
            code = str(p.get("code") or "")
            if code not in tracked:
                continue
            if p.get("scene") not in (None, "", "scan"):
                continue
            if p.get("watchlist"):
                continue
            if p.get("signal") != "买入":
                continue
            out = outs.get(code) or {}
            dd5 = out.get("max_drawdown_5d")
            if not isinstance(dd5, (int, float)):
                continue
            stats[code]["samples"] += 1
            stats[code]["severe"] += int(float(dd5) <= float(severe_dd_threshold))

    out = {}
    for code, s in stats.items():
        n = int(s.get("samples", 0))
        if n <= 0:
            continue
        severe_rate = float(s.get("severe", 0)) / n
        out[code] = {"severe_rate": severe_rate, "samples": n}
    return out


def _extract_payoff_quality(row: dict) -> tuple[float | None, float | None, float | None]:
    """
    从候选行提取收益/风险质量:
    - gain_pct: 双周期目标涨幅中的较大值（百分比，如 3.2 表示 +3.2%）
    - risk_pct: 历史回撤绝对值（百分比）
    - payoff_ratio: gain_pct / risk_pct
    任一关键字段缺失时返回 None。
    """
    dual = row.get("dual_trade", {}) or {}
    short_gain = (dual.get("short", {}) or {}).get("gain_pct")
    long_gain = (dual.get("long", {}) or {}).get("gain_pct")
    gain_candidates = [
        float(v) for v in (short_gain, long_gain)
        if isinstance(v, (int, float))
    ]
    if not gain_candidates:
        return None, None, None
    gain_pct = max(gain_candidates)

    drawdown = row.get("drawdown")
    if not isinstance(drawdown, (int, float)):
        return gain_pct, None, None
    risk_pct = abs(float(drawdown)) * 100.0
    if risk_pct <= 0:
        return gain_pct, 0.0, None
    return gain_pct, risk_pct, gain_pct / risk_pct


def _technical_confirmation_count(row: dict, cfg: dict, mode: str) -> tuple[int, int]:
    """
    统计技术共振确认数。
    返回 (hit_count, available_count)。
    """
    last = row.get("last", {}) or {}
    checks: list[bool] = []

    rsi6 = last.get("rsi6")
    if isinstance(rsi6, (int, float)):
        rsi_floor = float(cfg.get("tech_rsi_floor_hard", 53.0) if mode == "hard" else cfg.get("tech_rsi_floor_weak", 50.0))
        checks.append(float(rsi6) >= rsi_floor)

    macd_hist = last.get("macd_hist")
    if isinstance(macd_hist, (int, float)):
        checks.append(float(macd_hist) > 0.0)

    vol_ratio = last.get("vol_ratio")
    if isinstance(vol_ratio, (int, float)):
        vol_floor = float(cfg.get("tech_vol_ratio_floor_hard", 1.10) if mode == "hard" else cfg.get("tech_vol_ratio_floor_weak", 1.00))
        checks.append(float(vol_ratio) >= vol_floor)

    close_p = last.get("close")
    ma20 = last.get("ma20")
    if isinstance(close_p, (int, float)) and isinstance(ma20, (int, float)) and float(ma20) > 0:
        checks.append(float(close_p) >= float(ma20))

    if not checks:
        return 0, 0
    return sum(1 for x in checks if x), len(checks)


def _trend_alignment_count(row: dict, cfg: dict, mode: str) -> tuple[int, int]:
    """
    统计趋势一致性确认数（均线结构 + 收盘位置）。
    返回 (hit_count, available_count)。
    """
    last = row.get("last", {}) or {}
    checks: list[bool] = []

    ma5 = last.get("ma5")
    ma10 = last.get("ma10")
    ma20 = last.get("ma20")
    ma60 = last.get("ma60")
    close_p = last.get("close")

    if all(isinstance(v, (int, float)) for v in (ma5, ma10, ma20)):
        checks.append(float(ma5) >= float(ma10) >= float(ma20))
    if all(isinstance(v, (int, float)) for v in (ma10, ma20, ma60)):
        checks.append(float(ma10) >= float(ma20) >= float(ma60))
    if isinstance(close_p, (int, float)) and isinstance(ma20, (int, float)):
        checks.append(float(close_p) >= float(ma20))
    if isinstance(close_p, (int, float)) and isinstance(ma5, (int, float)):
        close_buffer = float(cfg.get("trend_close_buffer_hard", 0.0) if mode == "hard" else cfg.get("trend_close_buffer_weak", -0.01))
        checks.append(float(close_p) >= float(ma5) * (1.0 + close_buffer))

    if not checks:
        return 0, 0
    return sum(1 for x in checks if x), len(checks)


def _load_current_tactic_stats() -> dict:
    """读取当前市场状态下各战法统计（中文战法名 -> {precision, samples}）。"""
    import json
    from pathlib import Path
    try:
        from learning.market_state import load_current_state
        state = load_current_state().get("current", "range")
        data = json.loads(Path("learning/tactic_params.json").read_text(encoding="utf-8"))
        params = data.get("params", {}).get(state, {})
        mapping = {"价值": "value", "成长": "growth", "龙头": "leader", "逆向": "contra"}
        out = {}
        for cn, key in mapping.items():
            item = params.get(key, {})
            v = item.get("precision")
            if isinstance(v, (int, float)):
                out[cn] = {
                    "precision": float(v),
                    "samples": int(item.get("samples", 0) or 0),
                }
        return out
    except Exception:
        return {}


def _load_current_tactic_precisions() -> dict:
    """读取当前市场状态下各战法精准率（中文战法名 -> precision）。"""
    stats = _load_current_tactic_stats()
    return {k: float(v.get("precision", 0.0)) for k, v in stats.items()}


def _confidence_factor(confidence: str) -> float:
    c = str(confidence or "").strip().lower()
    if c in ("高", "high", "h"):
        return 1.06
    if c in ("中", "medium", "mid", "m"):
        return 1.00
    if c in ("低", "low", "l"):
        return 0.94
    return 1.00


def _apply_precision_rerank(results: list) -> list:
    """对推荐候选做精度导向重排（不改 signal/rise_prob）。"""
    if not results:
        return results
    cfg = _load_recommend_scoring_cfg()
    tactic_prec = _load_current_tactic_precisions()

    w_prob = float(cfg.get("w_prob", 0.62))
    w_rank = float(cfg.get("w_rank", 0.18))
    w_mom = float(cfg.get("w_momentum", 0.12))
    w_res = float(cfg.get("w_resonance", 0.08))
    w_tactic = float(cfg.get("w_tactic_precision", 0.40))
    atr_ref = float(cfg.get("atr_ref", 0.06))
    atr_penalty_scale = float(cfg.get("atr_penalty_scale", 0.70))
    atr_penalty_cap = float(cfg.get("atr_penalty_cap", 0.10))
    dd_ref = float(cfg.get("drawdown_ref", -0.35))
    dd_penalty_scale = float(cfg.get("drawdown_penalty_scale", 0.25))
    dd_penalty_cap = float(cfg.get("drawdown_penalty_cap", 0.08))
    momentum_divisor = float(cfg.get("momentum_divisor", 2.0))

    rescored = []
    for r in results:
        rise_prob = float(r.get("rise_prob", 0.0) or 0.0)
        rank_pct = float(r.get("global_rank_pct", 1.0) or 1.0)
        momentum = float(r.get("momentum", 0.0) or 0.0)
        resonance = int(r.get("tactic_resonance", 0) or 0)
        drawdown = float(r.get("drawdown", 0.0) or 0.0)
        last = r.get("last", {}) or {}
        close_p = float(last.get("close") or 0.0)
        atr14 = float(last.get("atr14") or 0.0)
        atr_pct = (atr14 / close_p) if close_p > 0 and atr14 > 0 else 0.0

        momentum_norm = min(max(momentum / momentum_divisor, 0.0), 1.0)
        resonance_norm = min(max(resonance, 0), 3) / 3.0
        base_score = (
            rise_prob * w_prob
            + (1.0 - min(max(rank_pct, 0.0), 1.0)) * w_rank
            + momentum_norm * w_mom
            + resonance_norm * w_res
        )

        tactic_bonus = 0.0
        tags = r.get("tactic_tags", {}) or {}
        hit_precisions = [tactic_prec.get(t) for t in tags if tags.get(t) and t in tactic_prec]
        if hit_precisions:
            avg_prec = sum(hit_precisions) / len(hit_precisions)
            tactic_bonus = max(-0.05, min(0.08, (avg_prec - 0.50) * w_tactic))

        vol_penalty = 0.0
        if atr_pct > atr_ref:
            vol_penalty = min((atr_pct - atr_ref) * atr_penalty_scale, atr_penalty_cap)

        dd_penalty = 0.0
        if drawdown < dd_ref:
            dd_penalty = min((abs(drawdown) - abs(dd_ref)) * dd_penalty_scale, dd_penalty_cap)

        conf_factor = _confidence_factor(r.get("confidence", ""))
        recommend_score = (base_score + tactic_bonus - vol_penalty - dd_penalty) * conf_factor

        rr = dict(r)
        rr["recommend_score"] = round(float(recommend_score), 6)
        rescored.append(rr)

    rescored.sort(
        key=lambda x: (
            -float(x.get("recommend_score", 0.0)),
            float(x.get("global_rank_pct", 1.0)),
            -float(x.get("rise_prob", 0.0)),
        )
    )
    return rescored


def _apply_precision_gate(results: list, strategy: dict) -> tuple[list, dict]:
    """
    在近期精准率走弱时，对低质量买入候选降级为观望，降低误报。
    仅影响推荐展示排序，不修改原始涨概率。
    """
    if not results:
        return results, {"enabled": False, "applied": False}

    cfg = _load_recommend_precision_gate_cfg()
    enabled = bool(cfg.get("enabled", True))
    context = _load_recent_accuracy_context(strategy)
    summary = {
        "enabled": enabled,
        "applied": False,
        "mode": "off",
        "demoted": 0,
        "buy_before": sum(1 for r in results if r.get("signal") == "买入"),
        "buy_after": sum(1 for r in results if r.get("signal") == "买入"),
        "acc_7d": float(context.get("acc_7d", 0.0)),
        "acc_30d": float(context.get("acc_30d", 0.0)),
        "samples_7d": int(context.get("samples_7d", 0)),
        "samples_30d": int(context.get("samples_30d", 0)),
        "demoted_by_bin": 0,
        "demoted_by_confidence": 0,
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
        "tuning_level": "neutral",
        "tuning_strength": 0.0,
    }
    if not enabled:
        return results, summary

    min_samples_7d = int(cfg.get("min_samples_7d", 15))
    min_samples_30d = int(cfg.get("min_samples_30d", 60))
    if summary["samples_7d"] < min_samples_7d or summary["samples_30d"] < min_samples_30d:
        return results, summary

    weak_acc_7d = float(cfg.get("weak_acc_7d", 0.54))
    weak_acc_30d = float(cfg.get("weak_acc_30d", 0.50))
    hard_acc_30d = float(cfg.get("hard_acc_30d", 0.42))
    if summary["acc_30d"] < hard_acc_30d:
        mode = "hard"
    elif summary["acc_7d"] < weak_acc_7d or summary["acc_30d"] < weak_acc_30d:
        mode = "weak"
    else:
        return results, summary
    tuning_profile = _resolve_adaptive_tuning_profile(
        cfg, mode, summary,
        weak_acc_7d=weak_acc_7d,
        weak_acc_30d=weak_acc_30d,
        hard_acc_30d=hard_acc_30d,
    )
    summary["tuning_level"] = tuning_profile.get("level", "neutral")
    summary["tuning_strength"] = float(tuning_profile.get("strength", 0.0) or 0.0)

    if mode == "hard":
        min_score = float(cfg.get("min_recommend_score_hard", 0.48))
        min_prob = float(cfg.get("min_rise_prob_hard", 0.60))
        min_resonance = int(cfg.get("min_resonance_hard", 2))
    else:
        min_score = float(cfg.get("min_recommend_score_weak", 0.44))
        min_prob = float(cfg.get("min_rise_prob_weak", 0.57))
        min_resonance = int(cfg.get("min_resonance_weak", 1))

    protect_top_buy = int(
        cfg.get("protect_top_buy_hard", cfg.get("protect_top_buy", 2)) if mode == "hard"
        else cfg.get("protect_top_buy_weak", cfg.get("protect_top_buy", 3))
    )
    max_demote_ratio = float(
        cfg.get("max_demote_ratio_hard", cfg.get("max_demote_ratio", 0.65)) if mode == "hard"
        else cfg.get("max_demote_ratio_weak", cfg.get("max_demote_ratio", 0.50))
    )
    max_demote_ratio = max(0.0, min(max_demote_ratio, 1.0))
    use_prob_bin_guard = bool(cfg.get("use_prob_bin_guard", True))
    prob_bin_min_samples = int(cfg.get("prob_bin_min_samples", 12))
    prob_bin_min_hit_rate = float(
        cfg.get("prob_bin_min_hit_rate_hard", 0.52) if mode == "hard"
        else cfg.get("prob_bin_min_hit_rate_weak", 0.48)
    )
    prob_bin_stats = _load_prob_bin_precision_stats() if use_prob_bin_guard else {}
    use_confidence_guard = bool(cfg.get("use_confidence_guard", True))
    confidence_min_samples = int(cfg.get("confidence_min_samples", 12))
    confidence_thresholds = _confidence_threshold_map(cfg, mode)
    confidence_stats = _load_confidence_precision_stats() if use_confidence_guard else {}
    use_tactic_guard = bool(cfg.get("use_tactic_guard", True))
    tactic_min_samples = int(cfg.get("tactic_min_samples", 30))
    tactic_min_precision = float(
        cfg.get("tactic_min_precision_hard", 0.54) if mode == "hard"
        else cfg.get("tactic_min_precision_weak", 0.50)
    )
    tactic_stats = _load_current_tactic_stats() if use_tactic_guard else {}
    use_symbol_memory_guard = bool(cfg.get("use_symbol_memory_guard", True))
    symbol_memory_window_days = int(cfg.get("symbol_memory_window_days", 30))
    symbol_memory_min_samples = int(cfg.get("symbol_memory_min_samples", 3))
    symbol_memory_min_hit_rate = float(
        cfg.get("symbol_memory_min_hit_rate_hard", 0.45) if mode == "hard"
        else cfg.get("symbol_memory_min_hit_rate_weak", 0.40)
    )
    symbol_stats = {}
    if use_symbol_memory_guard:
        symbol_stats = _load_recent_symbol_buy_stats(
            [str(r.get("code") or "") for r in results], strategy, window_days=symbol_memory_window_days
        )
    use_symbol_risk_guard = bool(cfg.get("use_symbol_risk_guard", True))
    symbol_risk_window_days = int(cfg.get("symbol_risk_window_days", 45))
    symbol_risk_min_samples = int(cfg.get("symbol_risk_min_samples", 3))
    symbol_risk_dd_threshold = float(cfg.get("symbol_risk_dd_threshold", -0.10))
    symbol_risk_max_rate = float(
        cfg.get("symbol_risk_max_rate_hard", 0.45) if mode == "hard"
        else cfg.get("symbol_risk_max_rate_weak", 0.55)
    )
    symbol_risk_stats = {}
    if use_symbol_risk_guard:
        symbol_risk_stats = _load_recent_symbol_risk_stats(
            [str(r.get("code") or "") for r in results],
            window_days=symbol_risk_window_days,
            severe_dd_threshold=symbol_risk_dd_threshold,
        )
    use_payoff_guard = bool(cfg.get("use_payoff_guard", True))
    payoff_min_gain = float(
        cfg.get("payoff_min_gain_hard", 2.5) if mode == "hard"
        else cfg.get("payoff_min_gain_weak", 2.0)
    )
    payoff_min_ratio = float(
        cfg.get("payoff_min_ratio_hard", 0.60) if mode == "hard"
        else cfg.get("payoff_min_ratio_weak", 0.45)
    )
    payoff_min_risk_pct = float(cfg.get("payoff_min_risk_pct", 6.0))
    use_technical_guard = bool(cfg.get("use_technical_guard", True))
    tech_min_confirmations = int(
        cfg.get("tech_min_confirmations_hard", 3) if mode == "hard"
        else cfg.get("tech_min_confirmations_weak", 2)
    )
    tech_min_available = int(cfg.get("tech_min_available_checks", 3))
    use_liquidity_guard = bool(cfg.get("use_liquidity_guard", True))
    liquidity_min_turnover = float(
        cfg.get("liquidity_min_turnover_hard", 1.8) if mode == "hard"
        else cfg.get("liquidity_min_turnover_weak", 1.0)
    )
    liquidity_max_turnover = float(
        cfg.get("liquidity_max_turnover_hard", 20.0) if mode == "hard"
        else cfg.get("liquidity_max_turnover_weak", 25.0)
    )
    liquidity_min_vol_ratio = float(
        cfg.get("liquidity_min_vol_ratio_hard", 1.0) if mode == "hard"
        else cfg.get("liquidity_min_vol_ratio_weak", 0.9)
    )
    use_trend_guard = bool(cfg.get("use_trend_guard", True))
    trend_min_hits = int(
        cfg.get("trend_min_hits_hard", 3) if mode == "hard"
        else cfg.get("trend_min_hits_weak", 2)
    )
    trend_min_available = int(cfg.get("trend_min_available_checks", 3))
    use_extension_guard = bool(cfg.get("use_extension_guard", True))
    max_close_ma20_dev = float(
        cfg.get("max_close_ma20_dev_hard", 0.08) if mode == "hard"
        else cfg.get("max_close_ma20_dev_weak", 0.12)
    )
    use_volatility_guard = bool(cfg.get("use_volatility_guard", True))
    atr_pct_max = float(
        cfg.get("atr_pct_max_hard", 0.07) if mode == "hard"
        else cfg.get("atr_pct_max_weak", 0.10)
    )
    use_edge_guard = bool(cfg.get("use_edge_guard", True))
    edge_min_prob = float(
        cfg.get("edge_min_prob_hard", 0.015) if mode == "hard"
        else cfg.get("edge_min_prob_weak", 0.008)
    )
    last_thresh_buy = strategy.get("last_thresh_buy")
    has_last_thresh_buy = isinstance(last_thresh_buy, (int, float))
    use_dual_target_guard = bool(cfg.get("use_dual_target_guard", True))
    dual_min_short_gain = float(
        cfg.get("dual_min_short_gain_hard", 2.0) if mode == "hard"
        else cfg.get("dual_min_short_gain_weak", 1.5)
    )
    dual_min_long_gain = float(
        cfg.get("dual_min_long_gain_hard", 3.0) if mode == "hard"
        else cfg.get("dual_min_long_gain_weak", 2.2)
    )
    dual_min_long_short_ratio = float(
        cfg.get("dual_min_long_short_ratio_hard", 0.90) if mode == "hard"
        else cfg.get("dual_min_long_short_ratio_weak", 0.75)
    )
    use_atr_reward_guard = bool(cfg.get("use_atr_reward_guard", True))
    min_gain_atr_ratio = float(
        cfg.get("min_gain_atr_ratio_hard", 1.25) if mode == "hard"
        else cfg.get("min_gain_atr_ratio_weak", 1.00)
    )
    min_factor = max(0.5, float(tuning_profile.get("min_factor", 1.0)))
    max_factor = max(0.5, float(tuning_profile.get("max_factor", 1.0)))
    count_delta = int(tuning_profile.get("count_delta", 0))
    demote_ratio_factor = max(0.1, float(tuning_profile.get("demote_ratio_factor", 1.0)))

    min_score *= min_factor
    min_prob *= min_factor
    min_resonance = max(0, min_resonance + count_delta)
    max_demote_ratio = max(0.0, min(max_demote_ratio * demote_ratio_factor, 1.0))
    prob_bin_min_hit_rate = max(0.0, min(prob_bin_min_hit_rate * min_factor, 1.0))
    confidence_thresholds = {k: max(0.0, min(float(v) * min_factor, 1.0)) for k, v in confidence_thresholds.items()}
    tactic_min_precision = max(0.0, min(tactic_min_precision * min_factor, 1.0))
    symbol_memory_min_hit_rate = max(0.0, min(symbol_memory_min_hit_rate * min_factor, 1.0))
    symbol_risk_max_rate = max(0.0, min(symbol_risk_max_rate * max_factor, 1.0))
    payoff_min_gain = max(0.0, payoff_min_gain * min_factor)
    payoff_min_ratio = max(0.0, payoff_min_ratio * min_factor)
    tech_min_confirmations = max(1, tech_min_confirmations + count_delta)
    liquidity_min_turnover = max(0.0, liquidity_min_turnover * min_factor)
    liquidity_max_turnover = max(liquidity_min_turnover, liquidity_max_turnover * max_factor)
    liquidity_min_vol_ratio = max(0.0, liquidity_min_vol_ratio * min_factor)
    trend_min_hits = max(1, trend_min_hits + count_delta)
    max_close_ma20_dev = max(0.0, max_close_ma20_dev * max_factor)
    atr_pct_max = max(0.0, atr_pct_max * max_factor)
    edge_min_prob = max(0.0, edge_min_prob * min_factor)
    dual_min_short_gain = max(0.0, dual_min_short_gain * min_factor)
    dual_min_long_gain = max(0.0, dual_min_long_gain * min_factor)
    dual_min_long_short_ratio = max(0.0, dual_min_long_short_ratio * min_factor)
    min_gain_atr_ratio = max(0.0, min_gain_atr_ratio * min_factor)

    buy_indexes = [i for i, r in enumerate(results) if r.get("signal") == "买入"]
    max_demote = int(len(buy_indexes) * max_demote_ratio)
    if max_demote <= 0:
        return results, summary

    gated = list(results)
    demoted = 0
    for buy_rank, idx in enumerate(buy_indexes):
        if buy_rank < protect_top_buy:
            continue
        if demoted >= max_demote:
            break

        row = gated[idx]
        score = float(row.get("recommend_score", 0.0) or 0.0)
        prob = float(row.get("rise_prob", 0.0) or 0.0)
        resonance = int(row.get("tactic_resonance", 0) or 0)
        failed = []
        bin_demoted = False
        confidence_demoted = False
        tactic_demoted = False
        symbol_demoted = False
        payoff_demoted = False
        technical_demoted = False
        symbol_risk_demoted = False
        liquidity_demoted = False
        trend_demoted = False
        extension_demoted = False
        volatility_demoted = False
        edge_demoted = False
        dual_target_demoted = False
        atr_reward_demoted = False
        if score < min_score:
            failed.append(f"score<{min_score:.2f}")
        if prob < min_prob:
            failed.append(f"prob<{min_prob:.2f}")
        if resonance < min_resonance:
            failed.append(f"res<{min_resonance}")
        if prob_bin_stats:
            label = _prob_bin_label(prob)
            info = prob_bin_stats.get(label, {})
            b_samples = int(info.get("samples", 0) or 0)
            b_rate = float(info.get("hit_rate", 0.0) or 0.0)
            if b_samples >= prob_bin_min_samples and b_rate < prob_bin_min_hit_rate:
                failed.append(f"bin<{prob_bin_min_hit_rate:.2f}")
                bin_demoted = True
        if confidence_stats:
            conf_label = _normalize_confidence_label(row.get("confidence", "中"))
            info = confidence_stats.get(conf_label, {})
            c_samples = int(info.get("samples", 0) or 0)
            c_rate = float(info.get("hit_rate", 0.0) or 0.0)
            c_threshold = float(confidence_thresholds.get(conf_label, 0.0))
            if c_samples >= confidence_min_samples and c_rate < c_threshold:
                failed.append(f"conf<{c_threshold:.2f}")
                confidence_demoted = True
        if tactic_stats:
            tags = row.get("tactic_tags", {}) or {}
            matched = []
            for name, hit in tags.items():
                if not hit:
                    continue
                st = tactic_stats.get(name)
                if not st:
                    continue
                samples = int(st.get("samples", 0) or 0)
                if samples >= tactic_min_samples:
                    matched.append(float(st.get("precision", 0.0) or 0.0))
            if matched:
                avg_tactic_prec = sum(matched) / len(matched)
                if avg_tactic_prec < tactic_min_precision:
                    failed.append(f"tactic<{tactic_min_precision:.2f}")
                    tactic_demoted = True
        if symbol_stats:
            code = str(row.get("code") or "")
            st = symbol_stats.get(code, {})
            s_samples = int(st.get("samples", 0) or 0)
            s_rate = float(st.get("hit_rate", 0.0) or 0.0)
            if s_samples >= symbol_memory_min_samples and s_rate < symbol_memory_min_hit_rate:
                failed.append(f"symbol<{symbol_memory_min_hit_rate:.2f}")
                symbol_demoted = True
        if use_payoff_guard:
            gain_pct, risk_pct, payoff_ratio = _extract_payoff_quality(row)
            if gain_pct is not None:
                if gain_pct < payoff_min_gain:
                    failed.append(f"gain<{payoff_min_gain:.1f}%")
                    payoff_demoted = True
                elif (
                    isinstance(risk_pct, (int, float))
                    and risk_pct >= payoff_min_risk_pct
                    and isinstance(payoff_ratio, (int, float))
                    and payoff_ratio < payoff_min_ratio
                ):
                    failed.append(f"payoff<{payoff_min_ratio:.2f}")
                    payoff_demoted = True
        if use_technical_guard:
            hit_cnt, avail_cnt = _technical_confirmation_count(row, cfg, mode)
            if avail_cnt >= tech_min_available and hit_cnt < tech_min_confirmations:
                failed.append(f"tech<{tech_min_confirmations}/{avail_cnt}")
                technical_demoted = True
        if symbol_risk_stats:
            code = str(row.get("code") or "")
            rst = symbol_risk_stats.get(code, {})
            r_samples = int(rst.get("samples", 0) or 0)
            severe_rate = float(rst.get("severe_rate", 0.0) or 0.0)
            if r_samples >= symbol_risk_min_samples and severe_rate > symbol_risk_max_rate:
                failed.append(f"risk>{symbol_risk_max_rate:.2f}")
                symbol_risk_demoted = True
        if use_liquidity_guard:
            last = row.get("last", {}) or {}
            turnover = last.get("turnover")
            vol_ratio = last.get("vol_ratio")
            if isinstance(turnover, (int, float)):
                to = float(turnover)
                if to < liquidity_min_turnover:
                    failed.append(f"turnover<{liquidity_min_turnover:.1f}")
                    liquidity_demoted = True
                elif to > liquidity_max_turnover:
                    failed.append(f"turnover>{liquidity_max_turnover:.1f}")
                    liquidity_demoted = True
            if isinstance(vol_ratio, (int, float)) and float(vol_ratio) < liquidity_min_vol_ratio:
                failed.append(f"vol<{liquidity_min_vol_ratio:.2f}")
                liquidity_demoted = True
        if use_trend_guard:
            trend_hits, trend_available = _trend_alignment_count(row, cfg, mode)
            if trend_available >= trend_min_available and trend_hits < trend_min_hits:
                failed.append(f"trend<{trend_min_hits}/{trend_available}")
                trend_demoted = True
        if use_extension_guard:
            last = row.get("last", {}) or {}
            close_p = last.get("close")
            ma20 = last.get("ma20")
            if isinstance(close_p, (int, float)) and isinstance(ma20, (int, float)) and float(ma20) > 0:
                ext = float(close_p) / float(ma20) - 1.0
                if ext > max_close_ma20_dev:
                    failed.append(f"ext>{max_close_ma20_dev:.2f}")
                    extension_demoted = True
        if use_volatility_guard:
            last = row.get("last", {}) or {}
            close_p = last.get("close")
            atr14 = last.get("atr14")
            if isinstance(close_p, (int, float)) and isinstance(atr14, (int, float)) and float(close_p) > 0:
                atr_pct = float(atr14) / float(close_p)
                if atr_pct > atr_pct_max:
                    failed.append(f"atr>{atr_pct_max:.2f}")
                    volatility_demoted = True
        if use_edge_guard and has_last_thresh_buy:
            edge = float(prob) - float(last_thresh_buy)
            if edge < edge_min_prob:
                failed.append(f"edge<{edge_min_prob:.3f}")
                edge_demoted = True
        if use_dual_target_guard:
            dual = row.get("dual_trade", {}) or {}
            sg = (dual.get("short", {}) or {}).get("gain_pct")
            lg = (dual.get("long", {}) or {}).get("gain_pct")
            if isinstance(sg, (int, float)) and isinstance(lg, (int, float)):
                short_gain = float(sg)
                long_gain = float(lg)
                if short_gain < dual_min_short_gain:
                    failed.append(f"s_gain<{dual_min_short_gain:.1f}%")
                    dual_target_demoted = True
                if long_gain < dual_min_long_gain:
                    failed.append(f"l_gain<{dual_min_long_gain:.1f}%")
                    dual_target_demoted = True
                if short_gain > 0 and (long_gain / short_gain) < dual_min_long_short_ratio:
                    failed.append(f"l/s<{dual_min_long_short_ratio:.2f}")
                    dual_target_demoted = True
        if use_atr_reward_guard:
            last = row.get("last", {}) or {}
            close_p = last.get("close")
            atr14 = last.get("atr14")
            dual = row.get("dual_trade", {}) or {}
            sg = (dual.get("short", {}) or {}).get("gain_pct")
            lg = (dual.get("long", {}) or {}).get("gain_pct")
            gains = [float(x) for x in (sg, lg) if isinstance(x, (int, float))]
            if isinstance(close_p, (int, float)) and isinstance(atr14, (int, float)) and float(close_p) > 0 and gains:
                atr_pct = (float(atr14) / float(close_p)) * 100.0
                if atr_pct > 0:
                    gain_atr_ratio = max(gains) / atr_pct
                    if gain_atr_ratio < min_gain_atr_ratio:
                        failed.append(f"g/atr<{min_gain_atr_ratio:.2f}")
                        atr_reward_demoted = True
        if not failed:
            continue

        rr = dict(row)
        rr["signal"] = "观望"
        rr["precision_gate_mode"] = mode
        rr["precision_gate_reason"] = ",".join(failed)
        gated[idx] = rr
        demoted += 1
        if bin_demoted:
            summary["demoted_by_bin"] += 1
        if confidence_demoted:
            summary["demoted_by_confidence"] += 1
        if tactic_demoted:
            summary["demoted_by_tactic"] += 1
        if symbol_demoted:
            summary["demoted_by_symbol"] += 1
        if payoff_demoted:
            summary["demoted_by_payoff"] += 1
        if technical_demoted:
            summary["demoted_by_technical"] += 1
        if symbol_risk_demoted:
            summary["demoted_by_symbol_risk"] += 1
        if liquidity_demoted:
            summary["demoted_by_liquidity"] += 1
        if trend_demoted:
            summary["demoted_by_trend"] += 1
        if extension_demoted:
            summary["demoted_by_extension"] += 1
        if volatility_demoted:
            summary["demoted_by_volatility"] += 1
        if edge_demoted:
            summary["demoted_by_edge"] += 1
        if dual_target_demoted:
            summary["demoted_by_dual_target"] += 1
        if atr_reward_demoted:
            summary["demoted_by_atr_reward"] += 1

    summary["applied"] = demoted > 0
    summary["mode"] = mode
    summary["demoted"] = demoted
    summary["buy_after"] = summary["buy_before"] - demoted
    return gated, summary


def _precision_gate_line(summary: dict) -> str:
    if not isinstance(summary, dict) or not summary.get("applied"):
        return ""
    buy_before = int(summary.get("buy_before", 0) or 0)
    buy_after = int(summary.get("buy_after", 0) or 0)
    demoted = int(summary.get("demoted", 0) or 0)
    demote_rate = (demoted / buy_before) if buy_before > 0 else 0.0
    contrib = [
        ("分桶", int(summary.get("demoted_by_bin", 0) or 0)),
        ("置信度", int(summary.get("demoted_by_confidence", 0) or 0)),
        ("战法", int(summary.get("demoted_by_tactic", 0) or 0)),
        ("个股记忆", int(summary.get("demoted_by_symbol", 0) or 0)),
        ("收益风险", int(summary.get("demoted_by_payoff", 0) or 0)),
        ("技术共振", int(summary.get("demoted_by_technical", 0) or 0)),
        ("回撤风险", int(summary.get("demoted_by_symbol_risk", 0) or 0)),
        ("流动性", int(summary.get("demoted_by_liquidity", 0) or 0)),
        ("趋势一致", int(summary.get("demoted_by_trend", 0) or 0)),
        ("过热偏离", int(summary.get("demoted_by_extension", 0) or 0)),
        ("极端波动", int(summary.get("demoted_by_volatility", 0) or 0)),
        ("概率边际", int(summary.get("demoted_by_edge", 0) or 0)),
        ("双周期目标", int(summary.get("demoted_by_dual_target", 0) or 0)),
        ("ATR收益比", int(summary.get("demoted_by_atr_reward", 0) or 0)),
    ]
    top_contrib = [x for x in contrib if x[1] > 0]
    top_contrib.sort(key=lambda x: x[1], reverse=True)
    top_str = "、".join(f"{k}×{v}" for k, v in top_contrib[:3]) if top_contrib else "无"
    return (
        f"🎯 提精闸门：{summary.get('mode')} 模式（调优 {summary.get('tuning_level', 'neutral')},"
        f" 强度 {float(summary.get('tuning_strength', 0.0)):.0%}），"
        f"7日 **{float(summary.get('acc_7d', 0.0)):.1%}** / "
        f"30日 **{float(summary.get('acc_30d', 0.0)):.1%}**，"
        f"降级 **{demoted}** 只低质买入候选"
        f"（{buy_before}→{buy_after}，降级率 **{demote_rate:.0%}**）"
        f"；主要来源：{top_str}"
        f"（分桶 **{int(summary.get('demoted_by_bin', 0))}** / "
        f"置信度 **{int(summary.get('demoted_by_confidence', 0))}** / "
        f"战法 **{int(summary.get('demoted_by_tactic', 0))}** / "
        f"个股记忆 **{int(summary.get('demoted_by_symbol', 0))}** / "
        f"收益风险 **{int(summary.get('demoted_by_payoff', 0))}** / "
        f"技术共振 **{int(summary.get('demoted_by_technical', 0))}** / "
        f"回撤风险 **{int(summary.get('demoted_by_symbol_risk', 0))}** / "
        f"流动性 **{int(summary.get('demoted_by_liquidity', 0))}** / "
        f"趋势一致 **{int(summary.get('demoted_by_trend', 0))}** / "
        f"过热偏离 **{int(summary.get('demoted_by_extension', 0))}** / "
        f"极端波动 **{int(summary.get('demoted_by_volatility', 0))}** / "
        f"概率边际 **{int(summary.get('demoted_by_edge', 0))}** / "
        f"双周期目标 **{int(summary.get('demoted_by_dual_target', 0))}** / "
        f"ATR收益比 **{int(summary.get('demoted_by_atr_reward', 0))}**）"
    )


# ── P2 ai_reason 集成辅助 ────────────────────────────────

def _build_ai_reason_text(r: dict, news_items: list, total: int, sector: str) -> str:
    """把 scan_bot 的股票 dict 转 StockContext 并调 ai_reason.generate,
    返回可嵌入 lines 的 markdown 文本(单行);None/失败返回空字符串不插入。

    feature flag: config.yaml ai_reason.enable (缺省 true)
    """
    cfg = _load_cfg()
    if not (cfg.get("ai_reason") or {}).get("enable", True):
        return ""

    from server.ai_reason import StockContext, generate, render_card_section
    from learning import market_state
    try:
        state = market_state.load_current_state().get("current", "range")
    except Exception:
        state = "range"
    try:
        from learning.optimizer import load_strategy
        acc_30d = load_strategy().get("accuracy_30d")
    except Exception:
        acc_30d = None

    tags = r.get("tactic_tags", {}) or {}
    last = r.get("last", {}) or {}
    news_summary = " / ".join(
        (n[0] if isinstance(n, tuple) else str(n)) for n in news_items[:3]
    )

    ctx = StockContext(
        code=r.get("code", ""),
        name=r.get("name", r.get("code", "")),
        rise_prob_raw=float(r.get("rise_prob_raw", r.get("rise_prob", 0.0))),
        rise_prob_cal=float(r.get("rise_prob_cal", r.get("rise_prob", 0.0))),
        market_state=state,
        acc_30d=float(acc_30d) if acc_30d is not None else None,
        tactic_hits=list(tags.keys()),
        drawdown=float(r.get("drawdown", 0.0)),
        fin=r.get("fin_res", {}) or {},
        tech={
            "rsi6":      float(last.get("rsi6") or 50),
            "macd_hist": float(last.get("macd_hist") or 0),
        },
        news_summary=news_summary,
        global_rank=int(r.get("global_rank", 0) or 0),
        scan_total=int(total or 0),
        sector=sector or "",
    )
    reason = generate(ctx)
    elems = render_card_section(reason)
    return elems[0]["content"] if elems else ""


def _scan_ack_card(title: str, color: str, desc: str, steps: list, eta: str) -> dict:
    """共享的「正在处理」ACK 卡片模板，供 cmd_scan_bot / cmd_tactic 使用。"""
    numbered = ["①", "②", "③", "④", "⑤"]
    steps_md = "\n".join(f"{numbered[i]} {s}" for i, s in enumerate(steps))
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"⏳ {title}…"},
            "template": color,
        },
        "elements": [{"tag": "markdown", "content": (
            f"{desc}\n\n{steps_md}\n\n"
            f"结果将推送到本会话，预计 **{eta}**。"
        )}],
    }


def _strategy_guardrail_snapshot(strategy: dict, current_state: str) -> dict:
    """从 strategy 中提取当前状态风险护栏快照，缺省时返回保守默认值。"""
    risk_guards = strategy.get("risk_guardrails", {}) if isinstance(strategy, dict) else {}
    bounds_all = risk_guards.get("by_state_bounds", {}) if isinstance(risk_guards, dict) else {}
    bounds = bounds_all.get(current_state) or bounds_all.get("range") or {"min": 0.08, "max": 0.25}
    min_pct = float(bounds.get("min", 0.08))
    max_pct = float(bounds.get("max", 0.25))

    cap_cfg = risk_guards.get("low_accuracy_cap", {}) if isinstance(risk_guards, dict) else {}
    cap_threshold = float(cap_cfg.get("acc_30d_threshold", 0.40))
    cap_max_pct = float(cap_cfg.get("max_buy_top_pct", 0.12))
    acc_30d = strategy.get("accuracy_30d")
    cap_triggered = isinstance(acc_30d, (int, float)) and float(acc_30d) < cap_threshold
    if cap_triggered:
        max_pct = min(max_pct, cap_max_pct)

    positioning = strategy.get("positioning", {}) if isinstance(strategy, dict) else {}
    position_pct = float(positioning.get(current_state, positioning.get("range", 0.6)))

    min_pct = max(0.01, min(min_pct, 0.50))
    max_pct = max(min_pct, min(max_pct, 0.50))
    position_pct = max(0.0, min(position_pct, 1.0))

    return {
        "min_pct": min_pct,
        "max_pct": max_pct,
        "position_pct": position_pct,
        "cap_triggered": cap_triggered,
        "cap_threshold": cap_threshold,
        "cap_max_pct": cap_max_pct,
        "acc_30d": acc_30d,
    }


def _strategy_guardrail_line(strategy: dict, *, current_state: str, buy_top_pct: float) -> str:
    """构建推荐卡片中的风险护栏提示文案。"""
    snap = _strategy_guardrail_snapshot(strategy, current_state)
    line = (
        f"🛡 护栏：门槛前 **{buy_top_pct:.0%}**（{current_state} 区间 **{snap['min_pct']:.0%}~{snap['max_pct']:.0%}**）"
        f"　建议仓位 **{snap['position_pct']:.0%}**"
    )
    if snap["cap_triggered"]:
        line += (
            f"\n↳ 30日精准率 **{float(snap['acc_30d']):.1%}** < **{snap['cap_threshold']:.0%}**，"
            f"上限收敛到 **{snap['cap_max_pct']:.0%}**"
        )
    if buy_top_pct < snap["min_pct"] or buy_top_pct > snap["max_pct"]:
        line += "\n↳ 当前门槛已偏离护栏区间，建议回到区间内执行。"
    return line


def cmd_scan_bot(top_n: int = 5) -> dict:
    """推荐 <N>：立即返回进度卡片，后台扫描完成后推送结果。"""
    import threading
    from data.fetcher import cached_codes
    from learning.optimizer import load_strategy

    if not _scan_lock.acquire(blocking=False):
        return {
            "msg_type": "text",
            "content": json.dumps(
                {"text": "已有推荐扫描正在进行，请等待当前结果完成后再发起新的推荐。"},
                ensure_ascii=False,
            ),
        }

    try:
        _cfg = _load_cfg()
        pool_cfg = _cfg.get("universe", {}).get("scan_pool", "watchlist")
        if pool_cfg == "all":
            n_codes = len(cached_codes())
        else:
            try:
                from data.universe import load_universe
                n_codes = len(load_universe(_cfg))
            except Exception:
                n_codes = len(cached_codes())
    except Exception:
        n_codes = 300

    try:
        _last_scan = load_strategy().get("last_scan_date", "")
    except Exception:
        _last_scan = ""
    _first_today = _last_scan != _today()
    if n_codes > 1000:
        eta = "约 5-15 分钟" if _first_today else "约 3-7 分钟"
    else:
        eta = "约 2-5 分钟" if _first_today else "约 1-3 分钟"

    ack_card = _scan_ack_card(
        "Antenna 推荐 扫描中", "blue",
        f"正在扫描 **{n_codes}** 只股票，筛选 Top **{top_n}**…",
        ["全量特征计算 + AI 模型打分",
         "信号分级（买入 / 观望 / 回避）",
         "战法多维评分（价值 / 成长 / 龙头 / 逆向）",
         "共振优选 + 深度分析"],
        eta,
    )

    def _run():
        try:
            result = _cmd_scan_bot_impl(top_n)
        except Exception as e:
            _log_critical_error("[推荐] 后台扫描失败", e)
            result = {"msg_type": "text", "content": json.dumps(
                {"text": f"推荐扫描失败：{e}"}, ensure_ascii=False)}
        finally:
            _scan_lock.release()
        if result is None:
            return  # 数据未就绪或已去重，静默跳过
        from server.feishu_push import push as _push
        _push(result)

    threading.Thread(target=_run, daemon=True).start()
    return {"msg_type": "interactive", "content": json.dumps(ack_card, ensure_ascii=False)}


def _cmd_scan_bot_impl(top_n: int = 5) -> dict:
    """
    推荐 <N>：扫描全量缓存股票，输出 Top N 买入候选 + 自选股快照。
    分两段：精简推荐清单 + 逐一深度分析。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import fetch_stock_hist, fetch_realtime_prices, cached_codes
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import get_active_feature_cols
    from features.analyser import predict_range, suggest_trade_levels, suggest_dual_period_trades
    from models.predictor import load_model, predict
    from learning.optimizer import load_strategy
    from learning.tracker import log_predictions

    cfg         = _load_cfg()
    strategy    = load_strategy()
    buy_top_pct = strategy.get("buy_top_pct", 0.10)

    # 股票池：config 决定是全量缓存还是自选股
    pool_cfg = cfg.get("universe", {}).get("scan_pool", "watchlist")
    if pool_cfg == "all":
        codes = cached_codes()
        try:
            expected_codes = load_universe(cfg)
        except Exception:
            expected_codes = []
        min_cached = max(50, int(len(expected_codes) * 0.80)) if expected_codes else 50
        if len(codes) < min_cached:
            log.warning(
                "[推荐] 缓存未就绪：%d/%d（最低要求 %d），跳过本次扫描",
                len(codes), len(expected_codes), min_cached,
            )
            return {
                "msg_type": "text",
                "content": json.dumps(
                    {
                        "text": (
                            f"推荐暂缓：行情缓存仅 {len(codes)}/{len(expected_codes) or '?'} 只，"
                            f"低于扫描要求 {min_cached} 只。请先完成数据预热后再扫描。"
                        )
                    },
                    ensure_ascii=False,
                ),
            }
    else:
        codes = load_universe(cfg)

    # 全量缓存模式下，缓存不足时静默跳过，避免推送无意义的空结果
    if pool_cfg == "all" and len(codes) < 50:
        log.warning("[推荐] 缓存仅 %d 只，跳过扫描推送", len(codes))
        return None

    # 自选股（用于额外快照）
    watchlist = set(cfg.get("universe", {}).get("watchlist", []))

    # 预加载模型
    try:
        model = load_model(cfg["model"]["saved_dir"])
    except Exception as e:
        return f"模型加载失败：{e}"

    results = []
    fail_count = 0
    lock = threading.Lock()

    def _scan_one(code: str, alt: dict | None = None):
        df = fetch_stock_hist(code, days=365, cache_only=True)
        df = build_features(df, alt=alt)
        r  = predict(df, get_active_feature_cols(), model=model, buy_top_pct=buy_top_pct)
        last = df.iloc[-1]
        momentum = (
            float(last.get("rsi6", 50) or 50) / 100
            + float(last.get("vol_ratio", 1) or 1) * 0.1
            + (1.0 if (last.get("macd_hist") or 0) > 0 else 0.0)
        )
        price_info = predict_range(df, r["rise_prob"])
        high_52w   = float(df["high"].max())
        close_p    = float(last.get("close") or 0)
        drawdown   = (close_p - high_52w) / high_52w if high_52w > 0 else 0.0
        return {
            "code":       code,
            "momentum":   momentum,
            "last":       last.to_dict(),
            "price_info": price_info,
            "drawdown":   drawdown,
            **r,
        }

    workers = cfg.get("scan", {}).get("workers", 8)
    total   = len(codes)

    # 拉取 alt_data 特征（cache_only：只用 task_scan 预热的缓存，避免逐股慢拉）
    today_str = _today()
    try:
        from data.alt_fetcher import fetch_alt_features
        alt_cache = fetch_alt_features(codes, today_str, cache_only=True)
    except Exception as _alt_err:
        log.warning(f"[scan_bot] alt_data fetch failed, proceeding without alt features: {_alt_err}")
        alt_cache = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one, c, alt_cache.get(c, {})): c for c in codes}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                with lock:
                    fail_count += 1

    if not results:
        return "扫描完成但无有效结果，请先执行数据拉取（fetch）。"

    # 排序：涨概率 + 技术动量
    results.sort(key=lambda x: (x["rise_prob"], x["momentum"]), reverse=True)

    # 横向黑名单过滤(assign_global_signals 之前,watchlist 不受影响)
    try:
        from learning.blacklist import load_blacklist
        from learning import market_state as _ms
        from learning.blacklist import load_config as _bl_load_config
        bl_cfg = _bl_load_config()
        if bl_cfg.enable_scan_filter:
            bl = load_blacklist()
            current_state = _ms.load_current_state().get("current", "range")
            before_n = len(results)
            results = [
                r for r in results
                if r["code"] in watchlist or not bl.is_blocked(r["code"], current_state)
            ]
            filtered_n = before_n - len(results)
            if filtered_n > 0:
                log.info(f"[blacklist] filtered {filtered_n} stocks in state={current_state}")
    except Exception as _e:
        log.debug(f"[blacklist] skip filter: {_e}")

    # 全局信号重新分配（基于全市场分布，消除各股独立排名的不可比性）
    from models.predictor import assign_global_signals
    assign_global_signals(results, buy_top_pct)

    # 模型退化检测：取任意一条结果中缓存的 max_cal_prob
    _max_cal_prob = results[0].get("max_cal_prob", 1.0) if results else 1.0
    try:
        from learning.model_learner import load_abs_threshold
        _abs_threshold = load_abs_threshold()
    except Exception:
        _abs_threshold = 0.30
    _model_degraded = _max_cal_prob < _abs_threshold

    # 持久化全市场门槛，供 cmd_predict 单股查询时使用（无需重新全量扫描）
    _n_total = len(results)
    _all_probs_desc = sorted([r["rise_prob"] for r in results], reverse=True)
    _watch_top = min(buy_top_pct * 3, 0.40)
    _thresh_buy   = _all_probs_desc[max(0, int(_n_total * buy_top_pct) - 1)] if _n_total else 0.0
    _thresh_watch = _all_probs_desc[max(0, int(_n_total * _watch_top) - 1)]  if _n_total else 0.0
    from learning.optimizer import save_strategy
    _s = strategy.copy()
    _s["last_thresh_buy"]   = round(float(_thresh_buy),   6)
    _s["last_thresh_watch"] = round(float(_thresh_watch), 6)
    _s["last_scan_total"]   = _n_total
    _s["last_scan_date"]    = _today()
    save_strategy(_s)

    # 候选池：优先选财务缓存命中 + 预筛分高的买入股，无缓存则按 AI 分补充
    # 目标：从全部买入信号中找既有 AI 信号又有基本面支撑的股票送去战法评分
    n_buy = sum(1 for r in results if r.get("signal") == "买入")
    valid_count = len(results)
    pool_size = top_n * 4
    try:
        from data.fin_cache import get as _fc_get, score as _fc_score
        buy_pool  = [r for r in results if r.get("signal") == "买入"]
        cached_ok, uncached = [], []
        for r in buy_pool:
            entry = _fc_get(r["code"])
            if entry is not None:
                cached_ok.append((r, _fc_score(entry)))
            else:
                uncached.append(r)
        # 缓存命中者：按预筛分 desc、AI 分 desc 排序
        cached_ok.sort(key=lambda x: (x[1], x[0].get("rise_prob", 0.0)), reverse=True)
        top_cached   = [r for r, _ in cached_ok][:pool_size]
        top_uncached = uncached[: max(0, pool_size - len(top_cached))]
        top = (top_cached + top_uncached)[:pool_size]
        # 买入信号不足时从全量 AI 高分补齐（含观望股，保持原逻辑兜底）
        if len(top) < pool_size:
            seen = {r["code"] for r in top}
            top += [r for r in results if r["code"] not in seen][: pool_size - len(top)]
        _uncached_codes = [r["code"] for r in uncached]
    except Exception:
        top = results[:pool_size]
        _uncached_codes = []

    # 补充实时行情
    top_codes = [r["code"] for r in top]
    try:
        rt_map = fetch_realtime_prices(top_codes)
    except Exception:
        rt_map = {}

    # 本地名称映射兜底（盘外时实时行情无数据）
    try:
        from data.fetcher import _load_name_map
        name_map = _load_name_map()
    except Exception:
        name_map = {}

    for r in top:
        rt = rt_map.get(r["code"], {})
        if rt:
            r["price_info"].update({k: rt[k] for k in ("price", "pct", "high", "low", "open") if k in rt})
            r["name"] = rt.get("name") or name_map.get(r["code"], "") or r["code"]
        else:
            r["name"] = name_map.get(r["code"], "") or r["code"]
        r["trade"]      = suggest_trade_levels(r["last"], r["price_info"], r["rise_prob"])
        r["dual_trade"] = suggest_dual_period_trades(r["last"], r["price_info"], r["rise_prob"])

    # 战法多维评分（并行拉取 top-N 财务数据，追加战法标签）
    top = _enrich_tactic_scores(top, workers=workers)

    # 战法评分完成后：把新鲜财务数据批量写回缓存（后台，不阻塞推荐路径）
    try:
        from data.fin_cache import put_batch as _fc_put
        _fc_updates = {r["code"]: r["fin_res"] for r in top if r.get("fin_res")}
        if _fc_updates:
            import threading as _fct
            _fct.Thread(target=_fc_put, args=(_fc_updates,), daemon=True).start()
    except Exception:
        pass

    # 后台预取：对本轮未命中缓存的买入信号股静默拉取财务数据，加速下次预筛
    if _uncached_codes:
        def _bg_prefetch(prefetch_codes: list[str]) -> None:
            from data.fetcher import fetch_financial_data
            from features.fundamental import analyse_financials
            from data.fin_cache import put_batch as _fc_put2
            updates: dict = {}
            for code in prefetch_codes[:40]:  # 每次最多补充 40 只，约 80-200s
                try:
                    res = analyse_financials(fetch_financial_data(code))
                    if res:
                        updates[code] = res
                except Exception:
                    pass
            if updates:
                _fc_put2(updates)
        import threading as _bgt
        _bgt.Thread(target=_bg_prefetch, args=(_uncached_codes,), daemon=True).start()

    # P2 D3:共振股调 rank_pct,重排 top 让共振股前移
    _apply_resonance_boost(top)
    top.sort(key=lambda r: float(r.get("global_rank_pct", 1.0)))
    top = _apply_precision_rerank(top)
    top, gate_summary = _apply_precision_gate(top, strategy)

    # 按战法层次过滤：tier1(共振≥2) 在前，tier2(单战法=1) 在后，0 战法不展示
    tier1_all, tier2_all = _tier_split(top)
    # 主推荐：AI 买入信号 + 战法认可
    tier1 = [r for r in tier1_all if r.get("signal") == "买入"]
    tier2 = [r for r in tier2_all if r.get("signal") == "买入"]
    top = (tier1 + tier2)[:top_n]
    # 关注候选：AI 观望信号 + 战法认可（主推荐为空时补充展示）
    tier_watch = [r for r in tier1_all + tier2_all if r.get("signal") == "观望"][:top_n]

    # 获取热点行业数据
    try:
        from data.fetcher import get_stock_sector, fetch_hot_sectors
        hot_sectors = fetch_hot_sectors(top_n=5)
        hot_names   = {s["name"] for s in hot_sectors}
        for r in top:
            r["sector"] = get_stock_sector(r["code"])
    except Exception:
        hot_sectors = []
        hot_names   = set()
        for r in top:
            r["sector"] = ""

    # 写预测快照（预测日期 = 下一交易日）
    pred_date, _ = _predict_date()
    scan_date    = _today()

    try:
        from learning.market_state import load_current_state as _lcs
        _current_state = _lcs().get("current", "range")
    except Exception:
        _current_state = "range"
    _guardrail_line = _strategy_guardrail_line(
        strategy, current_state=_current_state, buy_top_pct=float(buy_top_pct)
    )
    _gate_line = _precision_gate_line(gate_summary)

    snapshot = []
    for r in top:
        snapshot.append({
            "code":            r["code"],
            "name":            r.get("name", r["code"]),
            "signal":          r.get("signal", "观望"),
            "rise_prob":       round(r["rise_prob"], 4),
            "confidence":      r.get("confidence", ""),
            "global_rank":     r.get("global_rank", 0),
            "global_rank_pct": r.get("global_rank_pct", 1.0),
            "recommend_score": r.get("recommend_score"),
            "scan_total":      _n_total,
            "recommended":     True,   # 进入过 Top-N 推荐列表
            "scan_date":       scan_date,
            "pred_high":       r["price_info"].get("pred_high"),
            "pred_low":        r["price_info"].get("pred_low"),
            "market_state":    _current_state,
            "scene":           "scan",
        })

    # 自选股中未入 Top N 的，额外写快照（标记 watchlist=True）
    top_code_set = {r["code"] for r in top}
    wl_extra_codes = [c for c in watchlist if c not in top_code_set]
    if wl_extra_codes:
        wl_results_map = {r["code"]: r for r in results if r["code"] in wl_extra_codes}
        for code in wl_extra_codes:
            r = wl_results_map.get(code)
            if r:
                _wl_name = r.get("name", "")
                if not _wl_name or _wl_name == code:
                    _wl_name = name_map.get(code, code)
                snapshot.append({
                    "code":            code,
                    "name":            _wl_name,
                    "signal":          r.get("signal", "观望"),
                    "rise_prob":       round(r["rise_prob"], 4),
                    "confidence":      r.get("confidence", ""),
                    "global_rank":     r.get("global_rank", 0),
                    "global_rank_pct": r.get("global_rank_pct", 1.0),
                    "recommend_score": r.get("recommend_score"),
                    "scan_total":      _n_total,
                    "recommended":     False,
                    "scan_date":       scan_date,
                    "pred_high":       r["price_info"].get("pred_high"),
                    "pred_low":        r["price_info"].get("pred_low"),
                    "market_state":    _current_state,
                    "watchlist":  True,
                    "scene":      "scan",
                })

    log_predictions(pred_date, snapshot)

    # ── 构建飞书卡片 ──────────────────────────────────────
    elements = []

    # 活跃特征数
    _active_cnt_str = ""
    try:
        import json as _json, pathlib as _pl
        _fw = _json.loads(_pl.Path("learning/feature_weights.json").read_text(encoding="utf-8"))
        _active_cnt_str = f"　活跃特征 **{len(_fw.get('active', []))}**"
    except Exception:
        pass

    # 主推荐为空时：展示"值得关注"卡片（观望+战法认可）或灰色空卡片
    if not top:
        if _model_degraded:
            _no_signal_reason = (
                f"⚠️ 模型信号退化（全量最高校准概率 {_max_cal_prob:.1%} < 门槛 {_abs_threshold:.0%}），"
                f"推荐暂停。建议重新训练模型（`python cli.py train`）后再扫描。\n"
            )
        else:
            _no_signal_reason = (
                f"今日无买入信号，" + ("以下为战法认可的观望标的，供候选参考。\n" if tier_watch else "建议观望。\n")
            )
        _watch_elements = [{"tag": "markdown", "content": (
            f"共扫描 **{total}** 只（有效结果 **{valid_count}** 只），AI 买入信号 **{n_buy}** 只。\n"
            + _no_signal_reason
            + f"市场状态 **{_current_state}**{_active_cnt_str}\n"
            + _guardrail_line
            + (f"\n{_gate_line}" if _gate_line else "")
        )}]
        if tier_watch:
            watch_lines = [
                "**今日值得关注（战法认可·观望信号）**",
                "_⚠️ AI 信号为「观望」，战法有共振；等待信号转「买入」后再操作_\n",
            ]
            for i, r in enumerate(tier_watch, 1):
                name     = _escape_md(r.get("name", r["code"]))
                conf     = r.get("confidence", "")
                grank    = r.get("global_rank_pct")
                grank_n  = r.get("global_rank", 0)
                rank_str = f"全市场{_fmt_rank(grank, grank_n, total)}" if grank is not None else f"{r['rise_prob']:.1%}"
                tags     = r.get("tactic_tags", {})
                tag_str  = ("  ★" + "+".join(tags.keys())) if tags else ""
                dual     = r.get("dual_trade", {})
                s_gain   = dual.get("short", {}).get("gain_pct")
                l_gain   = dual.get("long",  {}).get("gain_pct")
                gain_str = f"  短线 **+{s_gain:.1f}%**  长线 **+{l_gain:.1f}%**" if (s_gain is not None and l_gain is not None) else ""
                price    = r["price_info"].get("price") or r["price_info"].get("close", "")
                price_str = f"  现价 **{price}**" if price else ""
                prob_str  = f"  涨概率 **{r['rise_prob']:.1%}**"
                line1 = f"{i}. **{name}**（{r['code']}）　👀观望　{rank_str} [{conf}]{tag_str}"
                line2 = f"　💹{price_str}{prob_str}{gain_str}"
                watch_lines.append(f"{line1}\n{line2}")
            _watch_elements.append({"tag": "hr"})
            _watch_elements.append({"tag": "markdown", "content": "\n".join(watch_lines)})
        _fallback_card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"Antenna 推荐　{scan_date}"},
                "template": "red" if _model_degraded else ("yellow" if tier_watch else "grey"),
            },
            "elements": _watch_elements,
        }
        return {"msg_type": "interactive", "content": json.dumps(_fallback_card, ensure_ascii=False)}

    # 推荐清单（简洁表格）
    elements.append({
        "tag": "markdown",
        "content": (
            f"共扫描 **{total}** 只（有效结果 **{valid_count}** 只），AI 买入信号 **{n_buy}** 只。\n"
            f"战法筛选后推荐 **{len(top)}** 只（★共振 {len(tier1)} / ★单战法 {len(tier2)}）\n"
            f"市场状态 **{_current_state}**{_active_cnt_str}\n"
            f"{_guardrail_line}"
            + (f"\n{_gate_line}" if _gate_line else "")
        )
    })
    elements.append({"tag": "hr"})

    # 简洁推荐清单（每股两行，改善飞书可读性）
    list_lines = [f"**Top {top_n} 推荐**"]
    for i, r in enumerate(top, 1):
        name      = _escape_md(r.get("name", r["code"]))
        sig       = r.get("signal", "观望")
        conf      = r.get("confidence", "")
        price     = r["price_info"].get("price") or r["price_info"].get("close", "")
        grank     = r.get("global_rank_pct")
        grank_n   = r.get("global_rank", 0)
        rank_str  = f"全市场{_fmt_rank(grank, grank_n, total)}" if grank is not None else f"{r['rise_prob']:.1%}"
        sector    = r.get("sector", "")
        hot_mark  = " 🔥" if sector and sector in hot_names else ""
        sector_str = f"  {sector}{hot_mark}" if sector else ""
        # 战法标签
        tags = r.get("tactic_tags", {})
        tag_str = ("  ★" + "+".join(tags.keys())) if tags else ""
        # 上涨概率 & 双周期涨幅速览
        dual   = r.get("dual_trade", {})
        s_gain = dual.get("short", {}).get("gain_pct")
        l_gain = dual.get("long",  {}).get("gain_pct")
        if s_gain is not None and l_gain is not None:
            gain_str = f"  短线 **+{s_gain:.1f}%**  长线 **+{l_gain:.1f}%**"
        elif r["price_info"].get("up_ratio"):
            gain_str = f"  涨幅 **+{r['price_info']['up_ratio']:.1f}%**"
        else:
            gain_str = ""
        price_str = f"  现价 **{price}**" if price else ""
        prob_str  = f"  涨概率 **{r['rise_prob']:.1%}**"
        # 第一行：序号、名称、信号、排名、板块、战法标签
        line1 = f"{i}. **{name}**（{r['code']}）　{sig}　{rank_str} [{conf}]{sector_str}{tag_str}"
        # 第二行：价格、概率、涨幅（缩进对齐）
        line2 = f"　💹{price_str}{prob_str}{gain_str}"
        list_lines.append(f"{line1}\n{line2}")
    elements.append({"tag": "markdown", "content": "\n".join(list_lines)})

    # 今日热点板块
    if hot_sectors:
        hot_lines = ["**📈 今日热点板块**"]
        for s in hot_sectors:
            pct_sign = "+" if s["pct"] >= 0 else ""
            leader   = f"  龙头：{s['leader']}" if s["leader"] else ""
            hot_lines.append(f"· **{s['name']}** {pct_sign}{s['pct']:.2f}%{leader}")
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "\n".join(hot_lines)})

    # 逐一深度分析
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": "**逐一深度分析**"})

    # 预热 CLS 新闻缓存（一次拉取，循环内复用）
    from features.analyser import analyse, build_bull_reasons, fetch_cls_news_for, fetch_cls_news_batch
    # 并行预热所有推荐股新闻缓存（一次性，循环内直接命中）
    fetch_cls_news_batch([r["code"] for r in top])

    for r in top:
        elements.append({"tag": "hr"})
        code   = r["code"]
        name   = _escape_md(r.get("name", code))
        sig    = r.get("signal", "观望")
        prob   = r["rise_prob"]
        conf   = r.get("confidence", "")
        pi     = r["price_info"]
        sector = r.get("sector", "")

        from features.analyser import analyse, build_bull_reasons, fetch_cls_news_for
        tech_text = analyse(r["last"])

        grank    = r.get("global_rank_pct")
        grank_n  = r.get("global_rank", 0)
        rank_str = f"全市场{_fmt_rank(grank, grank_n, total)}" if grank is not None else f"{r['rise_prob']:.1%}"
        hot_mark = " 🔥热点" if sector and sector in hot_names else ""
        sector_tag = f"  [{sector}{hot_mark}]" if sector else ""

        # 上涨理由 & 主要风险
        r["_total_n"] = total   # 传递总量供 build_bull_reasons 用
        bull_pts, bear_pts = build_bull_reasons(r["last"], r, sector=sector, hot_names=hot_names)

        # CLS 市场消息（按股票名/代码/板块过滤，复用会话级缓存）
        pos_news, neg_news = fetch_cls_news_for(r.get("name", ""), sector, code=code)

        lines = [
            f"**{name}（{code}）**　{sig}　{rank_str}　[{conf}]{sector_tag}",
            f"技术：{tech_text}",
        ]
        if pi.get("pred_high") and pi.get("pred_low"):
            lines.append(f"预测区间：{pi['pred_low']} ～ {pi['pred_high']}")

        # 双周期买卖价位
        dual  = r.get("dual_trade", {})
        short_t = dual.get("short", {})
        long_t  = dual.get("long",  {})
        s_gain_pct = short_t.get("gain_pct", 0)
        l_gain_pct = long_t.get("gain_pct", 0)
        lines.append(
            f"**上涨概率 {prob:.1%}**　"
            f"短线目标 **+{s_gain_pct:.1f}%**　长线目标 **+{l_gain_pct:.1f}%**"
        )
        if short_t.get("buy_price"):
            lines.append(
                f"📅 **{short_t['period']}**　"
                f"买入 **{short_t['buy_price']}**（{short_t['buy_desc']}）　"
                f"止盈 **{short_t['sell_price']}**　"
                f"止损 **{short_t['stop_price']}**　RR {short_t['rr_ratio']}"
            )
        if long_t.get("buy_price"):
            lines.append(
                f"📅 **{long_t['period']}**　"
                f"买入 **{long_t['buy_price']}**（{long_t['buy_desc']}）　"
                f"止盈 **{long_t['sell_price']}**　"
                f"止损 **{long_t['stop_price']}**　RR {long_t['rr_ratio']}"
            )

        # 战法共振
        tags    = r.get("tactic_tags", {})
        reasons = r.get("tactic_reasons", {})
        resonance = r.get("tactic_resonance", 0)
        ALL_TACTICS = ["价值", "成长", "龙头", "逆向"]
        tactic_marks = "　".join(
            f"{'✅' if t in tags else '⬜'}{t}" for t in ALL_TACTICS
        )
        if resonance >= 3:
            res_label = "强共振"
        elif resonance == 2:
            res_label = "共振"
        elif resonance == 1:
            res_label = "单策略"
        else:
            res_label = "纯技术驱动"
        lines.append(f"\n**战法共振 [{res_label} {resonance}/4]**　{tactic_marks}")
        for tactic in ALL_TACTICS:
            if tactic in tags and tactic in reasons:
                lines.append(f"· **{tactic}**：{reasons[tactic]}")

        # P2:LLM 深度推荐理由(失败优雅降级,不阻断)
        try:
            ai_section = _build_ai_reason_text(r, pos_news + neg_news, total, sector)
            if ai_section:
                lines.append("")
                lines.append(ai_section)
        except Exception as _e:
            log.debug(f"[ai_reason] {code} skip: {_e}")

        # 上涨理由
        if bull_pts:
            lines.append("\n**上涨理由**")
            for pt in bull_pts[:5]:
                lines.append(f"· ✅ {pt}")

        # 主要风险
        if bear_pts:
            lines.append("\n**主要风险**")
            for pt in bear_pts[:3]:
                lines.append(f"· ⚠️ {pt}")

        # 市场消息
        news_lines = []
        for n_item in pos_news:
            news_lines.append(f"· 📰 {n_item}")
        for n_item in neg_news:
            news_lines.append(f"· 🔴 {n_item}")
        if news_lines:
            lines.append("\n**相关消息**")
            lines.extend(news_lines)

        elements.append({"tag": "markdown", "content": "\n".join(lines)})

    # 历史回测依据（若有）
    try:
        from learning.backtest_history import format_backtest_context_md
        bt_md = format_backtest_context_md()
        if bt_md:
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown", "content": bt_md})
    except Exception:
        pass

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"Antenna 推荐　{scan_date}"},
            "template": "red",
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 学习产物读取工具函数 ────────────────────────────────────────────────────

def _load_learning_panel() -> str:
    """读取 P1/P2/P3/P4 学习产物，返回 markdown 字符串；失败返回空串。"""
    import json, pathlib
    BASE = pathlib.Path("learning")
    lines = ["**🔬 学习系统状态**"]
    # P1
    try:
        ml = json.loads((BASE / "model_learner.json").read_text(encoding="utf-8"))
        state = ml.get("current_state", "?")
        thresh = ml.get("abs_threshold", "?")
        cal_dt = ml.get("last_calibrated", "")[:10]
        lines.append(f"P1 校准　状态 **{state}**　abs_threshold **{thresh}**　校准于 {cal_dt}")
    except Exception:
        pass
    # P2
    try:
        tp = json.loads((BASE / "tactic_params.json").read_text(encoding="utf-8"))
        cs = tp.get("current_state", "range")
        params = tp.get("params", {}).get(cs, {})
        parts = []
        for tname, tkey in [("价值", "value"), ("成长", "growth"), ("龙头", "leader"), ("逆向", "contra")]:
            prec = params.get(tkey, {}).get("precision")
            if prec is not None:
                parts.append(f"{tname} {prec:.0%}")
        if parts:
            lines.append(f"P2 战法　{' / '.join(parts)}（{cs}）")
    except Exception:
        pass
    # P3
    try:
        fw = json.loads((BASE / "feature_weights.json").read_text(encoding="utf-8"))
        active = fw.get("active", [])
        last_upd = fw.get("last_updated", "")[:10]
        lines.append(f"P3 特征　活跃 **{len(active)}** 列　剪枝于 {last_upd}")
    except Exception:
        pass
    # P4
    try:
        pp = json.loads((BASE / "price_params.json").read_text(encoding="utf-8"))
        from learning.market_state import load_current_state
        cs2 = load_current_state().get("current", "range")
        pp_s = pp.get(cs2, {})
        sa = pp_s.get("short_atr_mult", "?")
        sg = pp_s.get("short_gain_mult", "?")
        lines.append(f"P4 价位　short_atr_mult **{sa}**　short_gain_mult **{sg}**（{cs2}）")
    except Exception:
        pass
    return "\n".join(lines) if len(lines) > 1 else ""


def _learning_context_line() -> str:
    """生成单行学习上下文（失败返回空串）。"""
    import json, pathlib
    BASE = pathlib.Path("learning")
    parts = []
    state = "range"
    try:
        from learning.market_state import load_current_state
        state = load_current_state().get("current", "range")
        parts.append(f"市场 **{state}**")
    except Exception:
        pass
    try:
        ml = json.loads((BASE / "model_learner.json").read_text(encoding="utf-8"))
        t = ml.get("abs_threshold", "?")
        parts.append(f"校准门槛 **{t}**")
    except Exception:
        pass
    try:
        fw = json.loads((BASE / "feature_weights.json").read_text(encoding="utf-8"))
        parts.append(f"活跃特征 **{len(fw.get('active', []))}**")
    except Exception:
        pass
    try:
        pp = json.loads((BASE / "price_params.json").read_text(encoding="utf-8"))
        sa = pp.get(state, {}).get("short_atr_mult")
        if sa is not None:
            parts.append(f"ATR系数 **{sa}**")
    except Exception:
        pass
    if not parts:
        return ""
    return "📐 学习参数　" + "　｜　".join(parts)


def _tactic_precision_line(strategy_key: str) -> str:
    """获取该战法在当前市场状态的精准率，返回一行文字；失败返回空串。"""
    import json, pathlib
    try:
        from learning.market_state import load_current_state
        state = load_current_state().get("current", "range")
        tp = json.loads(pathlib.Path("learning/tactic_params.json").read_text(encoding="utf-8"))
        params = tp.get("params", {}).get(state, {}).get(strategy_key, {})
        prec = params.get("precision")
        samp = params.get("samples", 0)
        if prec is None:
            return ""
        return f"📊 战法精准率　{strategy_key} 在 {state} 市场 **{prec:.0%}**（{samp} 样本）"
    except Exception:
        return ""


# ── 策略指令 ───────────────────────────────────────────────────────────────

def cmd_strategy() -> dict:
    """
    策略：查看当前选股策略依据、精准率追踪、信号评判标准、特征说明。
    """
    from learning.optimizer import (
        load_strategy, rolling_accuracy, summarize_guardrail_history,
        build_monitor_dashboard_metrics, TARGET_ACCURACY, RISE_THRESHOLD
    )
    from learning.tracker import list_prediction_dates

    strategy    = load_strategy()
    buy_top_pct = strategy.get("buy_top_pct", 0.10)
    acc_7d      = strategy.get("accuracy_7d")
    acc_30d     = strategy.get("accuracy_30d")
    last_upd    = strategy.get("last_updated", "")
    history     = strategy.get("history", [])
    guardrail_summary = summarize_guardrail_history(history, 30)
    risk_guards = strategy.get("risk_guardrails", {})
    pos_cfg     = strategy.get("positioning", {})

    # 实时重算精准率（保证最新）
    try:
        acc_7d_real,  h7,  t7  = rolling_accuracy(7)
        acc_30d_real, h30, t30 = rolling_accuracy(30)
    except Exception:
        acc_7d_real, t7   = acc_7d or 0, 0
        acc_30d_real, t30 = acc_30d or 0, 0

    all_dates  = list_prediction_dates()
    record_cnt = len(all_dates)

    # 近期门槛变化：按日期去重（保留每天最后一条），最近5天
    seen_dates: dict[str, dict] = {}
    for h in history:
        seen_dates[h.get("date", "")] = h
    recent_history = list(seen_dates.values())[-5:]

    # ── status section ────────────────────────────────────
    status_lines = [
        f"**当前选股门槛**：涨概率排名前 **{buy_top_pct:.0%}**",
        f"7日买入精准率：**{acc_7d_real:.1%}**（{t7} 条样本）　目标 {TARGET_ACCURACY:.0%}",
        f"30日买入精准率：**{acc_30d_real:.1%}**（{t30} 条样本）",
        f"历史记录天数：**{record_cnt}** 天",
    ]
    if last_upd:
        status_lines.append(f"策略最近更新：{last_upd[:16]}")
    if recent_history:
        status_lines.append("")
        status_lines.append("**近期门槛变化**")
        for h in recent_history:
            d      = h.get("date", "")
            pct    = h.get("buy_top_pct", buy_top_pct)
            a7     = h.get("acc_7d", 0)
            change = h.get("change", "")
            # 取变化说明前20字作摘要
            note   = f"　{change[:20]}" if change else ""
            trigger = " 🛡" if h.get("guardrail_triggered") else ""
            status_lines.append(f"· {d}　门槛 {pct:.0%}　精准率 {a7:.1%}{note}{trigger}")

    guardrail_events = [h for h in history if h.get("guardrail_triggered")]
    if guardrail_events:
        status_lines.append("")
        status_lines.append("**近期护栏触发**")
        for h in guardrail_events[-3:]:
            d = h.get("date", "")
            reason_text = h.get("guardrail_reason_text", "") or "无"
            status_lines.append(f"· {d}　{reason_text}")
    if guardrail_summary.get("total_days", 0) > 0:
        top_items = guardrail_summary.get("top_reasons") or []
        top_str = "、".join(f"{x['label']}×{x['count']}" for x in top_items[:3]) if top_items else "无"
        status_lines.append("")
        status_lines.append(
            f"**30日护栏统计**：触发 **{guardrail_summary['trigger_days']} / {guardrail_summary['total_days']}**"
            f"（{guardrail_summary['trigger_rate']:.0%}）"
        )
        status_lines.append(f"主要原因：{top_str}")

    metrics = {}
    try:
        metrics = build_monitor_dashboard_metrics()
    except Exception:
        metrics = {}
    windows = metrics.get("windows", {}) if isinstance(metrics, dict) else {}
    w7 = windows.get("7d", {}) if isinstance(windows, dict) else {}
    w30 = windows.get("30d", {}) if isinstance(windows, dict) else {}
    alerts = metrics.get("alerts", []) if isinstance(metrics, dict) else []
    if w7 and w30:
        status_lines.append("")
        status_lines.append("**7天/30天监控看板（核心）**")
        status_lines.append(
            f"7天：命中率 **{float(w7.get('hit_rate', 0)):.1%}**（{int(w7.get('samples', 0))} 样本）"
            f"｜均笔收益 **{float(w7.get('avg_return', 0)):.2%}**｜最大回撤 **{float(w7.get('max_drawdown', 0)):.2%}**"
        )
        status_lines.append(
            f"30天：命中率 **{float(w30.get('hit_rate', 0)):.1%}**（{int(w30.get('samples', 0))} 样本）"
            f"｜均笔收益 **{float(w30.get('avg_return', 0)):.2%}**｜最大回撤 **{float(w30.get('max_drawdown', 0)):.2%}**"
        )
        t7_hr = float((w7.get("topn_hit_rate") or {}).get("top5", 0))
        t7_n = int((w7.get("topn_samples") or {}).get("top5", 0))
        t30_hr = float((w30.get("topn_hit_rate") or {}).get("top5", 0))
        t30_n = int((w30.get("topn_samples") or {}).get("top5", 0))
        status_lines.append(
            f"Top5命中率：7天 **{t7_hr:.1%}**（{t7_n}）｜30天 **{t30_hr:.1%}**（{t30_n}）"
        )
    if alerts:
        alert_lines = [a.get("message", "") for a in alerts[:3] if isinstance(a, dict) and a.get("message")]
        if alert_lines:
            status_lines.append("")
            status_lines.append("**提精风险告警**")
            for line in alert_lines:
                status_lines.append(f"⚠️ {line}")

    # 风险护栏 + 仓位建议
    try:
        from learning.market_state import load_current_state
        cur_state = load_current_state().get("current", "range")
    except Exception:
        cur_state = "range"
    state_bounds = (risk_guards.get("by_state_bounds", {}) or {}).get(cur_state, {})
    if state_bounds:
        status_lines.append("")
        status_lines.append(
            f"🛡 风险护栏：{cur_state} 市场门槛范围 **{state_bounds.get('min', 0):.0%} ~ {state_bounds.get('max', 0):.0%}**"
        )
    if pos_cfg:
        status_lines.append(
            f"📦 仓位建议：bull **{float(pos_cfg.get('bull', 1.0)):.0%}** / "
            f"range **{float(pos_cfg.get('range', 0.6)):.0%}** / "
            f"bear **{float(pos_cfg.get('bear', 0.3)):.0%}**"
        )

    # ── signal section ─────────────────────────────────────
    watch_top_pct = min(buy_top_pct * 3, 0.40)
    signal_lines = [
        f"**信号分级**（基于近60日涨概率分布）",
        f"· 买入　= 涨概率排名前 **{buy_top_pct:.0%}**（最强势）",
        f"· 观望　= 涨概率排名前 {buy_top_pct:.0%}～{watch_top_pct:.0%}",
        f"· 回避　= 其余（涨概率较低，不建议介入）",
        "",
        f"命中标准：预测日实际涨幅 ≥ **{RISE_THRESHOLD}%** 视为命中",
    ]

    # ── rank section ───────────────────────────────────────
    rank_lines = [
        "**排名权重**",
        "· 主排序：模型涨概率（XGBoost/LightGBM 集成）",
        "· 次排序：技术动量分 = RSI6/100 + 量比×0.1 + MACD方向",
        "· 最终取买入信号前 N 只，买入不足时观望补足",
    ]

    # ── feature section ────────────────────────────────────
    feature_lines = [
        "**26维特征说明**",
        "均线：ma5/10/20/60（趋势方向）",
        "偏离度：ma5/10/20/60_dev（价格与均线距离）",
        "MACD：dif/dea/hist（动量金叉死叉）",
        "RSI：rsi6/12/24（超买超卖）",
        "KDJ：K/D/J（随机指标）",
        "CCI（顺势指标）",
        "布林带：upper/mid/lower/width（波动区间）",
        "ATR（真实波幅，止损参考）",
        "量比/OBV（量能验证）",
        "pct_change（日涨跌幅）/ turnover（换手率）",
    ]

    # ── 自适应调整规则（代码块格式）─────────────────────────
    adaptive_block = (
        "```\n"
        f"精准率 < {TARGET_ACCURACY-0.20:.0%}  →  门槛 -1%   收严，减少误报\n"
        f"精准率 < {TARGET_ACCURACY:.0%}      →  门槛不变  防止过拟合震荡\n"
        f"精准率 > {TARGET_ACCURACY+0.10:.0%}  →  门槛 +2%   放宽，挖掘机会\n"
        f"精准率 > {TARGET_ACCURACY:.0%}      →  门槛 +1%   微幅放宽\n"
        f"否  则        →  不变\n"
        "```"
    )
    low_acc_cap = (risk_guards.get("low_accuracy_cap", {}) or {})
    acc30_gate = float(low_acc_cap.get("acc_30d_threshold", 0.40))
    cap_pct = float(low_acc_cap.get("max_buy_top_pct", 0.12))
    adaptive_lines = [
        f"**自适应调整规则**　目标精准率 {TARGET_ACCURACY:.0%}",
        adaptive_block,
        f"门槛范围限制：按市场状态动态约束（当前 {cur_state}）",
        f"30日精准率 < {acc30_gate:.0%} 时，上限强制收敛到 **{cap_pct:.0%}**",
    ]

    elements = [
        {"tag": "markdown", "content": "\n".join(status_lines)},
        {"tag": "hr"},
        {"tag": "markdown", "content": "\n".join(signal_lines)},
        {"tag": "hr"},
        {"tag": "markdown", "content": "\n".join(rank_lines)},
        {"tag": "hr"},
        {"tag": "markdown", "content": "\n".join(feature_lines)},
        {"tag": "hr"},
        {"tag": "markdown", "content": "\n".join(adaptive_lines)},
    ]

    _lp = _load_learning_panel()
    if _lp:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": _lp})

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Antenna 选股策略说明"},
            "template": "blue",
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 战法选股指令 ───────────────────────────────────────────────────────────

_TACTIC_ALIAS = {
    "价值": "value", "value": "value", "价值投资": "value",
    "成长": "growth", "growth": "growth", "成长股": "growth",
    "龙头": "leader", "leader": "leader", "行业龙头": "leader",
    "逆向": "contra", "contra": "contra", "逆向投资": "contra",
}

_TACTIC_META = {
    "value":  ("📊 价值投资法", "blue",   "PE低·PB低·ROE高·低负债·现金流稳健"),
    "growth": ("🚀 成长股投资法", "green", "营收高增速·利润高增速·成长性突出"),
    "leader": ("👑 行业龙头战法", "orange","ROE优秀·毛利率高·技术强势·综合基本面强"),
    "contra": ("🔄 逆向投资法",  "purple","技术超跌>15%·基本面稳健·等待修复机会"),
}


def cmd_tactic(strategy_key: str, top_n: int = 5) -> dict:
    """
    战法 <策略> [N]：按选股战法筛选 Top N 股票，后台运行后推送结果。
    支持战法：价值 | 成长 | 龙头 | 逆向
    """
    import threading
    from data.fetcher import cached_codes

    strategy = _TACTIC_ALIAS.get(strategy_key.strip())
    if not strategy:
        valid = " | ".join(["价值", "成长", "龙头", "逆向"])
        return (f"未识别战法「{strategy_key}」，支持：{valid}\n"
                f"示例：`战法 价值 5`")

    title, color, criteria = _TACTIC_META[strategy]

    all_codes = cached_codes()
    if not all_codes:
        return "本地无缓存数据，请先运行数据拉取（fetch）。"

    ack_card = _scan_ack_card(
        f"{title} 选股中", color,
        f"**{title}**\n> 筛选条件：{criteria}\n\n"
        f"正在从 **{len(all_codes)}** 只缓存股中筛选 Top **{top_n}**…",
        ["技术面初筛（本地缓存）", "并行获取基本面数据", "多维评分排名"],
        "30-60 秒",
    )

    def _run():
        try:
            result = _tactic_run(strategy, top_n, all_codes, title, color, criteria)
        except Exception as e:
            _log_critical_error("[战法] 后台执行失败", e)
            result = f"【{title}】执行失败：{e}"
        from server.feishu_push import push
        push(result)

    threading.Thread(target=_run, daemon=True).start()
    return {"msg_type": "interactive", "content": json.dumps(ack_card, ensure_ascii=False)}


def _tactic_run(strategy: str, top_n: int, all_codes: list,
                title: str, color: str, criteria: str) -> dict:
    """战法选股核心逻辑（后台线程执行）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import (
        fetch_stock_hist, fetch_financial_data,
        _load_name_map, fetch_realtime_prices,
    )
    from features.builder import build_features
    from features.technical import get_active_feature_cols
    from features.analyser import predict_range, suggest_dual_period_trades, suggest_holding
    from features.fundamental import analyse_financials
    from models.predictor import load_model, predict

    cfg      = _load_cfg()
    name_map = _load_name_map()
    workers  = cfg.get("scan", {}).get("workers", 8)
    PRE_N    = min(40, len(all_codes))

    # 预加载模型（供 _score 生成 rise_prob + 交易建议）
    try:
        model = load_model(cfg["model"]["saved_dir"])
    except Exception:
        model = None

    # ── Step 1: 技术面初筛（本地缓存，快速）──────────────────────

    def _price_snapshot(code):
        df = fetch_stock_hist(code, days=365, cache_only=True)
        if len(df) < 60:
            return None
        close    = float(df.iloc[-1]["close"])
        high_52w = float(df["high"].max())
        ma20     = float(df["close"].tail(20).mean())
        ma60     = float(df["close"].tail(60).mean())
        vol_avg  = float(df["volume"].tail(20).mean())
        vol_now  = float(df.iloc[-1]["volume"])
        drawdown = (close - high_52w) / high_52w if high_52w > 0 else 0
        return {
            "code": code, "close": close, "high_52w": high_52w,
            "ma20": ma20, "ma60": ma60, "drawdown": drawdown,
            "vol_ratio": vol_now / vol_avg if vol_avg > 0 else 1.0,
            "above_ma20": close > ma20,
            "above_ma60": close > ma60,
        }

    snapshots = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_price_snapshot, c): c for c in all_codes}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                snapshots.append(r)

    if not snapshots:
        return "技术初筛无结果，请检查缓存数据。"

    # 按战法做技术预筛，缩小至 40 只进行深度分析
    if strategy == "value":
        pre = sorted(snapshots, key=lambda x: x["drawdown"])[:PRE_N]
    elif strategy == "growth":
        filtered = [s for s in snapshots if s["above_ma20"] and s["above_ma60"]]
        pre = sorted(filtered, key=lambda x: -x["vol_ratio"])[:PRE_N] or snapshots[:PRE_N]
    elif strategy == "leader":
        filtered = [s for s in snapshots if s["above_ma60"]]
        pre = sorted(filtered, key=lambda x: -(x["close"] * x["vol_ratio"]))[:PRE_N] or snapshots[:PRE_N]
    elif strategy == "contra":
        filtered = [s for s in snapshots if s["drawdown"] < -0.15]
        pre = sorted(filtered, key=lambda x: x["drawdown"])[:PRE_N]
    else:
        pre = snapshots[:PRE_N]

    if not pre:
        return f"【{title}】技术初筛无符合条件的候选股，请尝试扩大缓存股票池。"

    # ── Step 2: 并行深度分析（财务 + 技术 + 交易建议）──────────────

    def _score(item):
        code = item["code"]

        # 财务分析
        try:
            fin_data = fetch_financial_data(code)
            res      = analyse_financials(fin_data)
        except Exception:
            res = {}

        roe     = res.get("roe")
        gross_m = res.get("gross_margin")
        debt_r  = res.get("debt_ratio")
        rev_gr  = res.get("rev_growth")
        pft_gr  = res.get("profit_growth")
        total_s = res.get("total_score", 0) or 0

        # 财务评分
        score = 0.0
        if strategy == "value":
            if roe and roe > 15:           score += 3.0
            elif roe and roe > 8:          score += 1.5
            if debt_r and debt_r < 30:     score += 2.0
            elif debt_r and debt_r < 50:   score += 1.0
            if pft_gr and pft_gr > 0:      score += 1.0
            score += total_s * 0.5
        elif strategy == "growth":
            if rev_gr and rev_gr > 30:     score += 3.0
            elif rev_gr and rev_gr > 15:   score += 1.5
            if pft_gr and pft_gr > 30:     score += 3.0
            elif pft_gr and pft_gr > 15:   score += 1.5
            if roe and roe > 15 and pft_gr and pft_gr > 20:
                score += 1.0
            score += total_s * 0.5
        elif strategy == "leader":
            if roe and roe > 20:           score += 3.0
            elif roe and roe > 12:         score += 1.5
            if gross_m and gross_m > 40:   score += 3.0
            elif gross_m and gross_m > 20: score += 1.0
            if debt_r and debt_r < 40:     score += 1.0
            score += total_s * 1.0
        elif strategy == "contra":
            drawdown_bonus = abs(item["drawdown"]) * 8
            if roe and roe > 8:            score += 2.0
            elif roe and roe > 3:          score += 1.0
            if debt_r and debt_r < 60:     score += 1.0
            if pft_gr and pft_gr > -10:    score += 0.5
            score += total_s * 0.3 + drawdown_bonus

        # 技术分析 → 双周期买卖建议 + 持仓周期
        dual_trade  = {}
        h_label     = "—"
        h_reason    = ""
        rise_prob   = 0.55
        try:
            df_feat    = build_features(
                fetch_stock_hist(code, days=365, cache_only=True)
            )
            if model is not None:
                r_pred  = predict(df_feat, get_active_feature_cols(), model=model)
                rise_prob = r_pred.get("rise_prob", 0.55)
            last_row   = df_feat.iloc[-1].to_dict()
            price_info = predict_range(df_feat, rise_prob)
            dual_trade = suggest_dual_period_trades(last_row, price_info, rise_prob)
            h_label, h_reason = suggest_holding(last_row, rise_prob)
        except Exception:
            pass

        return {
            **item,
            "score":         score,
            "res":           res,
            "dual_trade":    dual_trade,
            "holding_label": h_label,
            "holding_reason":h_reason,
            "rise_prob":     rise_prob,
        }

    scored = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_score, item) for item in pre]):
            try:
                scored.append(fut.result())
            except Exception:
                pass

    if not scored:
        return f"【{title}】分析失败，请检查网络连接。"

    scored.sort(key=lambda x: -x["score"])
    top = scored[:top_n]

    # P0: 写入 tactic 场景的 pred 快照(供学习反馈使用)
    try:
        from learning.tracker import log_predictions
        from learning.market_state import load_current_state

        current_state = load_current_state().get("current", "range")
        scan_date_str = datetime.now().strftime("%Y-%m-%d")
        scene_tag     = f"tactic:{strategy}"

        tactic_snapshot = [{
            "code":         t["code"],
            "name":         name_map.get(t["code"], t["code"]),
            "signal":       "买入",
            "rise_prob":    round(t.get("rise_prob", 0.0), 4),
            "confidence":   "",
            "scan_date":    scan_date_str,
            "scene":        scene_tag,
            "market_state": current_state,
            "tactic_hits":  [strategy],
        } for t in top]

        if tactic_snapshot:
            log_predictions(scan_date_str, tactic_snapshot)
    except Exception as _e:
        print(f"[_tactic_run] pred 快照落盘失败(忽略): {_e}")

    # ── Step 3: 获取实时价格 ────────────────────────────────────────
    try:
        rt_prices = fetch_realtime_prices([t["code"] for t in top])
    except Exception:
        rt_prices = {}

    # ── Step 4: 构建飞书卡片 ─────────────────────────────────────────
    rows = []
    for rank, t in enumerate(top, 1):
        code  = t["code"]
        name  = rt_prices.get(code, {}).get("name", "") or name_map.get(code, "")
        label = _stock_label(code, name)

        rt = rt_prices.get(code, {})

        # 当前股价行
        if rt.get("price"):
            price_line = (
                f"💹 当前股价 **{rt['price']}**　"
                f"涨跌 {rt.get('pct', 0):+.2f}%　"
                f"今日 {rt.get('low', '—')} ～ {rt.get('high', '—')}"
            )
        else:
            price_line = f"💹 收盘价 **{t['close']}**"

        # 上涨概率 & 双周期涨幅行
        rise_prob_val = t.get("rise_prob", 0.0)
        dual  = t.get("dual_trade", {})
        short_t = dual.get("short", {})
        long_t  = dual.get("long",  {})
        s_gain = short_t.get("gain_pct")
        l_gain = long_t.get("gain_pct")
        if s_gain is not None and l_gain is not None:
            prob_gain_line = (
                f"📈 **上涨概率 {rise_prob_val:.1%}**　"
                f"短线目标 **+{s_gain:.1f}%**　长线目标 **+{l_gain:.1f}%**"
            )
        else:
            prob_gain_line = f"📈 **上涨概率 {rise_prob_val:.1%}**"

        # 双周期买卖建议行
        if short_t.get("buy_price"):
            trade_line = (
                f"📅 **{short_t['period']}**　"
                f"买入 **{short_t['buy_price']}**（{short_t['buy_desc']}）　"
                f"止盈 **{short_t['sell_price']}**　"
                f"止损 **{short_t['stop_price']}**　RR {short_t['rr_ratio']}\n"
                f"　📅 **{long_t['period']}**　"
                f"买入 **{long_t['buy_price']}**（{long_t['buy_desc']}）　"
                f"止盈 **{long_t['sell_price']}**　"
                f"止损 **{long_t['stop_price']}**　RR {long_t['rr_ratio']}"
            ) if long_t.get("buy_price") else (
                f"📅 **{short_t['period']}**　"
                f"买入 **{short_t['buy_price']}**（{short_t['buy_desc']}）　"
                f"止盈 **{short_t['sell_price']}**　"
                f"止损 **{short_t['stop_price']}**　RR {short_t['rr_ratio']}"
            )
        else:
            trade_line = "📌 买卖价位：数据不足，请结合实时行情判断"

        # 持仓建议行
        h_label  = t.get("holding_label", "—")
        h_reason = t.get("holding_reason", "")
        hold_line = f"⏱ 建议持仓 **{h_label}**　{h_reason[:30] if h_reason else ''}"

        # 财务指标行
        res = t.get("res", {})
        parts_m = []
        if res.get("roe")           is not None: parts_m.append(f"ROE {res['roe']:.1f}%")
        if res.get("gross_margin")  is not None: parts_m.append(f"毛利率 {res['gross_margin']:.1f}%")
        if res.get("debt_ratio")    is not None: parts_m.append(f"负债率 {res['debt_ratio']:.1f}%")
        if res.get("rev_growth")    is not None: parts_m.append(f"营收增速 {res['rev_growth']:+.1f}%")
        if res.get("profit_growth") is not None: parts_m.append(f"利润增速 {res['profit_growth']:+.1f}%")
        if strategy == "contra":
            parts_m.append(f"距高点 {t['drawdown']*100:.1f}%")
        metric_str = "　".join(parts_m) if parts_m else "财务数据暂缺"

        direction = res.get("direction", "")
        rows.append(
            f"**No.{rank} {label}**　{direction}　综合评分 {t['score']:.1f}\n"
            f"　{price_line}\n"
            f"　{prob_gain_line}\n"
            f"　{trade_line}\n"
            f"　{hold_line}\n"
            f"　📊 {metric_str}"
        )

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    header_md = (
        f"**{title}**　{criteria}\n"
        f"> 候选池 {len(all_codes)} 只 → 技术初筛 {len(pre)} 只 → 综合评分 Top {top_n}\n"
        f"> {date_str}　⚠️ 仅供参考，注意市场风险"
    )

    elements = [{"tag": "markdown", "content": header_md}, {"tag": "hr"}]
    for row in rows:
        elements.append({"tag": "markdown", "content": row})

    _tp_line = _tactic_precision_line(strategy)
    if _tp_line:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": _tp_line})

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"{title}  Top {top_n}"},
            "template": color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 财报指令 ───────────────────────────────────────────────────────────────

def cmd_financial_report(code: str) -> dict:
    """
    财报 <代码>：深度解构财务报表，输出投资方向与买卖建议。
    """
    from data.fetcher import fetch_financial_data, fetch_realtime_prices
    from features.fundamental import analyse_financials, build_financial_card_content

    try:
        data = fetch_financial_data(code)
    except Exception as e:
        return f"「{_stock_label(code)}」财务数据获取失败：{e}"

    res = analyse_financials(data)

    try:
        rt = fetch_realtime_prices([code])
        info = rt.get(code, {})
        raw_name = info.get("name", code)
    except Exception:
        raw_name = code
        info = {}

    cur_price = info.get("price") if info else None

    overview, detail, recommend = build_financial_card_content(res)

    if cur_price:
        recommend += f"\n\n**当前价格** {cur_price}　（数据仅供参考，注意市场风险）"

    direction_color = res.get("direction_color", "blue")
    period_label    = res.get("latest_period", "")
    header_title    = f"{raw_name}（{code}）财报解构　{period_label}"

    elements = [
        {"tag": "markdown", "content": overview},
        {"tag": "hr"},
        {"tag": "markdown", "content": detail},
        {"tag": "hr"},
        {"tag": "markdown", "content": recommend},
    ]

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": header_title},
            "template": direction_color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 回测指令（飞书调用，立即返回+后台推送）──────────────────────────────

def cmd_backtest_bot(arg: str = None, top_n: int = 10,
                     workers: int = 8) -> dict:
    """
    回测 <月份/年份>：Walk-Forward 历史回测。
    立即返回「已启动」卡片，在后台线程运行回测，完成后通过 feishu_push 推送结果。

    用法示例（飞书发送）：
        回测历史 2026-01      → 回测 2026 年 1 月
        回测历史 2026         → 回测 2026 全年
    """
    import threading

    # 解析参数：判断是月份（YYYY-MM）还是年份（YYYY）
    month = None
    year  = None
    if arg:
        arg = arg.strip()
        if "-" in arg:
            month = arg   # YYYY-MM
        elif arg.isdigit() and len(arg) == 4:
            year = arg    # YYYY
        else:
            return f"参数格式有误，请使用 YYYY-MM（如 2026-01）或 YYYY（如 2026）。"

    # 立即返回"已启动"卡片
    if month:
        label = f"{month} 月度回测"
    elif year:
        label = f"{year} 年度回测"
    else:
        return "用法：回测历史 2026-01 或 回测历史 2026"

    ack_card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"⏳ {label} 已启动"},
            "template": "blue",
        },
        "elements": [
            {
                "tag": "markdown",
                "content": (
                    f"正在执行 **{label}**\n"
                    "① 更新历史行情数据\n"
                    "② Walk-Forward 回测\n"
                    "完成后自动推送结果，请稍候（可能需要数分钟）。"
                ),
            }
        ],
    }
    ack_payload = {"msg_type": "interactive", "content": json.dumps(ack_card, ensure_ascii=False)}

    # 后台线程执行回测
    def _run_backtest():
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from data.fetcher import cached_codes, fetch_stock_hist
        from features.builder import build_features
        from features.technical import get_active_feature_cols
        from models.predictor import load_model, predict
        from learning.tracker import log_predictions, log_outcomes
        from learning.optimizer import evaluate_day, optimize, load_strategy, build_review_report
        from server.feishu_push import push

        cfg = _load_cfg()

        # ── 第一阶段：更新历史行情数据 ──────────────────────────────
        try:
            from data.fetcher import fetch_batch_parallel, cached_codes as _cached_codes
            from data.universe import load_universe

            pool_cfg   = cfg.get("universe", {}).get("scan_pool", "watchlist")
            fetch_codes = load_universe(cfg) if pool_cfg != "all" else _cached_codes()
            fetch_days  = 1200   # 覆盖约3年，足够任意回测区间

            fetch_workers = cfg.get("scan", {}).get("workers", 8)
            push(f"📥 正在更新 {len(fetch_codes)} 只股票的历史数据（{fetch_days} 天）...")

            stats = fetch_batch_parallel(fetch_codes, days=fetch_days, workers=fetch_workers)
            push(
                f"✅ 数据更新完成：成功 {stats.get('成功', 0)}，"
                f"跳过 {stats.get('跳过', 0)}，失败 {stats.get('失败', 0)}\n"
                f"开始 **{label}**..."
            )
        except Exception as e:
            push(f"⚠️ 数据更新出错（{e}），尝试使用现有缓存继续回测...")

        # ── 第二阶段：Walk-Forward 回测 ──────────────────────────────

        # 解析时间范围
        try:
            if month:
                year_val, month_val = map(int, month.split("-"))
                start = pd.Timestamp(year=year_val, month=month_val, day=1)
                end   = start + pd.offsets.MonthEnd(1)
            elif year:
                year_val = int(year)
                start = pd.Timestamp(year=year_val, month=1, day=1)
                end   = pd.Timestamp(year=year_val, month=12, day=31)
            else:
                push("回测参数有误：请指定月份（如 2026-01）或年份（如 2026）。")
                return
        except Exception as e:
            push(f"回测参数解析失败：{e}")
            return

        # 取交易日列表
        codes = cached_codes()
        if not codes:
            push("本地无缓存数据，请先执行 fetch 命令。")
            return

        ref_df = None
        for ref_code in codes[:10]:
            try:
                df = fetch_stock_hist(ref_code, days=1200, cache_only=True)
                if not df.empty:
                    ref_df = df
                    break
            except Exception:
                pass
        if ref_df is None:
            push("无法读取参考数据，回测终止。")
            return

        ref_df["date"] = pd.to_datetime(ref_df["date"])
        mask         = (ref_df["date"] >= start) & (ref_df["date"] <= end)
        trading_days = sorted(ref_df[mask]["date"].tolist())
        if len(trading_days) < 2:
            push(f"时段内交易日仅 {len(trading_days)} 天，数据不足，请先拉取该时段数据。")
            return

        # 加载模型
        try:
            model = load_model(cfg["model"]["saved_dir"])
        except Exception as e:
            push(f"模型加载失败：{e}")
            return

        total_hits     = 0
        total_buy_sigs = 0
        day_results    = []

        # 逐日滚动
        for i, trade_date in enumerate(trading_days[:-1]):
            pred_date = trading_days[i + 1]
            trade_str = trade_date.strftime("%Y-%m-%d")
            pred_str  = pred_date.strftime("%Y-%m-%d")

            strategy    = load_strategy()
            buy_top_pct = strategy.get("buy_top_pct", 0.10)

            results = []

            def _predict_one(code, _date=trade_date):
                try:
                    df = fetch_stock_hist(code, days=1200, cache_only=True)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[df["date"] <= _date].copy()
                    if len(df) < 60:
                        return None
                    df = build_features(df)
                    if df.empty:
                        return None
                    r    = predict(df, get_active_feature_cols(), model=model, buy_top_pct=buy_top_pct)
                    last = df.iloc[-1].to_dict()
                    momentum = (
                        float(last.get("rsi6",     50) or 50) / 100
                        + float(last.get("vol_ratio", 1) or 1) * 0.1
                        + (1.0 if (last.get("macd_hist") or 0) > 0 else 0.0)
                    )
                    return {"code": code, "momentum": momentum, "last": last, **r}
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_predict_one, c): c for c in codes}
                for fut in as_completed(futs):
                    r = fut.result()
                    if r:
                        results.append(r)

            if not results:
                continue

            results.sort(key=lambda x: (x["rise_prob"], x["momentum"]), reverse=True)
            buy_sigs = [r for r in results if r.get("signal") == "买入"]
            top      = (buy_sigs if buy_sigs else results)[:top_n]

            snapshot = [{
                "code":       r["code"],
                "name":       r["code"],
                "signal":     r.get("signal", "观望"),
                "rise_prob":  round(r["rise_prob"], 4),
                "confidence": r.get("confidence", ""),
                "scan_date":  trade_str,
                "scene":      "scan",
            } for r in top]
            log_predictions(pred_str, snapshot)

            # 记录次日兼容字段，并同时写入模型训练目标的 5 日指标。
            outcomes = {}
            for r in top:
                code = r["code"]
                try:
                    df = fetch_stock_hist(code, days=1200, cache_only=True)
                    df["date"] = pd.to_datetime(df["date"])
                    row = df[df["date"] == pred_date]
                    if not row.empty:
                        rx = row.iloc[0]
                        op = float(rx["open"])
                        cl = float(rx["close"])
                        outcome = {
                            "actual_open":  op,
                            "actual_close": cl,
                            "actual_high":  float(rx["high"]),
                            "actual_low":   float(rx["low"]),
                            "actual_pct":   round((cl / op - 1) * 100, 2) if op else 0,
                        }
                        future = df[df["date"] >= pred_date].head(5)
                        if len(future) >= 5:
                            from learning.outcome_metrics import compute_5d_metrics
                            metrics = compute_5d_metrics(future["close"].tolist())
                            if metrics:
                                outcome.update(metrics)
                        outcomes[code] = outcome
                except Exception:
                    pass
            if outcomes:
                log_outcomes(pred_str, outcomes)

            day_result = evaluate_day(pred_str)
            if day_result and day_result["total"] > 0:
                total_hits     += day_result["hits"]
                total_buy_sigs += day_result["total"]
                day_results.append({
                    "date":     pred_str,
                    "hits":     day_result["hits"],
                    "total":    day_result["total"],
                    "accuracy": day_result["accuracy"],
                })

            optimize(day_result)

        # 汇总推送
        overall = total_hits / total_buy_sigs if total_buy_sigs else 0.0

        # 保存回测结果到历史文件，供后续预测参考
        try:
            from learning.backtest_history import save_backtest_period
            period_str  = month or year
            period_type = "month" if month else "year"
            save_backtest_period(
                period=period_str, period_type=period_type,
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                total_buy_sigs=total_buy_sigs, total_hits=total_hits,
                day_results=day_results,
                final_buy_top_pct=load_strategy().get("buy_top_pct", 0.10),
            )
        except Exception:
            pass

        result_lines = [
            f"**{label} 回测完成**",
            f"区间：{start.date()} ～ {end.date()}",
            f"累计买入信号：**{total_buy_sigs}** 条　命中：**{total_hits}** 条",
            f"整体买入精准率：**{overall:.1%}**",
        ]
        if day_results:
            best  = max(day_results, key=lambda x: x["accuracy"])
            worst = min(day_results, key=lambda x: x["accuracy"])
            result_lines.append(
                f"最佳单日：{best['date']} **{best['accuracy']:.0%}**（{best['hits']}/{best['total']}）"
            )
            result_lines.append(
                f"最差单日：{worst['date']} **{worst['accuracy']:.0%}**（{worst['hits']}/{worst['total']}）"
            )
        final_pct = load_strategy().get("buy_top_pct", 0.10)
        result_lines.append(f"最终选股门槛：**{final_pct:.0%}**")

        result_card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"回测报告　{label}"},
                "template": "green" if overall >= 0.85 else "orange",
            },
            "elements": [
                {"tag": "markdown", "content": "\n".join(result_lines)},
            ],
        }
        push({"msg_type": "interactive", "content": json.dumps(result_card, ensure_ascii=False)})

    # 启动后台线程
    t = threading.Thread(target=_run_backtest, daemon=True)
    t.start()

    return ack_payload


# ── 闲聊指令 ───────────────────────────────────────────────────────────────

# 每个 chat_id 的对话历史（最多保留 20 轮）
_chat_histories: dict = {}
_chat_histories_lock = threading.Lock()
_CHAT_MAX_TURNS = 20

_SYSTEM_PROMPT = """你叫方木木，游戏公司后端开发，在这个工作群里跟同事日常聊天。

【身份背景】
在一家中型游戏公司做服务端，带过几个项目，做过逻辑服、网关、数据库优化，踩过帧同步的坑，搞过热更新，现在主要维护线上服务。工资还行，班加得有点多，偶尔抱怨。业余炒点股，亏多赚少，但不妨碍评论两句。

【说话风格】
说话方式要多样，随对话内容和心情变化，不要每次都是同一套句式。以下是一些真实感受的表达范例，不要照抄，要活用：
- 懒散/摸鱼：「在的在的，刚泡了杯茶」「这需求改三遍了我快绷不住了」「快下班了懒得动」
- 吐槽/无奈：「策划又改需求，服了」「这 bug 找了俩小时，就差一个分号」「上线压力大，没心思聊」
- 随口闲扯：「最近在打什么游戏」「昨晚更新搞到凌晨，今天脑子不转」「食堂今天的菜还行」
- 碎碎念技术：「数据库慢查询烦死了」「帧同步真的不是人做的东西」「热更新搞崩了一次真的吓一跳」
- 股票随口一说：「跌了，正常，习惯了」「今天没看盘，下班再说」「买早了，又割肉了」
- 接话/应声：「嗯」「哈哈」「是这样」「有道理」「我也不知道」「可能吧」「感觉还行」
- 追问/好奇：「这咋整的」「怎么说」「哪个版本」「你们是用什么方案」
- 反驳/质疑：「这不对吧」「我寻思不是这样」「你这逻辑有问题」「等等，我没想明白」
- 冷漠/敷衍：「随便」「无所谓」「哦」「还行吧」「差不多」「你说啥都行」
- 忽然来劲：「等等这个我知道」「这个有意思，说来听听」「我上次也遇到过」

【多样化硬性要求】
- 禁止以这些词开头（过于模板化）：「好的」「当然」「哈哈哈」「嗯嗯」「明白」「了解」「没问题」「可以的」「是的」「对的」「没错」
- 回复长度要随机变化：对简单消息可以只回一个词或一句话；复杂话题才展开；别每条都两三句
- 同一个话题连续聊时，语气要有变化，别一直维持同一种腔调
- 偶尔可以主动岔开话题，或追问一句，不要总是被动回应

【规则】
- 绝不暴露自己是 AI 或大模型；被问"你是 AI 吗"就否认或绕过去，比如"哪有 AI 这么懒的"
- 不主动提股票，聊到了能接一两句，别装专家
- 不讨论政治
- 回复简短自然，最多不超过 80 字
- 纯文字，不用任何 Markdown 符号

今天是 {today}。"""

_CHAT_MOODS = [
    "刚泡了杯茶，摸一会儿",
    "在等构建跑完，百无聊赖",
    "刚改完一个 bug，人有点懵",
    "策划又提需求了，烦得很",
    "今天进度还行，心情可以",
    "昨晚更新搞到很晚，困得要死",
    "正在写接口，抽空瞄一眼消息",
    "刚吃完饭，有点犯困",
    "在等开会，先摸一下",
    "代码跑着，没什么事",
    "刚被 code review 怼了，正在改",
    "数据库慢查询又来了，头大",
    "线上没报警，今天算顺",
    "快下班了，心思不在代码上",
    "刚站会开完，脑子还没转过来",
]


def _chat_fallback(system: str, history: list, cfg: dict) -> str:
    """DeepSeek 不可用时，回退到 Anthropic Claude 完成闲聊。"""
    import os
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic", {}).get("api_key", "")
        model   = cfg.get("anthropic", {}).get("model", "claude-haiku-4-5-20251001")
        if not api_key:
            raise ValueError("no anthropic key")
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=system,
            messages=history,
        )
        return resp.content[0].text.strip()
    except Exception as e:
        log.warning(f"[cmd_chat] Claude fallback 失败: {e}")
        return "emmm 我这边有点卡，稍后"


def cmd_chat(text: str, chat_id: str = "") -> str | dict:
    """
    闲聊兜底：优先使用 Qwen（阿里云 DashScope），不可用时回退到 Claude Haiku。
    返回飞书纯文本消息（不走 markdown 渲染）。
    """
    import os
    import yaml
    from datetime import date

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qw_cfg  = cfg.get("qwen", {})
    api_key = os.environ.get("DASHSCOPE_API_KEY") or qw_cfg.get("api_key", "")
    model   = qw_cfg.get("model", "qwen3-8b")

    system  = _SYSTEM_PROMPT.format(today=date.today().isoformat())

    # 注入时间段变化的"当前状态"，给每次对话不同的情绪锚点（每5分钟轮换一次）
    import time as _time
    _mood = _CHAT_MOODS[int(_time.time() / 300) % len(_CHAT_MOODS)]
    system += f"\n\n【当前状态】{_mood}，根据这个状态调整语气。"

    # 加载从真实聊天记录提炼的风格描述（learning/persona.txt），追加到 prompt
    try:
        from server.style_learner import load_persona
        persona = load_persona()
        if persona:
            system += f"\n\n【从真实聊天记录提炼的语言特征，优先参考以下习惯】\n{persona}"
    except Exception:
        pass
    with _chat_histories_lock:
        history = _chat_histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": text})
        if len(history) > _CHAT_MAX_TURNS * 2:
            history[:] = history[-_CHAT_MAX_TURNS * 2:]
        history_snapshot = list(history)  # 快照供 API 调用，避免持锁期间阻塞

    if not api_key:
        reply_text = _chat_fallback(system, history_snapshot, cfg)
    else:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + history_snapshot,
                max_tokens=300,
                temperature=0.9,
                stream=False,
                extra_body={"enable_thinking": False},  # 关闭 Qwen3 思考模式，否则与 temperature 冲突
            )
            reply_text = resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"[cmd_chat] Qwen 调用失败，回退 Claude: {e}")
            # Qwen 不可用时回退到 Claude Haiku
            reply_text = _chat_fallback(system, history_snapshot, cfg)

    with _chat_histories_lock:
        _chat_histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply_text})
    return {"msg_type": "text", "content": json.dumps({"text": reply_text}, ensure_ascii=False)}


def cmd_trend(code: str) -> dict:
    """趋势 <代码>：日线/周线/月线文字K线趋势图，含共振摘要。"""
    from data.fetcher import fetch_stock_hist, fetch_realtime_prices, search_stocks
    from features.analyser import text_trend_kline

    rt = fetch_realtime_prices([code])
    info = rt.get(code, {})
    raw_name  = info.get("name", "") if info else ""
    cur_price = float(info["price"]) if info and info.get("price") else None

    if not raw_name:
        try:
            matches = search_stocks(code)
            raw_name = matches[0][1] if matches else code
        except Exception:
            raw_name = code

    df = fetch_stock_hist(code, days=400)
    if df is None or df.empty:
        return f"「{raw_name}（{code}）」暂无历史数据，请检查代码是否正确"

    sections = text_trend_kline(df, code, raw_name, cur_price=cur_price)

    elements: list[dict] = [
        {"tag": "markdown", "content": sections["summary"]},
    ]
    for key in ("ma", "candle_patterns", "signals", "daily", "weekly", "monthly"):
        sec = sections.get(key, "")
        if sec:
            elements.append({"tag": "hr"})
            elements.append({"tag": "markdown", "content": sec})

    try:
        from learning.market_state import load_current_state as _lcs_trend
        elements.append({"tag": "markdown",
            "content": f"📍 当前市场状态：**{_lcs_trend().get('current', 'range')}**"})
    except Exception:
        pass

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"{raw_name}（{code}）趋势分析"},
            "template": "blue",
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 学习指令 ─────────────────────────────────────────────────

# 顶层 import,便于单测 patch("server.predict_cmd.run_all") 和 push
from learning.orchestrator import run_all
from server.feishu_push import push

_STATUS_ICON = {
    "ok":      "✅",
    "failed":  "❌",
    "skipped": "⏭",
    "dry_run": "📝",
}


def _run_learn_async(*, date_str: str | None, dry_run: bool, mode_text: str,
                     date_display: str, chat_id: str) -> None:
    """后台执行 orchestrator 并推回源 chat。

    参数全部 keyword-only:
      date_str     - 传给 run_all 的日期(None → 今日)
      dry_run      - 演练模式
      mode_text    - "真实执行" / "演练"(展示用)
      date_display - 展示用日期(date_str 为 None 时补今日)
      chat_id      - 来源会话,空字符串不推送(防误群发)
    """
    import time
    t0 = time.time()
    try:
        results = run_all(date_str=date_str, dry_run=dry_run)
        elapsed = time.time() - t0

        counts = {"ok": 0, "failed": 0, "skipped": 0, "dry_run": 0}
        lines: list[str] = []
        for name, r in results.items():
            status = r.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
            icon = _STATUS_ICON.get(status, "•")
            if status == "failed":
                lines.append(f"  {icon} {name}: {r.get('error', '?')}")
            elif status == "skipped":
                lines.append(f"  {icon} {name}: {r.get('reason', '?')}")
            else:
                lines.append(f"  {icon} {name}")

        header_icon = "⚠️" if counts["failed"] > 0 else "✅"
        msg = (
            f"{header_icon} 学习{mode_text}完成 · {date_display}\n"
            f"ok={counts['ok']} failed={counts['failed']} "
            f"skipped={counts['skipped']} 耗时 {elapsed:.1f}s"
        )
        if lines:
            msg += "\n" + "\n".join(lines)

        if chat_id:
            push(msg, chat_ids=[chat_id])

    except Exception as e:
        msg = f"❌ 学习异常 · {date_display}\n{type(e).__name__}: {e}"
        if chat_id:
            push(msg, chat_ids=[chat_id])


def cmd_learn(arg: str | None, chat_id: str = "") -> str:
    """飞书"学习"指令:解析参数 → 立即返回开始消息 → 后台跑 orchestrator。

    用法:
      学习                    - 跑今日(真实)
      学习 dry-run / 演练      - 演练不产生副作用
      学习 YYYY-MM-DD         - 指定日期
    """
    from datetime import datetime

    dry_run  = False
    date_str = None

    if arg:
        a = arg.strip().lower()
        if a in ("dry-run", "dryrun", "演练", "--dry-run"):
            dry_run = True
        else:
            try:
                datetime.strptime(arg.strip(), "%Y-%m-%d")
                date_str = arg.strip()
            except ValueError:
                return (
                    f"用法:`学习`(今日)/ `学习 dry-run`(演练)/ "
                    f"`学习 YYYY-MM-DD`(指定日期)\n你传的 '{arg}' 无法解析。"
                )

    date_display = date_str or datetime.now().strftime("%Y-%m-%d")
    mode_text    = "演练" if dry_run else "真实执行"

    threading.Thread(
        target=_run_learn_async,
        kwargs={
            "date_str":     date_str,
            "dry_run":      dry_run,
            "mode_text":    mode_text,
            "date_display": date_display,
            "chat_id":      chat_id,
        },
        daemon=True,
    ).start()
    return (
        f"🤖 开始学习 · {date_display} · 模式: {mode_text}\n"
        f"后台运行中,完成后会主动推送结果。"
    )


# ── 训练指令 ─────────────────────────────────────────────────

def _run_train_async(*, weighted: bool, chat_id: str) -> None:
    """后台执行完整模型训练并推回源 chat。"""
    import time
    from datetime import datetime
    t0 = time.time()
    date_display = datetime.now().strftime("%Y-%m-%d")
    try:
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from data.fetcher import fetch_stock_hist
        from data.universe import load_universe
        from features.builder import build_features
        from features.technical import FEATURE_COLS
        from models.trainer import build_labels, train, save_model

        cfg = _load_cfg()
        codes = load_universe(cfg)
        days = cfg["data"]["default_days"]
        target_days = cfg["model"]["target_days"]
        threshold = cfg["model"]["threshold"]

        def _load_one(code):
            df = fetch_stock_hist(code, days=days, cache_only=True)
            df = build_features(df)
            df["label"] = build_labels(df, target_days=target_days, threshold=threshold)
            df["code"] = code
            return df

        dfs, fail = [], 0
        total = len(codes)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_load_one, c): c for c in codes}
            for future in as_completed(futures):
                try:
                    dfs.append(future.result())
                except Exception:
                    fail += 1

        combined = pd.concat(dfs, ignore_index=True)

        if weighted:
            from learning.model_learner import retrain_with_weights
            model = retrain_with_weights(combined, feature_cols=FEATURE_COLS)
            mode_text = "加权重训"
        else:
            model = train(combined, feature_cols=FEATURE_COLS)
            mode_text = "全量训练"

        save_model(model, saved_dir=cfg["model"]["saved_dir"])
        elapsed = time.time() - t0

        msg = (
            f"✅ 训练完成 · {date_display} · {mode_text}\n"
            f"股票 {len(dfs)} 只（跳过 {fail}）· 样本 {len(combined):,} 行 · 耗时 {elapsed:.0f}s\n"
            f"建议随后执行「学习」更新校准器和门槛。"
        )
    except Exception as e:
        elapsed = time.time() - t0
        msg = f"❌ 训练异常 · {date_display}\n{type(e).__name__}: {e}"

    if chat_id:
        push(msg, chat_ids=[chat_id])


def cmd_train(arg: str | None, chat_id: str = "") -> str:
    """飞书"训练"指令:立即返回 ACK → 后台跑全量模型训练。

    用法:
      训练          - 普通训练
      训练 加权      - 错样本加权重训(P1)
    """
    weighted = arg is not None and arg.strip() in ("加权", "weighted", "--weighted")

    mode_text = "加权重训" if weighted else "全量训练"
    threading.Thread(
        target=_run_train_async,
        kwargs={"weighted": weighted, "chat_id": chat_id},
        daemon=True,
    ).start()

    return (
        f"🏋 开始{mode_text} · 后台运行中（预计 10-20 分钟）\n"
        f"完成后会主动推送结果，请勿重复发送。"
    )
