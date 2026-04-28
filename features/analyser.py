"""
analyser.py - 根据最新一行技术指标生成简要分析文字，供飞书推送使用。
"""
import re
import pandas as pd

_SPARK8 = "▁▂▃▄▅▆▇█"   # 8 级 sparkline 字符


def _to_spark(values: list) -> str:
    """将数值列表转为等长 sparkline 字符串（8 级）。"""
    vals = [float(v) for v in values if v is not None and v == v]
    if not vals:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    chars = []
    for v in vals:
        idx = round((v - mn) / rng * 7) if rng else 3
        chars.append(_SPARK8[max(0, min(7, idx))])
    return "".join(chars)



def text_intraday_kline(df_intraday: pd.DataFrame, code: str,
                        name: str, price_info: dict) -> str:
    """
    生成文字版分时 K 线摘要，用于飞书推送（无需图片权限）。

    布局：
      标题行     股票名  现价  涨跌幅
      涨跌轴     -10%~+10% 横轴，◆标记当前涨跌幅，直观显示强弱
      日内温度   低~高横轴，◆标记现价位置，显示日内运行位置
      价格走势   分时收盘价 sparkline
      量能走势   分时成交量 sparkline
      关键价位   开/现/高/低
      预测区间   横轴标记现价在预测区间中的位置
    """
    cur    = price_info.get("price")
    pct    = price_info.get("pct", 0.0)
    d_open = price_info.get("open")
    d_high = price_info.get("high")
    d_low  = price_info.get("low")
    p_high = price_info.get("pred_high")
    p_low  = price_info.get("pred_low")

    arrow = "▲" if pct >= 0 else "▼"
    # 转义 markdown 特殊字符（*ST 等名称会触发斜体）
    safe_name = name.replace("*", "＊").replace("_", "\\_") if name else name
    label = f"{safe_name}（{code}）" if safe_name and safe_name != code else code
    BAR   = 20

    # ST/*ST 股涨跌停幅度为 ±20%，普通股为 ±10%
    is_st   = name and ("ST" in name.upper())
    pct_lim = 20.0 if is_st else 10.0

    # ── 涨跌幅横轴 ────────────────────────────────────────
    pct_clamped = max(-pct_lim, min(pct_lim, pct if pct is not None else 0.0))
    pct_pos     = (pct_clamped + pct_lim) / (2 * pct_lim)   # 0.0 ~ 1.0
    mid_idx     = BAR // 2                               # 0% 对应的格子（第10格）
    cur_idx     = max(0, min(BAR - 1, round(pct_pos * (BAR - 1))))

    pct_chars = ["░"] * BAR
    pct_chars[mid_idx] = "|"    # 0% 基准线
    pct_chars[cur_idx] = "◆"    # 当前涨跌幅位置
    if pct >= 0 and cur_idx > mid_idx:
        for i in range(mid_idx + 1, cur_idx):
            pct_chars[i] = "█"
    elif pct < 0 and cur_idx < mid_idx:
        for i in range(cur_idx + 1, mid_idx):
            pct_chars[i] = "█"
    pct_bar  = "".join(pct_chars)
    lim_str  = f"{pct_lim:.0f}%"
    pct_line = f"`-{lim_str} ▕{pct_bar}▏ +{lim_str}`  {arrow}{abs(pct):.2f}%"

    # ── 日内价格温度计（低~高，◆标记现价）────────────────
    range_line = ""
    if d_high and d_low and cur and d_high > d_low:
        pos    = (cur - d_low) / (d_high - d_low)
        fill   = max(0, min(BAR, round(pos * BAR)))
        day_bar = "█" * fill + "░" * (BAR - fill)
        range_line = f"`{d_low:.2f} ▕{day_bar}▏ {d_high:.2f}`  ◆{cur}"

    # ── 价格与量能 sparkline ──────────────────────────────
    price_bar  = ""
    volume_bar = ""
    if df_intraday is not None and not df_intraday.empty:
        closes  = df_intraday["close"].tolist()
        volumes = df_intraday["volume"].tolist()
        t0 = str(df_intraday["time"].iloc[0])[11:16]   # HH:MM
        t1 = str(df_intraday["time"].iloc[-1])[11:16]

        spark_p = _to_spark(closes)
        spark_v = _to_spark(volumes)
        price_bar  = f"`{t0} ▕{spark_p}▏ {t1}`"
        volume_bar = f"`{t0} ▕{spark_v}▏ {t1}`"

    # ── 关键价位 ──────────────────────────────────────────
    stats_parts = []
    if d_open:  stats_parts.append(f"开 {d_open}")
    if cur:     stats_parts.append(f"现 **{cur}**")
    if d_high:  stats_parts.append(f"高 {d_high}")
    if d_low:   stats_parts.append(f"低 {d_low}")
    stats_line = "  ".join(stats_parts)

    # ── 预测区间横轴 ──────────────────────────────────────
    pred_line = ""
    if p_low and p_high and cur:
        total_rng = p_high - p_low
        cur_pos   = (cur - p_low) / total_rng if total_rng else 0.5
        cur_idx2  = max(0, min(BAR - 1, round(cur_pos * (BAR - 1))))
        pred_bar  = "─" * cur_idx2 + "◆" + "─" * (BAR - cur_idx2 - 1)
        pred_line = f"预测  {p_low} `{pred_bar}` {p_high}"

    # ── 拼装 ──────────────────────────────────────────────
    lines = [
        f"**{label}**  {cur}  {arrow} {abs(pct):.2f}%",
        f"涨跌  {pct_line}",
    ]
    if range_line:
        lines.append(f"日内  {range_line}")
    if price_bar:
        lines.append(f"价格  {price_bar}")
    if volume_bar:
        lines.append(f"量能  {volume_bar}")
    if stats_line:
        lines.append(f"  {stats_line}")
    if pred_line:
        lines.append(pred_line)

    return "\n".join(lines)


def analyse(row) -> str:
    """
    输入 DataFrame 最后一行（Series 或 dict），
    返回 2-4 个关键信号拼成的中文短句。
    """
    def _f(key, default=0.0):
        v = row.get(key, default)
        return float(v) if v is not None and v == v else default  # NaN → default

    points = []

    # ── MACD ──────────────────────────────────────────
    hist = _f("macd_hist")
    dif  = _f("macd_dif")
    dea  = _f("macd_dea")
    if dif > dea and hist > 0:
        points.append("MACD金叉")
    elif dif < dea and hist < 0:
        points.append("MACD死叉")

    # ── RSI ───────────────────────────────────────────
    rsi6 = _f("rsi6", 50)
    if rsi6 >= 75:
        points.append(f"RSI超买({rsi6:.0f})")
    elif rsi6 <= 25:
        points.append(f"RSI超卖({rsi6:.0f})")
    else:
        points.append(f"RSI={rsi6:.0f}")

    # ── 均线位置 ──────────────────────────────────────
    dev5  = _f("ma5_dev")
    dev20 = _f("ma20_dev")
    if dev5 > 0 and dev20 > 0:
        points.append("站上MA5/20")
    elif dev5 < 0 and dev20 < 0:
        points.append("跌破MA5/20")
    elif dev5 > 0 > dev20:
        points.append("MA5上穿20酝酿")

    # ── 量比 ──────────────────────────────────────────
    vol = _f("vol_ratio", 1.0)
    if vol >= 2.5:
        points.append(f"量比{vol:.1f}x大幅放量")
    elif vol >= 1.5:
        points.append(f"量比{vol:.1f}x温和放量")
    elif vol <= 0.4:
        points.append("明显缩量")

    # ── KDJ ───────────────────────────────────────────
    k = _f("kdj_k", 50)
    d = _f("kdj_d", 50)
    j = _f("kdj_j", 50)
    if k > d and j > 80:
        points.append("KDJ高位金叉")
    elif k < d and j < 20:
        points.append("KDJ低位死叉待反弹")

    # ── 布林带 ────────────────────────────────────────
    bb_upper = _f("bb_upper")
    bb_lower = _f("bb_lower")
    close    = _f("close") or _f("收盘")
    if bb_upper and close:
        if close > bb_upper:
            points.append("突破布林上轨")
        elif bb_lower and close < bb_lower:
            points.append("跌破布林下轨")

    # ── CCI ───────────────────────────────────────────
    cci = _f("cci")
    if cci > 150:
        points.append(f"CCI强势({cci:.0f})")
    elif cci < -150:
        points.append(f"CCI弱势({cci:.0f})")

    return "、".join(points[:4]) if points else "指标中性"


def predict_range(df: pd.DataFrame, rise_prob: float = 0.5) -> dict:
    """
    基于历史振幅分布预测当日高低点。

    原理：
    - 统计近 30 个交易日 (high-open)/open 和 (open-low)/open 的中位数
    - 以今日开盘价为基准，乘以相应振幅比率估算区间
    - 根据 rise_prob 对区间做偏移（看涨 → 高点上抬，低点上移）

    Returns: {"open": float, "close": float,
              "pred_high": float, "pred_low": float,
              "up_ratio": float, "down_ratio": float}
    """
    if df is None or len(df) < 10:
        return {}

    recent = df.tail(31).copy()
    # 需要 open/high/low/close 列
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(set(recent.columns)):
        return {}

    hist = recent.iloc[:-1]  # 用前30天统计振幅
    up_ratios   = ((hist["high"] - hist["open"]) / hist["open"]).clip(0, 0.1)
    down_ratios = ((hist["open"] - hist["low"])  / hist["open"]).clip(0, 0.1)

    up_med   = float(up_ratios.median())
    down_med = float(down_ratios.median())

    today = recent.iloc[-1]
    open_price  = float(today["open"])
    close_price = float(today["close"])

    # 根据涨概率微调（±20% 的振幅调整）
    bias = (rise_prob - 0.5) * 0.4   # -0.2 ~ +0.2
    pred_high = open_price * (1 + up_med   * (1 + bias))
    pred_low  = open_price * (1 - down_med * (1 - bias))

    return {
        "open":       round(open_price, 2),
        "close":      round(close_price, 2),
        "pred_high":  round(pred_high, 2),
        "pred_low":   round(pred_low, 2),
        "up_ratio":   round(up_med * 100, 2),
        "down_ratio": round(down_med * 100, 2),
    }


def build_commentary(df: pd.DataFrame, r: dict, price_info: dict) -> tuple[str, str]:
    """
    根据技术指标 + 模型结果生成：
      - 关键点评（叙述性 1-3 句）
      - 涨跌幅依据（历史振幅 + 风险收益）

    Returns: (commentary_md, basis_md)
    """
    if df is None or df.empty:
        return "数据不足", ""

    last = df.iloc[-1].to_dict()

    def _f(key, default=0.0):
        v = last.get(key, default)
        return float(v) if v is not None and v == v else default

    rise_prob  = r.get("rise_prob", 0.5)
    signal     = r.get("signal", "观望")

    # ── 关键点评 ──────────────────────────────────────────
    parts = []

    # 趋势倾向
    if rise_prob >= 0.65:
        parts.append("多头信号占优，短线偏多")
    elif rise_prob <= 0.35:
        parts.append("空头信号偏强，短线偏空")
    else:
        parts.append("多空信号均衡，方向待确认")

    # MACD
    dif, dea, hist = _f("macd_dif"), _f("macd_dea"), _f("macd_hist")
    if dif > dea and hist > 0:
        parts.append("MACD 金叉向上，动能增强")
    elif dif < dea and hist < 0:
        parts.append("MACD 死叉向下，动能偏弱")

    # RSI
    rsi = _f("rsi6", 50)
    if rsi >= 75:
        parts.append(f"RSI({rsi:.0f}) 超买，注意高位回调风险")
    elif rsi <= 25:
        parts.append(f"RSI({rsi:.0f}) 超卖，存在超跌反弹机会")
    elif rsi >= 58:
        parts.append(f"RSI({rsi:.0f}) 偏强，动能尚可")
    elif rsi <= 42:
        parts.append(f"RSI({rsi:.0f}) 偏弱，承压明显")

    # 均线
    dev5, dev20 = _f("ma5_dev"), _f("ma20_dev")
    if dev5 > 0 and dev20 > 0:
        parts.append("价格站上 MA5/MA20，均线多头排列")
    elif dev5 < 0 and dev20 < 0:
        parts.append("价格跌破 MA5/MA20，均线空头排列")
    elif dev5 > 0 > dev20:
        parts.append("价格在 MA5 上方但仍受 MA20 压制")

    # 量比
    vol = _f("vol_ratio", 1.0)
    if vol >= 2.5:
        parts.append(f"量比 {vol:.1f}x，大幅放量，需关注方向确认")
    elif vol >= 1.5:
        parts.append(f"量比 {vol:.1f}x，温和放量，人气尚可")
    elif vol <= 0.4:
        parts.append("明显缩量，观望情绪浓厚")

    commentary = "；".join(parts[:4]) + "。"

    # ── 涨跌幅依据 ────────────────────────────────────────
    up_ratio   = price_info.get("up_ratio", 0)
    down_ratio = price_info.get("down_ratio", 0)
    pred_high  = price_info.get("pred_high")
    pred_low   = price_info.get("pred_low")
    cur        = price_info.get("price") or price_info.get("close")

    basis_lines = [
        f"近 30 日振幅中位：上方 **+{up_ratio:.1f}%**，下方 **-{down_ratio:.1f}%**",
    ]

    if pred_high and pred_low and cur and cur > 0:
        up_space   = (pred_high - cur) / cur * 100
        down_risk  = (cur - pred_low)  / cur * 100
        basis_lines.append(
            f"预测区间 [{pred_low}, {pred_high}]，"
            f"上方空间 **+{up_space:.1f}%**，下方风险 **-{down_risk:.1f}%**"
        )
        if down_risk > 0:
            rr = up_space / down_risk
            rr_desc = "性价比较高" if rr >= 1.5 else ("下行风险偏大" if rr <= 0.7 else "风险收益均衡")
            basis_lines.append(f"风险收益比约 **{rr:.1f}:1**，{rr_desc}")

    # 涨概率解读
    if rise_prob >= 0.6:
        bias_desc = f"模型给出 **{rise_prob:.1%}** 上涨概率，偏向看多"
    elif rise_prob <= 0.4:
        bias_desc = f"模型给出 **{rise_prob:.1%}** 上涨概率，偏向看空"
    else:
        bias_desc = f"模型给出 **{rise_prob:.1%}** 上涨概率，方向中性"
    basis_lines.append(bias_desc)

    basis = "\n".join(basis_lines)
    return commentary, basis


def suggest_trade_levels(last: dict, price_info: dict, rise_prob: float) -> dict:
    """
    基于技术指标 + 振幅历史给出具体买卖价位建议。

    Returns: {
        buy_price, buy_desc,    — 建议买入价 & 说明
        sell_price, sell_desc,  — 建议卖出价（止盈）& 说明
        stop_price, stop_desc,  — 建议止损价 & 说明
        rr_ratio,               — 风险收益比
    }
    """
    def _f(key, default=0.0):
        v = last.get(key, default)
        return float(v) if v is not None and v == v else default

    cur       = price_info.get("price") or price_info.get("close", 0)
    pred_high = price_info.get("pred_high")
    pred_low  = price_info.get("pred_low")
    up_ratio  = (price_info.get("up_ratio") or 2.0) / 100
    down_ratio = (price_info.get("down_ratio") or 1.5) / 100

    ma5      = _f("ma5")
    ma20     = _f("ma20")
    bb_upper = _f("bb_upper")
    bb_lower = _f("bb_lower")
    atr      = _f("atr")

    if not cur or cur <= 0:
        return {}

    # ── 买入价：MA5 支撑 > 预测低点 > 现价 ──────────────
    if ma5 and 0 < ma5 <= cur * 1.02:        # MA5 在现价附近或下方
        buy_price = round(ma5 * 0.998, 2)    # MA5 下方 0.2%，接近支撑挂单
        buy_desc  = f"MA5 支撑 {buy_price}"
    elif bb_lower and bb_lower < cur:
        buy_price = round(bb_lower * 1.003, 2)
        buy_desc  = f"布林下轨 {buy_price}"
    elif pred_low and pred_low < cur:
        buy_price = round(pred_low * 1.005, 2)
        buy_desc  = f"预测低点 {buy_price}"
    else:
        buy_price = round(cur, 2)
        buy_desc  = f"现价 {buy_price}"

    # ── 卖出价：取 BB 上轨 / pred_high 中较小（保守目标）──
    targets = []
    if bb_upper and bb_upper > cur:
        targets.append(("布林上轨", bb_upper))
    if pred_high and pred_high > cur:
        targets.append(("预测高点", pred_high))

    if targets:
        label, val = min(targets, key=lambda x: x[1])
        sell_price = round(val, 2)
        sell_desc  = f"{label} {sell_price}"
    else:
        sell_price = round(cur * (1 + up_ratio), 2)
        sell_desc  = f"历史振幅目标 {sell_price}"

    # 止盈不低于买入价 +1%
    if sell_price <= buy_price * 1.01:
        sell_price = round(buy_price * (1 + up_ratio), 2)
        sell_desc  = f"振幅目标 {sell_price}"

    # ── 止损价：ATR × 1.5 或 3%，取宽者，上限 5% ────────
    pct_stop = buy_price * 0.97
    atr_stop = (buy_price - atr * 1.5) if atr else None

    if atr_stop and atr_stop > pct_stop:
        stop_price = round(atr_stop, 2)
        stop_desc  = f"ATR×1.5 {stop_price}"
    else:
        stop_price = round(pct_stop, 2)
        stop_desc  = f"固定3% {stop_price}"

    # 若 MA20 提供更紧的止损则采用
    if ma20 and stop_price < ma20 < buy_price:
        stop_price = round(ma20 * 0.995, 2)
        stop_desc  = f"MA20 下方 {stop_price}"

    # ── 风险收益比 ─────────────────────────────────────────
    risk   = buy_price - stop_price
    reward = sell_price - buy_price
    rr     = round(reward / risk, 1) if risk > 0 else 0.0

    return {
        "buy_price":  buy_price,
        "buy_desc":   buy_desc,
        "sell_price": sell_price,
        "sell_desc":  sell_desc,
        "stop_price": stop_price,
        "stop_desc":  stop_desc,
        "rr_ratio":   rr,
    }


def suggest_dual_period_trades(last: dict, price_info: dict, rise_prob: float) -> dict:
    """
    分别为短线（1-5日）和长线（2-4周）给出买卖价位建议。

    短线：MA5 / BB下轨入场，BB上轨 / pred_high 止盈，ATR×1.5 / 2% 止损
    长线：MA20 入场，MA60 / 扩展目标止盈，MA60下方 / 5% 止损

    Returns: {
        "short": {buy_price, buy_desc, sell_price, sell_desc, stop_price, stop_desc, rr_ratio, gain_pct},
        "long":  {buy_price, buy_desc, sell_price, sell_desc, stop_price, stop_desc, rr_ratio, gain_pct},
    }
    """
    def _f(key, default=0.0):
        v = last.get(key, default)
        return float(v) if v is not None and v == v else default

    def _rr(buy, sell, stop):
        risk   = buy - stop
        reward = sell - buy
        return round(reward / risk, 1) if risk > 0 else 0.0

    cur       = price_info.get("price") or price_info.get("close", 0)
    pred_high = price_info.get("pred_high")
    pred_low  = price_info.get("pred_low")
    up_ratio  = (price_info.get("up_ratio") or 2.0) / 100

    ma5      = _f("ma5")
    ma20     = _f("ma20")
    ma60     = _f("ma60")
    bb_upper = _f("bb_upper")
    bb_lower = _f("bb_lower")
    atr      = _f("atr")

    if not cur or cur <= 0:
        return {"short": {}, "long": {}}

    # ── 短线（1-5 日）────────────────────────────────────────
    # 买入：MA5 / BB下轨 / 预测低点（近支撑精准挂单）
    if ma5 and 0 < ma5 <= cur * 1.02:
        s_buy      = round(ma5 * 0.998, 2)
        s_buy_desc = f"MA5支撑 {s_buy}"
    elif bb_lower and bb_lower < cur:
        s_buy      = round(bb_lower * 1.003, 2)
        s_buy_desc = f"布林下轨 {s_buy}"
    elif pred_low and pred_low < cur:
        s_buy      = round(pred_low * 1.005, 2)
        s_buy_desc = f"预测低点 {s_buy}"
    else:
        s_buy      = round(cur, 2)
        s_buy_desc = f"现价 {s_buy}"

    # 卖出：BB上轨 / pred_high，取较近者（短线保守锁利）
    s_targets = []
    if bb_upper and bb_upper > cur:
        s_targets.append(("布林上轨", bb_upper))
    if pred_high and pred_high > cur:
        s_targets.append(("预测高点", pred_high))

    if s_targets:
        s_sell_label, s_sell_val = min(s_targets, key=lambda x: x[1])
        s_sell      = round(s_sell_val, 2)
        s_sell_desc = f"{s_sell_label} {s_sell}"
    else:
        s_sell      = round(s_buy * (1 + up_ratio), 2)
        s_sell_desc = f"短线振幅 {s_sell}"

    if s_sell <= s_buy * 1.005:
        s_sell      = round(s_buy * (1 + up_ratio), 2)
        s_sell_desc = f"振幅目标 {s_sell}"

    # 止损：ATR×1.5 或 2%（短线更紧）
    s_stop_pct = round(s_buy * 0.98, 2)
    s_stop_atr = round(s_buy - atr * 1.5, 2) if atr else None
    if s_stop_atr and s_stop_atr > s_stop_pct:
        s_stop      = s_stop_atr
        s_stop_desc = f"ATR×1.5 {s_stop}"
    else:
        s_stop      = s_stop_pct
        s_stop_desc = f"2%止损 {s_stop}"

    s_gain = round((s_sell - s_buy) / s_buy * 100, 1)
    short  = {
        "buy_price":  s_buy,  "buy_desc":  s_buy_desc,
        "sell_price": s_sell, "sell_desc": s_sell_desc,
        "stop_price": s_stop, "stop_desc": s_stop_desc,
        "rr_ratio":   _rr(s_buy, s_sell, s_stop),
        "gain_pct":   s_gain,
        "period":     "短线 1-5日",
    }

    # ── 长线（2-4 周）────────────────────────────────────────
    # 买入：MA20 支撑（中期趋势锚点，允许更宽入场区间）
    if ma20 and 0 < ma20 <= cur * 1.05:
        l_buy      = round(ma20 * 0.997, 2)
        l_buy_desc = f"MA20支撑 {l_buy}"
    elif ma5 and 0 < ma5 <= cur * 1.03:
        l_buy      = round(ma5 * 0.995, 2)
        l_buy_desc = f"MA5支撑 {l_buy}"
    else:
        l_buy      = round(cur * 0.99, 2)
        l_buy_desc = f"回调介入 {l_buy}"

    # 卖出：
    #   • 若股价在 MA60 下方 → MA60 是修复目标
    #   • 若股价在 MA60 上方 → 取 BB上轨 / 短线振幅×3 中较大者
    if ma60 and cur < ma60:
        l_sell      = round(ma60 * 0.995, 2)
        l_sell_desc = f"MA60目标 {l_sell}"
    else:
        l_candidates = []
        if bb_upper and bb_upper > cur * 1.03:
            l_candidates.append(("布林上轨", bb_upper))
        l_candidates.append(("长线目标", l_buy * (1 + up_ratio * 3)))
        _, l_sell_val = max(l_candidates, key=lambda x: x[1])
        l_sell      = round(l_sell_val, 2)
        if bb_upper and bb_upper > cur * 1.03 and l_sell_val == bb_upper:
            l_sell_desc = f"布林上轨 {l_sell}"
        else:
            l_sell_desc = f"长线目标 {l_sell}"

    if l_sell <= l_buy * 1.02:
        l_sell      = round(l_buy * (1 + up_ratio * 3), 2)
        l_sell_desc = f"长线目标 {l_sell}"

    # 止损：MA60 下方 或 5%（长线可承受较大回撤）
    if ma60 and 0 < ma60 < l_buy * 0.98:
        l_stop_raw  = round(ma60 * 0.98, 2)
        l_stop_desc = f"MA60下方 {l_stop_raw}"
        # 超过 8% 时收紧为 5%
        l_stop      = l_stop_raw if l_stop_raw >= l_buy * 0.92 else round(l_buy * 0.95, 2)
        if l_stop != l_stop_raw:
            l_stop_desc = f"5%止损 {l_stop}"
    else:
        l_stop      = round(l_buy * 0.95, 2)
        l_stop_desc = f"5%止损 {l_stop}"

    l_gain = round((l_sell - l_buy) / l_buy * 100, 1)
    long   = {
        "buy_price":  l_buy,  "buy_desc":  l_buy_desc,
        "sell_price": l_sell, "sell_desc": l_sell_desc,
        "stop_price": l_stop, "stop_desc": l_stop_desc,
        "rr_ratio":   _rr(l_buy, l_sell, l_stop),
        "gain_pct":   l_gain,
        "period":     "长线 2-4周",
    }

    return {"short": short, "long": long}


def suggest_holding(last: dict, rise_prob: float) -> tuple[str, str]:
    """
    根据技术指标 + 模型置信度推荐持仓周期。

    Returns:
        label  — "短线 1-3日" / "中期 5-10日" / "中长期 2-4周" / "长期 1月+"
        reason — 1-2 句支撑理由
    """
    def _f(key, default=0.0):
        v = last.get(key, default)
        return float(v) if v is not None and v == v else default

    rsi      = _f("rsi6", 50)
    dif      = _f("macd_dif")
    dea      = _f("macd_dea")
    hist     = _f("macd_hist")
    dev5     = _f("ma5_dev")
    dev20    = _f("ma20_dev")
    vol      = _f("vol_ratio", 1.0)
    cci      = _f("cci")
    k        = _f("kdj_k", 50)
    d        = _f("kdj_d", 50)
    bb_upper = _f("bb_upper")
    close    = _f("close") or _f("收盘")

    # ── 过热信号 → 短线 ──────────────────────────────────
    near_upper = bb_upper and close and close >= bb_upper * 0.99
    if rsi >= 78 or (near_upper and vol >= 2.0):
        return (
            "短线 1-3日",
            f"RSI={rsi:.0f} 超买{'，突破布林上轨放量' if near_upper and vol>=2 else ''}，"
            "短线获利压力大，不宜重仓久持。"
        )

    # ── 空头排列 / MACD 死叉 → 短线观望 ─────────────────
    macd_dead  = dif < dea and hist < 0
    bear_ma    = dev5 < 0 and dev20 < 0
    if macd_dead and bear_ma:
        return (
            "短线 1-3日",
            "MACD 死叉且价格跌破 MA5/MA20，趋势偏弱，建议轻仓短炒或等待信号改善。"
        )

    # ── 中长期 / 长期：趋势型强势机会 ────────────────────
    macd_gold  = dif > dea and hist > 0
    bull_ma    = dev5 > 0 and dev20 > 0
    cci_strong = cci > 80
    vol_ok     = 0.6 <= vol <= 2.5   # 量能温和，非追涨过热

    if bull_ma and macd_gold and rise_prob >= 0.70 and cci_strong and vol_ok:
        return (
            "长期 1月+",
            f"均线多头排列、MACD 金叉、CCI={cci:.0f} 强势、量能温和，"
            "趋势型机会，可波段持有至信号走弱。"
        )

    if bull_ma and macd_gold and rise_prob >= 0.60 and vol_ok:
        return (
            "中长期 2-4周",
            "价格站稳 MA5/MA20、MACD 金叉向上，趋势初步确认，"
            "可持有观察均线是否维持多头。"
        )

    # ── 中期：震荡偏多 ────────────────────────────────────
    kdj_ok = k > d and 30 < k < 85
    rsi_ok = 40 <= rsi <= 68
    if (bull_ma or macd_gold) and (rsi_ok or kdj_ok) and rise_prob >= 0.55:
        return (
            "中期 5-10日",
            f"{'均线支撑向好' if bull_ma else 'MACD 金叉'}，RSI={rsi:.0f} 区间健康，"
            "短中线均可参与，注意量能配合。"
        )

    # ── 默认：短中线 ──────────────────────────────────────
    return (
        "短线 1-5日",
        "当前信号偏中性，建议短线参与，严格止损，待趋势明朗再决定是否续持。"
    )


# ── 上涨理由 & 市场利弊 ───────────────────────────────────────────────────

def build_bull_reasons(last: dict, result: dict,
                       sector: str = "", hot_names: set = None) -> tuple[list, list]:
    """
    生成「上涨理由」和「主要风险」两组 bullet 列表。
    Returns: (bull_list, bear_list)
    """
    hot_names = hot_names or set()

    def _f(key, default=0.0):
        v = last.get(key, default)
        return float(v) if v is not None and v == v else default

    bull, bear = [], []

    # 模型信号
    self_rank_pct = result.get("self_rank_pct", 0.5)
    if self_rank_pct <= 0.10:
        bull.append("模型评分创近60日新高" if self_rank_pct <= 0 else f"模型评分处于自身近60日前{self_rank_pct:.0%}强势区间")
    elif self_rank_pct >= 0.70:
        bear.append(f"模型评分处于自身近60日偏弱区间（前{self_rank_pct:.0%}）")

    # MACD
    dif  = _f("macd_dif")
    dea  = _f("macd_dea")
    hist = _f("macd_hist")
    if dif > dea and hist > 0:
        bull.append(f"MACD 金叉向上，柱值 +{hist:.4f}，动能持续增强")
    elif dif > dea and hist < 0:
        bull.append("MACD DIF 上穿 DEA，金叉初现，动能反转信号")
    elif dif < dea and hist < 0:
        bear.append("MACD 死叉运行，短线动能偏弱")

    # RSI
    rsi6 = _f("rsi6", 50)
    if 50 < rsi6 < 75:
        bull.append(f"RSI6={rsi6:.0f} 处于强势区间，动能充沛")
    elif rsi6 >= 75:
        bear.append(f"RSI6={rsi6:.0f} 超买，注意短线回调压力")
    elif rsi6 <= 30:
        bull.append(f"RSI6={rsi6:.0f} 超卖，超跌反弹机会")
    elif rsi6 <= 45:
        bear.append(f"RSI6={rsi6:.0f} 偏弱，动能尚未恢复")

    # 均线
    dev5  = _f("ma5_dev")
    dev20 = _f("ma20_dev")
    if dev5 > 0 and dev20 > 0:
        bull.append(f"价格站上 MA5/MA20，均线多头排列（MA20 偏离 {dev20:+.1f}%）")
    elif dev5 > 0 > dev20:
        bull.append("短线回到 MA5 上方")
        bear.append(f"MA20 仍压制价格（偏离 {dev20:.1f}%）")
    elif dev5 < 0 and dev20 < 0:
        bear.append(f"价格跌破 MA5/MA20，空头排列（MA20 偏离 {dev20:.1f}%）")

    # 量比
    vol = _f("vol_ratio", 1.0)
    if vol >= 2.5:
        bull.append(f"量比 {vol:.1f}x 大幅放量，主力资金积极介入")
    elif vol >= 1.5:
        bull.append(f"量比 {vol:.1f}x 温和放量，人气回升")
    elif vol <= 0.4:
        bear.append(f"量比 {vol:.1f}x 持续缩量，观望情绪浓厚")

    # KDJ
    k = _f("kdj_k", 50)
    d = _f("kdj_d", 50)
    j = _f("kdj_j", 50)
    if k > d and 20 < j < 80:
        bull.append(f"KDJ 金叉向上（K={k:.0f} D={d:.0f}），短线动能启动")
    elif k > d and j >= 80:
        bear.append(f"KDJ 高位金叉（J={j:.0f}），短线过热")
    elif k < d and j <= 20:
        bull.append(f"KDJ 低位死叉（J={j:.0f}），超卖反弹窗口临近")
    elif k < d and j > 20:
        bear.append(f"KDJ 死叉运行（K={k:.0f} D={d:.0f}），下行压力犹存")

    # 布林带
    bb_upper = _f("bb_upper")
    bb_lower = _f("bb_lower")
    bb_mid   = _f("bb_mid")
    close    = _f("close") or _f("收盘")
    if bb_upper and close:
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
        if close > bb_upper:
            bear.append("突破布林上轨，强势但追高风险加大")
        elif bb_lower and close < bb_lower:
            bull.append("跌破布林下轨，超跌反弹概率较高")
        elif bb_mid and close > bb_mid:
            bull.append("价格在布林中轨上方运行，趋势偏多")
        if bb_width < 0.05:
            bull.append("布林带收窄蓄势，变盘在即")

    # 板块
    if sector and sector in hot_names:
        bull.append(f"所属板块「{sector}」今日领涨市场，板块效应加持")
    elif sector:
        bear.append(f"所属板块「{sector}」今日未进入热点前列")

    return bull, bear


_NEG_WORDS = {"下跌", "跌", "减少", "亏损", "风险", "警告", "处罚", "调查",
              "暂停", "退市", "违规", "下调", "负面", "危机", "关税", "制裁",
              "下行", "萎缩", "收缩", "拖累", "压制", "利空"}
_POS_WORDS = {"上涨", "涨", "增长", "盈利", "突破", "利好", "获批", "中标",
              "合同", "业绩", "超预期", "上调", "回购", "增持", "扩产", "新高",
              "签约", "订单", "分红", "派息", "重组", "并购", "转型", "强势",
              "领涨", "龙头", "爆发", "启动", "反弹"}

# 逐股新闻缓存：{code: (fetch_timestamp, [(title, snippet, is_pos, is_neg), ...])}
_NEWS_CACHE: dict = {}
_NEWS_TTL   = 600   # 10 分钟


def _fetch_em_news_raw(code: str, page_size: int = 10) -> list:
    """
    调东财搜索 API 拉取个股新闻，返回原始 item 列表。
    每条 item 为 dict，含 title / art_abstract / date 字段。
    """
    import urllib.request, urllib.parse, json, time
    inner = {
        "uid": "", "keyword": code,
        "type": ["cmsArticleWebOld"], "client": "web",
        "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {
            "searchScope": "default", "sort": "default",
            "pageIndex": 1, "pageSize": page_size,
            "preTag": "", "postTag": "",
        }},
    }
    params = urllib.parse.urlencode({
        "cb": "cb",
        "param": json.dumps(inner, ensure_ascii=False),
        "_": str(int(time.time() * 1000)),
    })
    url = "https://search-api-web.eastmoney.com/search/jsonp?" + params
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": f"https://so.eastmoney.com/news/s?keyword={code}",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode("utf-8", errors="replace")
    raw = raw[raw.find("(") + 1: raw.rfind(")")]
    data = json.loads(raw)
    items = data.get("result", {}).get("cmsArticleWebOld", [])
    return items if isinstance(items, list) else []


def _classify_news(items: list) -> list:
    """将原始 item 转为 (title, snippet, is_pos, is_neg, date_str) 元组列表。"""
    result = []
    for it in items:
        title = (it.get("title") or "").replace("<em>", "").replace("</em>", "").strip()
        abst  = (it.get("art_abstract") or "").replace("<em>", "").replace("</em>", "").strip()
        text  = (title + " " + abst).strip()
        if not text:
            continue
        is_pos = any(w in text for w in _POS_WORDS)
        is_neg = any(w in text for w in _NEG_WORDS)
        snippet  = title if title else abst[:80]
        date_str = (it.get("date") or "")[:10]   # 取 YYYY-MM-DD
        result.append((title, snippet, is_pos, is_neg, date_str))
    return result


def fetch_cls_news_batch(codes: list = None) -> None:
    """
    预热指定股票代码的新闻缓存（并行拉取）。
    codes 为空时为空操作。调用后 fetch_cls_news_for 可直接命中缓存。
    """
    if not codes:
        return
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    to_fetch = [c for c in codes
                if c not in _NEWS_CACHE or time.time() - _NEWS_CACHE[c][0] >= _NEWS_TTL]
    if not to_fetch:
        return

    def _fetch(code):
        try:
            items = _fetch_em_news_raw(code)
            return code, _classify_news(items)
        except Exception:
            return code, []

    with ThreadPoolExecutor(max_workers=min(6, len(to_fetch))) as pool:
        for code, classified in pool.map(_fetch, to_fetch):
            _NEWS_CACHE[code] = (time.time(), classified)


def _dedup_news(items: list, threshold: float = 0.7) -> list:
    """
    对新闻列表去重：若两条新闻的字符集 Jaccard 相似度 >= threshold，
    保留较早出现的那条（通常是更完整的原版），丢弃后续重复。
    items 可以是字符串列表，也可以是 (title, date_str) 元组列表。
    """
    def _key(item):
        text = item[0] if isinstance(item, tuple) else item
        # 只取汉字和数字、百分号作为特征，去掉标点/空格干扰
        return set(re.findall(r'[\u4e00-\u9fa5\d%]', text))

    kept = []
    for item in items:
        k = _key(item)
        is_dup = False
        for prev in kept:
            pk = _key(prev)
            union = len(k | pk)
            if union == 0:
                continue
            if len(k & pk) / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
    return kept


def fetch_cls_news_for(name: str, sector: str = "", code: str = "",
                       top_n: int = 3) -> tuple[list, list]:
    """
    获取个股相关新闻，返回 (positive_snippets, negative_snippets)。
    优先按 code 从东财搜索；结果按情绪词分类。
    """
    import time
    if not code:
        return [], []

    # 检查缓存
    cached = _NEWS_CACHE.get(code)
    if not cached or time.time() - cached[0] >= _NEWS_TTL:
        try:
            items = _fetch_em_news_raw(code)
            classified = _classify_news(items)
        except Exception:
            classified = []
        _NEWS_CACHE[code] = (time.time(), classified)
    else:
        classified = cached[1]

    pos_texts, neg_texts, neu_texts = [], [], []
    for _title, snippet, is_pos, is_neg, _date in classified:
        if is_pos and not is_neg:
            pos_texts.append(snippet)
        elif is_neg and not is_pos:
            neg_texts.append(snippet)
        elif is_pos and is_neg:
            neg_texts.append(snippet)   # 混合情绪归负面
        else:
            neu_texts.append(snippet)

    # 去重（相似度 ≥ 70% 视为同一条）
    pos_texts = _dedup_news(pos_texts)
    neg_texts = _dedup_news(neg_texts)
    neu_texts = _dedup_news(neu_texts)

    pos_out = pos_texts[:top_n]
    neg_out = neg_texts[:top_n]
    if len(pos_out) < top_n:
        pos_out += neu_texts[: top_n - len(pos_out)]

    # 把新闻文本里的 6 位数字代码替换为「名称（代码）」格式
    def _replace_codes(texts: list) -> list:
        try:
            from data.fetcher import _load_name_map
            nm = _load_name_map()
        except Exception:
            return texts
        result = []
        for t in texts:
            def _sub(m):
                c = m.group(0)
                n = nm.get(c)
                return f"{n}（{c}）" if n else c
            result.append(re.sub(r'\b\d{6}\b', _sub, t))
        return result

    return _replace_codes(pos_out), _replace_codes(neg_out)


# ── 多周期趋势文字K线 ──────────────────────────────────────

def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    resampled = (
        df.set_index("date")
        .sort_index()
        .resample(freq)
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    return resampled


def _trend_label(chg_pct: float) -> tuple[str, str]:
    """(方向文字, 共振箭头)"""
    if chg_pct >= 5:   return "强势上涨", "↑↑"
    if chg_pct >= 1:   return "上涨",     "↑"
    if chg_pct > -1:   return "震荡",     "→"
    if chg_pct > -5:   return "下跌",     "↓"
    return "强势下跌", "↓↓"


def _pattern_desc(closes: list, chg_pct: float) -> str:
    """根据收盘价序列形状生成一句中文形态描述。"""
    n = len(closes)
    if n < 4:
        return ""
    min_idx = closes.index(min(closes))
    max_idx = closes.index(max(closes))
    mid = n // 2

    if chg_pct >= 8:
        return "持续强势上涨" if max_idx > mid else "快速拉升后高位"
    if chg_pct <= -8:
        return "持续弱势下跌" if min_idx > mid else "快速下跌后低位"

    if min_idx < max_idx:          # 先跌后涨（V形或W形）
        if chg_pct >= 2:
            return "先跌后涨，趋势转强"
        elif chg_pct >= -2:
            return "先跌后反弹，方向待确认"
        else:
            return "先跌后反弹，尚未收复"
    else:                          # 先涨后跌（倒V形）
        if chg_pct >= 2:
            return "先涨后回调，仍处高位"
        elif chg_pct >= -2:
            return "冲高回落，高位震荡"
        else:
            return "先涨后跌，回调明显"


def _volume_trend(volumes: list) -> str:
    """量能整体趋势标签（用于走势行末尾）。"""
    if len(volumes) < 4:
        return ""
    half = len(volumes) // 2
    avg_first  = sum(volumes[:half]) / half
    avg_second = sum(volumes[half:]) / (len(volumes) - half)
    if not avg_first:
        return ""
    ratio = avg_second / avg_first
    if ratio >= 1.4:   return "  量能放大"
    if ratio <= 0.65:  return "  量能萎缩"
    return ""


def _pos_bar(cur: float, lo: float, hi: float, width: int = 20) -> tuple[str, float]:
    if hi <= lo:
        return "─" * width, 50.0
    pct = (cur - lo) / (hi - lo) * 100
    fill = max(0, min(width, round(pct / 100 * width)))
    return "█" * fill + "░" * (width - fill), pct


def _pos_zone(pct: float) -> str:
    if pct <= 30:   return "偏低区"
    if pct <= 70:   return "中位区"
    return "偏高区"


def _ma_arrangement(close: float, ma5, ma20, ma60) -> str:
    vals = [(float(v), lbl) for v, lbl in [(ma5, "MA5"), (ma20, "MA20"), (ma60, "MA60")]
            if v is not None and pd.notna(v)]
    if not vals:
        return ""
    prices = [close] + [v for v, _ in vals]
    if all(prices[i] > prices[i + 1] for i in range(len(prices) - 1)):
        return "  **多头排列**"
    if all(prices[i] < prices[i + 1] for i in range(len(prices) - 1)):
        return "  **空头排列**"
    return "  混乱排列"


def _resonance_summary(d_arrow: str, w_arrow: str, m_arrow: str) -> str:
    score = {"↑↑": 2, "↑": 1, "→": 0, "↓": -1, "↓↓": -2}
    d, w, m = score.get(d_arrow, 0), score.get(w_arrow, 0), score.get(m_arrow, 0)
    total = d + w + m
    if total >= 4:
        conclusion = "三线共振向上，强势做多"
    elif total >= 2:
        conclusion = "多头共振，顺势做多"
    elif total == 1:
        conclusion = "短期回调，中长期看涨" if (m > 0 and w > 0) else "趋势初启，等待确认"
    elif total == 0:
        conclusion = "大跌中反弹，谨慎" if (m < 0 and d > 0) else "多空均衡，观望为主"
    elif total == -1:
        conclusion = "中长期下行，短期反弹" if (m < 0 and w < 0) else "短期承压，等待支撑"
    elif total >= -3:
        conclusion = "空头共振，控制仓位"
    else:
        conclusion = "三线共振向下，回避为主"
    return f"月线 **{m_arrow}**  周线 **{w_arrow}**  日线 **{d_arrow}**  →  {conclusion}"


def _timeframe_section(df_period: pd.DataFrame, title: str, n: int,
                       show_ma: bool = False,
                       cur_price: float = None) -> tuple[str, str]:
    """
    Returns (resonance_arrow, section_markdown).

    每个周期格式（4~5行）：
      标题行   周期名 + 趋势文字 + 涨跌幅
      形态行   先跌后涨 / 高位震荡 等中文描述
      走势行   sparkline + 高低价 + 量能趋势标签
      位置行   温度条 + 现价 + 区间百分比 + 偏低/中位/偏高
      均线行   MA5/MA20/MA60 + 排列判断（仅日线）
    """
    df_n = df_period.tail(n).reset_index(drop=True)
    if df_n.empty:
        return "→", ""

    closes  = df_n["close"].tolist()
    highs   = df_n["high"].tolist()
    lows    = df_n["low"].tolist()
    volumes = df_n["volume"].tolist()

    spark_p       = _to_spark(closes)
    display_price = cur_price if (cur_price and show_ma) else closes[-1]
    first, hi, lo = closes[0], max(highs), min(lows)
    chg_pct       = (closes[-1] - first) / first * 100 if first else 0

    trend_lbl, arrow = _trend_label(chg_pct)
    pattern          = _pattern_desc(closes, chg_pct)
    vol_tag          = _volume_trend(volumes)
    chg_sym          = "▲" if chg_pct >= 0 else "▼"
    sign             = "+" if chg_pct >= 0 else ""

    bar, pos_pct = _pos_bar(display_price, lo, hi)
    zone         = _pos_zone(pos_pct)

    lines = [
        f"**{title}**  {trend_lbl} {chg_sym} {sign}{chg_pct:.1f}%",
    ]
    if pattern:
        lines.append(f"形态  {pattern}")
    lines += [
        f"走势  `{spark_p}`  高 {hi:.2f}  低 {lo:.2f}{vol_tag}",
        f"位置  `{lo:.2f} ▕{bar}▏ {hi:.2f}`  ◆ {display_price:.2f}（{zone} {pos_pct:.0f}%）",
    ]

    if show_ma:
        ma5  = df_n["ma5"].iloc[-1]  if "ma5"  in df_n.columns else None
        ma20 = df_n["ma20"].iloc[-1] if "ma20" in df_n.columns else None
        ma60 = df_n["ma60"].iloc[-1] if "ma60" in df_n.columns else None
        ma_parts = []
        for v, lbl in [(ma5, "MA5"), (ma20, "MA20"), (ma60, "MA60")]:
            if v is not None and pd.notna(v):
                ma_parts.append(f"{lbl} {float(v):.2f}")
        if ma_parts:
            arrangement = _ma_arrangement(display_price, ma5, ma20, ma60)
            lines.append("均线  " + "  ".join(ma_parts) + arrangement)

    return arrow, "\n".join(lines)


def _signal_section(df: pd.DataFrame, cur_price: float = None) -> str:
    """根据技术指标生成买卖信号摘要。"""
    if df is None or df.empty or len(df) < 10:
        return ""

    last = df.iloc[-1]
    close = cur_price or (float(last["close"]) if "close" in df.columns and pd.notna(last.get("close")) else 0.0)

    bull: list[str] = []
    bear: list[str] = []

    def _last_cross(col_a: str, col_b: str, window: int = 10):
        if col_a not in df.columns or col_b not in df.columns:
            return None
        sub = df[[col_a, col_b]].tail(window).dropna()
        if len(sub) < 2:
            return None
        diff = (sub[col_a] - sub[col_b]).tolist()
        for i in range(len(diff) - 1, 0, -1):
            if diff[i - 1] < 0 and diff[i] >= 0:
                return ("up", len(diff) - 1 - i)
            if diff[i - 1] > 0 and diff[i] <= 0:
                return ("down", len(diff) - 1 - i)
        return None

    def _days_label(d: int) -> str:
        return "（今日）" if d == 0 else f"（{d}日前）"

    # MA5 / MA20 金叉死叉
    cross = _last_cross("ma5", "ma20")
    if cross:
        direction, days = cross
        if direction == "up":
            bull.append(f"MA5金叉MA20{_days_label(days)}")
        else:
            bear.append(f"MA5死叉MA20{_days_label(days)}")

    # 价格与 MA60 交叉
    cross60 = _last_cross("close", "ma60")
    if cross60:
        direction, days = cross60
        if direction == "up":
            bull.append(f"价格突破MA60{_days_label(days)}")
        else:
            bear.append(f"价格跌破MA60{_days_label(days)}")

    # MACD 金叉死叉；无近期交叉时报当前区间
    macd_cross = _last_cross("macd_dif", "macd_dea")
    if macd_cross:
        direction, days = macd_cross
        if direction == "up":
            bull.append(f"MACD金叉{_days_label(days)}")
        else:
            bear.append(f"MACD死叉{_days_label(days)}")
    else:
        if "macd_dif" in df.columns and "macd_dea" in df.columns:
            dif_s = df["macd_dif"].dropna()
            dea_s = df["macd_dea"].dropna()
            if len(dif_s) > 0 and len(dea_s) > 0:
                cur_dif = float(dif_s.iloc[-1])
                cur_dea = float(dea_s.iloc[-1])
                if cur_dif > cur_dea and cur_dif > 0:
                    bull.append("MACD红柱运行，动能向上")
                elif cur_dif < cur_dea and cur_dif < 0:
                    bear.append("MACD绿柱运行，动能向下")

    # RSI 超买超卖
    if "rsi6" in df.columns:
        rsi_s = df["rsi6"].dropna()
        if len(rsi_s) > 0:
            rsi6 = float(rsi_s.iloc[-1])
            if rsi6 < 30:
                bull.append(f"RSI6超卖（{rsi6:.0f}）")
            elif rsi6 > 75:
                bear.append(f"RSI6超买（{rsi6:.0f}）")

    # KDJ 金叉死叉及极值区
    if "kdj_k" in df.columns and "kdj_d" in df.columns:
        kdj_cross = _last_cross("kdj_k", "kdj_d", window=5)
        k_s = df["kdj_k"].dropna()
        d_s = df["kdj_d"].dropna()
        if len(k_s) > 0 and len(d_s) > 0:
            k_cur = float(k_s.iloc[-1])
            d_cur = float(d_s.iloc[-1])
            if kdj_cross:
                direction, days = kdj_cross
                if direction == "up":
                    tag = "低位" if k_cur < 50 else ""
                    bull.append(f"KDJ{tag}金叉{_days_label(days)}（K:{k_cur:.0f}）")
                else:
                    tag = "高位" if k_cur > 50 else ""
                    bear.append(f"KDJ{tag}死叉{_days_label(days)}（K:{k_cur:.0f}）")
            elif k_cur < 20 and d_cur < 20:
                bull.append(f"KDJ超卖区（K:{k_cur:.0f} D:{d_cur:.0f}）")
            elif k_cur > 80 and d_cur > 80:
                bear.append(f"KDJ超买区（K:{k_cur:.0f} D:{d_cur:.0f}）")

    # 布林带上下轨
    if "bb_upper" in df.columns and "bb_lower" in df.columns and close > 0:
        bb_u = df["bb_upper"].dropna()
        bb_l = df["bb_lower"].dropna()
        if len(bb_u) > 0 and len(bb_l) > 0:
            upper = float(bb_u.iloc[-1])
            lower = float(bb_l.iloc[-1])
            if close <= lower * 1.015:
                bull.append(f"触及布林下轨（{lower:.2f}），超卖区间")
            elif close >= upper * 0.985:
                bear.append(f"触及布林上轨（{upper:.2f}），超买区间")

    if not bull and not bear:
        return ""

    lines = ["**🎯 买卖信号**"]
    for s in bull:
        lines.append(f"▲ {s}")
    for s in bear:
        lines.append(f"▼ {s}")

    nb, ns = len(bull), len(bear)
    if nb >= 3 and ns == 0:
        conclusion = "多项做多信号共振，可积极关注买入机会。"
    elif nb > ns + 1:
        conclusion = "做多信号偏多，可关注介入时机。"
    elif ns >= 3 and nb == 0:
        conclusion = "多项做空信号共振，建议规避或减仓。"
    elif ns > nb + 1:
        conclusion = "做空信号偏多，建议谨慎或观望。"
    elif nb > 0 and ns > 0:
        conclusion = "多空信号交织，等待方向明确后操作。"
    elif nb > 0:
        conclusion = "出现做多信号，结合趋势方向确认后可介入。"
    else:
        conclusion = "出现做空信号，注意控制风险。"
    lines.append(f"综合  {conclusion}")

    # ── 价位建议 ────────────────────────────────────────────
    try:
        last_row   = df.iloc[-1].to_dict()
        price_info = {"price": close}
        trade = suggest_trade_levels(last_row, price_info, rise_prob=0.5)
        if trade:
            buy   = trade["buy_price"]
            hold  = trade["stop_price"]
            sell  = trade["sell_price"]
            rr    = trade["rr_ratio"]
            lines.append(f"价位  买入 **{buy}**  持仓 {hold}  卖出 **{sell}**  盈亏比 1:{rr}")
    except Exception:
        pass

    return "\n".join(lines)


def text_trend_kline(df: pd.DataFrame, code: str, name: str,
                     cur_price: float = None) -> dict:
    """
    多周期趋势分析，返回 dict:
      summary  : 共振摘要行
      ma5      : MA5 sparkline 区块
      ma20     : MA20 sparkline 区块
      ma60     : MA60 sparkline 区块
      daily    : 日线 markdown
      weekly   : 周线 markdown
      monthly  : 月线 markdown

    df: fetch_stock_hist() 返回的日线 DataFrame
    cur_price: 实时价格，有值时用于日线位置条
    """
    if df is None or df.empty:
        return {"summary": f"「{name}（{code}）」暂无历史数据",
                "ma5": "", "ma20": "", "ma60": "",
                "daily": "", "weekly": "", "monthly": ""}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    try:
        from features.technical import add_indicators
        df = add_indicators(df)
    except Exception:
        pass

    def _ma_block(col: str, label: str, n: int = 20) -> str:
        if col not in df.columns:
            return ""
        vals = [float(v) for v in df[col].tail(n).tolist() if pd.notna(v)]
        if not vals:
            return ""
        return f"{label}  {vals[-1]:.2f}  `{_to_spark(vals)}`"

    def _ma_slope_label(col: str, periods: int = 4) -> str:
        if col not in df.columns:
            return ""
        vals = df[col].dropna().tail(periods + 1).tolist()
        if len(vals) < 2:
            return ""
        diff = float(vals[-1]) - float(vals[0])
        if diff > 0:
            return "上翘↗"
        if diff < 0:
            return "下弯↘"
        return "走平→"

    def _arr_type(close: float, ma5, ma20, ma60) -> str:
        vals = [(float(v),) for v in [ma5, ma20, ma60] if v is not None and pd.notna(v)]
        prices = [close] + [v[0] for v in vals]
        if len(prices) >= 2 and all(prices[i] > prices[i+1] for i in range(len(prices)-1)):
            return "多头排列"
        if len(prices) >= 2 and all(prices[i] < prices[i+1] for i in range(len(prices)-1)):
            return "空头排列"
        return "混乱排列"

    def _ma_comment(close: float, ma5, ma20, ma60, arr: str, slope5: str) -> str:
        def _safe(v):
            return v is not None and not pd.isna(v)
        a5  = (close >= ma5)  if _safe(ma5)  else None
        a20 = (close >= ma20) if _safe(ma20) else None
        a60 = (close >= ma60) if _safe(ma60) else None
        if arr == "多头排列":
            if "上翘" in slope5:
                return "多头排列叠加MA5上翘，趋势强势，逢回调MA5可择机介入。"
            return "均线多头排列，多层支撑有效，趋势偏强，持股为主。"
        if arr == "空头排列":
            if "下弯" in slope5:
                return "空头排列叠加MA5下弯，弱势延续，控制仓位为主。"
            return "均线空头排列，多层压力压制，趋势偏弱，谨慎操作。"
        # 混乱排列
        if a5 and a20 and a60 is False and _safe(ma60):
            return f"价格站上MA5/MA20，挑战MA60（{float(ma60):.2f}）压力，突破则趋势转强。"
        if a5 and a20 is False and _safe(ma20):
            return f"价格在MA5上方，MA20（{float(ma20):.2f}）形成压力，等待进一步确认。"
        if a5 is False and a20 and _safe(ma5):
            return f"价格回踩MA5（{float(ma5):.2f}）下方，MA20支撑仍在，属正常回调。"
        if a5 is False and a20 is False and a60 and _safe(ma60):
            return f"价格跌破MA5/MA20，MA60（{float(ma60):.2f}）支撑待验证，短期偏弱。"
        if a5 is False and a20 is False and a60 is False:
            return "价格全部跌破三线，弱势明显，注意控制风险。"
        return "均线纠缠，多空分歧，方向待确认，观望为主。"

    def _get_latest(col: str):
        if col not in df.columns:
            return None
        s = df[col].dropna()
        return float(s.iloc[-1]) if len(s) > 0 else None

    ma_lines = [b for b in (
        _ma_block("ma5",  "MA5"),
        _ma_block("ma20", "MA20"),
        _ma_block("ma60", "MA60"),
    ) if b]

    if ma_lines:
        close_now = cur_price or _get_latest("close") or 0.0
        ma5_v, ma20_v, ma60_v = _get_latest("ma5"), _get_latest("ma20"), _get_latest("ma60")
        arr   = _arr_type(close_now, ma5_v, ma20_v, ma60_v)
        slope5 = _ma_slope_label("ma5")
        arr_display = f"**{arr}**" if arr in ("多头排列", "空头排列") else arr
        arrange_line = f"排列  {arr_display}  MA5{slope5}" if slope5 else f"排列  {arr_display}"
        comment_line = "点评  " + _ma_comment(close_now, ma5_v, ma20_v, ma60_v, arr, slope5)
        ma_sec = "\n".join(["**📈 均线（近20日）**"] + ma_lines + [arrange_line, comment_line])
    else:
        ma_sec = ""

    d_arrow, daily_sec = _timeframe_section(
        df, "📅 日线（近20日）", 20, show_ma=True, cur_price=cur_price)

    try:
        df_w = _resample_ohlcv(df, "W")
        w_arrow, weekly_sec = _timeframe_section(df_w, "📅 周线（近13周）", 13)
    except Exception:
        w_arrow, weekly_sec = "→", ""

    try:
        df_m = _resample_ohlcv(df, "ME")
    except Exception:
        try:
            df_m = _resample_ohlcv(df, "M")
        except Exception:
            df_m = pd.DataFrame()
    if not df_m.empty:
        m_arrow, monthly_sec = _timeframe_section(df_m, "📅 月线（近12月）", 12)
    else:
        m_arrow, monthly_sec = "→", ""

    signals_sec = _signal_section(df, cur_price)

    return {
        "summary": _resonance_summary(d_arrow, w_arrow, m_arrow),
        "ma":      ma_sec,
        "signals": signals_sec,
        "daily":   daily_sec,
        "weekly":  weekly_sec,
        "monthly": monthly_sec,
    }


