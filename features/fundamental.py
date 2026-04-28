"""
fundamental.py - 基于财务报表的深度解构与投资方向分析。

评分维度（每项 -2 ~ +2）：
  营收增速 / 净利增速 / 毛利率趋势 / ROE / 资产负债率 / 现金流质量
总分 → 投资方向 → 买卖建议
"""
import pandas as pd


# ── 工具函数 ────────────────────────────────────────────────

def _safe_float(val, default=None):
    if val is None:
        return default
    # 处理 pandas Series（重复索引时 .loc 返回 Series）
    if hasattr(val, "__len__") and not isinstance(val, str):
        try:
            val = val.iloc[0] if hasattr(val, "iloc") else list(val)[0]
        except Exception:
            return default
    s = str(val).strip()
    if s in ("", "—", "-", "nan", "None", "NaN", "null", "--", "N/A", "暂无"):
        return default
    # 处理带单位后缀
    multiplier = 1.0
    if s.endswith("万亿"):
        multiplier = 1e12; s = s[:-2]
    elif s.endswith("亿"):
        multiplier = 1e8;  s = s[:-1]
    elif s.endswith("万"):
        multiplier = 1e4;  s = s[:-1]
    s = s.replace(",", "").replace("%", "").strip()
    try:
        v = float(s)
        return default if v != v else v * multiplier   # NaN guard
    except Exception:
        return default


def _pct_str(v, default="—"):
    if v is None:
        return default
    return f"{v:+.1f}%"


def _growth(new, old):
    """同比增长率（%）。"""
    if old and old != 0:
        return (new - old) / abs(old) * 100
    return None


# ── 格式识别关键词 ───────────────────────────────────────────

_METRIC_KWS = ("收入", "利润", "成本", "资产", "负债", "现金",
               "费用", "营业", "毛利", "流量", "净额", "股东", "权益")


def _looks_like_metric(s: str) -> bool:
    return any(kw in s for kw in _METRIC_KWS)


# ── 核心提取器（不使用 set_index，避免重复索引问题）──────────

def _extract_metrics_as_rows(df: pd.DataFrame, wanted: list) -> list:
    """
    格式 A（THS 三张报表实际格式）：
      - 第一列（名为"报告期"）的值 = 指标名（如 *营业总收入）
      - 其余列的名称 = 报告期日期（如 2024-09-30）
    注意：存在重复列名（如两个 *营业总收入），不能用 set_index。
    """
    try:
        name_col = df.columns[0]
        # 期间列：名称不像指标名的列（即日期列）
        period_cols = [c for c in df.columns[1:]
                       if not _looks_like_metric(str(c))][:8]
        if not period_cols:
            period_cols = list(df.columns[1:])[:8]

        # 获取指标名 Series，用于 str.contains 匹配
        metric_names = df[name_col].astype(str)

        result = []
        for period in period_cols:
            rec = {"period": str(period)[:10]}
            for key, aliases in wanted:
                rec[key] = None
                for alias in aliases:
                    mask = metric_names.str.contains(alias, na=False, regex=False)
                    if mask.any():
                        # iloc[0] 取第一个匹配行，避免重复索引返回 Series
                        raw = df.loc[mask, period]
                        if hasattr(raw, "iloc"):
                            raw = raw.iloc[0]
                        v = _safe_float(raw)
                        if v is not None:
                            rec[key] = v
                            break
                if rec[key] is not None:
                    break
            result.append(rec)
        return result
    except Exception:
        return []


def _extract_periods_as_rows(df: pd.DataFrame, wanted: list) -> list:
    """
    格式 B（indicator / abstract 格式）：
      - 每行 = 一个报告期，第一列 = 日期
      - 其余列名 = 指标名
    """
    try:
        result = []
        for _, row in df.head(8).iterrows():
            period = str(row.iloc[0]).strip()[:10]
            if not period or period in ("nan", "None"):
                continue
            rec = {"period": period}
            for key, aliases in wanted:
                rec[key] = None
                for alias in aliases:
                    for col in df.columns[1:]:
                        if alias in str(col):
                            v = _safe_float(row[col])
                            if v is not None:
                                rec[key] = v
                                break
                    if rec[key] is not None:
                        break
            result.append(rec)
        return result
    except Exception:
        return []


def _count_data(records: list) -> int:
    return sum(
        1 for r in records
        for k, v in r.items()
        if k != "period" and v is not None
    )


def _auto_extract(df: pd.DataFrame, wanted: list) -> list:
    """
    自动检测格式并提取数据：两种格式都尝试，取数据更多的那个。
    """
    if df is None or df.empty:
        return []

    # 检测第一列值：包含财务关键词 → 格式 A（指标为行）
    first_vals = df.iloc[:, 0].astype(str).tolist()[:8]
    is_metrics_first = any(_looks_like_metric(v) for v in first_vals)

    if is_metrics_first:
        res_a = _extract_metrics_as_rows(df, wanted)
        if _count_data(res_a) > 0:
            return res_a
        # 尝试格式 B 兜底
        res_b = _extract_periods_as_rows(df, wanted)
        return res_b if _count_data(res_b) > _count_data(res_a) else res_a
    else:
        res_b = _extract_periods_as_rows(df, wanted)
        if _count_data(res_b) > 0:
            return res_b
        # 尝试格式 A 兜底
        res_a = _extract_metrics_as_rows(df, wanted)
        return res_a if _count_data(res_a) > _count_data(res_b) else res_b


# ── 各报表解析 ──────────────────────────────────────────────

def _parse_profit(df: pd.DataFrame) -> list:
    wanted = [
        ("revenue",      ["营业总收入", "一、营业总收入", "营业收入", "主营业务收入"]),
        ("net_profit",   ["归属母公司股东的净利润", "归属于母公司所有者的净利润",
                          "归母净利润", "净利润"]),
        ("gross_margin", ["毛利率", "销售毛利率", "综合毛利率"]),
        ("cogs",         ["营业成本", "主营业务成本", "销售成本"]),
    ]
    records = _auto_extract(df, wanted)

    # 毛利率兜底：用营收和成本计算
    for r in records:
        if r.get("gross_margin") is None:
            rev, cogs = r.get("revenue"), r.get("cogs")
            if rev and cogs and rev > 0:
                r["gross_margin"] = (rev - cogs) / rev * 100
        r.pop("cogs", None)

    return records


def _parse_balance(df: pd.DataFrame) -> list:
    wanted = [
        ("debt_ratio",   ["资产负债率"]),
        ("total_assets", ["资产合计", "资产总计", "总资产", "资产总额"]),
        ("total_liab",   ["负债合计", "负债总计", "总负债", "负债总额"]),
    ]
    records = _auto_extract(df, wanted)

    for r in records:
        if r.get("debt_ratio") is None:
            ta, tl = r.get("total_assets"), r.get("total_liab")
            if ta and tl and ta > 0:
                r["debt_ratio"] = tl / ta * 100
        r.pop("total_assets", None)
        r.pop("total_liab", None)

    return records


def _parse_cashflow(df: pd.DataFrame) -> list:
    wanted = [
        ("ocf", ["经营活动产生的现金流量净额", "经营活动现金流量净额",
                 "经营活动产生现金流量净额", "经营现金净流量",
                 "经营活动产生的现金流量"]),
    ]
    return _auto_extract(df, wanted)


def _parse_indicator(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    try:
        row = df.iloc[0]
        return {str(col): _safe_float(row[col])
                for col in df.columns
                if _safe_float(row[col]) is not None}
    except Exception:
        return {}


def _parse_abstract(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    try:
        row = df.iloc[0]
        return {str(col): _safe_float(row[col])
                for col in df.columns
                if _safe_float(row[col]) is not None}
    except Exception:
        return {}


# ── 综合评分 ────────────────────────────────────────────────

def _score_growth(rate):
    if rate is None: return 0
    if rate >= 30:  return 2
    if rate >= 15:  return 1
    if rate >= 0:   return 0
    if rate >= -10: return -1
    return -2


def _score_margin(cur, prev):
    if cur is None: return 0
    if prev is None: return (1 if cur >= 40 else 0)
    delta = cur - prev
    if delta >= 3:  return 2
    if delta >= 0:  return 1
    if delta >= -3: return -1
    return -2


def _score_roe(roe):
    if roe is None: return 0
    if roe >= 20: return 2
    if roe >= 12: return 1
    if roe >= 6:  return 0
    if roe >= 0:  return -1
    return -2


def _score_debt(ratio):
    if ratio is None: return 0
    if ratio <= 30: return 2
    if ratio <= 50: return 1
    if ratio <= 65: return 0
    if ratio <= 80: return -1
    return -2


def _score_cashflow(ocf, net_profit):
    if ocf is None or net_profit is None or net_profit == 0: return 0
    ratio = ocf / abs(net_profit)
    if ratio >= 1.5: return 2
    if ratio >= 1.0: return 1
    if ratio >= 0.5: return 0
    if ratio >= 0:   return -1
    return -2


# ── 深度洞察生成 ─────────────────────────────────────────────

def _detect_trend(profits: list, key: str) -> str:
    """检测最近 3~4 期指标的趋势方向。"""
    vals = [p.get(key) for p in profits[:4] if p.get(key) is not None]
    if len(vals) < 2:
        return ""
    # 计算相邻期变化方向
    changes = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
    up_cnt   = sum(1 for c in changes if c > 0)
    down_cnt = sum(1 for c in changes if c < 0)
    n = len(changes)
    if up_cnt == n:
        return f"连续{n}期增长"
    if down_cnt == n:
        return f"连续{n}期下滑"
    if up_cnt > down_cnt:
        return "整体向好"
    if down_cnt > up_cnt:
        return "整体走弱"
    return "震荡"


def _build_insights(profits, balances, cashflows,
                    roe, gross_margin, prev_margin, debt_ratio,
                    ocf, net_profit, rev_growth, profit_growth) -> tuple:
    """
    生成亮点列表（insights）和风险列表（warnings）。
    每条为 (emoji, text) 元组。
    """
    good, warn = [], []

    # ── 营收趋势 ─────────────────────────────────────────
    rev_trend = _detect_trend(profits, "revenue")
    if rev_growth is not None:
        if rev_growth >= 30:
            good.append(("📈", f"营收高增长 {rev_growth:+.1f}%（{rev_trend}），成长性优异"))
        elif rev_growth >= 15:
            good.append(("📈", f"营收稳健增长 {rev_growth:+.1f}%（{rev_trend}），景气向上"))
        elif rev_growth >= 0:
            good.append(("➡️", f"营收小幅增长 {rev_growth:+.1f}%（{rev_trend}），增速偏慢"))
        elif rev_growth >= -10:
            warn.append(("⚠️", f"营收同比下滑 {rev_growth:.1f}%（{rev_trend}），需关注业务压力"))
        else:
            warn.append(("🚨", f"营收大幅下滑 {rev_growth:.1f}%（{rev_trend}），经营形势严峻"))

    # ── 利润增速 vs 营收增速 ─────────────────────────────
    if rev_growth is not None and profit_growth is not None:
        if profit_growth >= 0 and rev_growth >= 0:
            if profit_growth > rev_growth * 1.5:
                good.append(("💡", "利润增速显著超越营收，盈利能力持续提升，规模效应显现"))
            elif profit_growth < rev_growth * 0.5 and profit_growth >= 0:
                warn.append(("⚠️", "利润增速明显低于营收，费用或成本快速扩张，利润率承压"))
        if profit_growth < 0 <= rev_growth:
            warn.append(("🔻", f"营收增长但净利润下滑 {profit_growth:.1f}%，增收不增利，成本管控需关注"))
        if profit_growth > 0 > rev_growth:
            good.append(("💡", f"营收下滑但净利提升 {profit_growth:+.1f}%，精细化降本增效，质量改善"))

    # ── 毛利率 ───────────────────────────────────────────
    if gross_margin is not None:
        gm_level = ("高毛利" if gross_margin >= 50 else
                    "中高毛利" if gross_margin >= 35 else
                    "中等毛利" if gross_margin >= 20 else "低毛利")
        if prev_margin is not None:
            delta = gross_margin - prev_margin
            if delta >= 3:
                good.append(("✅", f"毛利率 {gross_margin:.1f}%（{gm_level}）环比提升 {delta:+.1f}pct，定价能力增强"))
            elif delta <= -3:
                warn.append(("⚠️", f"毛利率 {gross_margin:.1f}% 环比下滑 {delta:.1f}pct，竞争加剧或原材料涨价"))
            else:
                good.append(("📊", f"毛利率 {gross_margin:.1f}%（{gm_level}），较上期变化 {delta:+.1f}pct，基本稳定"))
        else:
            good.append(("📊", f"毛利率 {gross_margin:.1f}%（{gm_level}）"))

    # ── ROE ─────────────────────────────────────────────
    if roe is not None:
        if roe >= 20:
            good.append(("🌟", f"ROE {roe:.1f}%，超越 20% 门槛，资本回报率卓越，属优质白马"))
        elif roe >= 15:
            good.append(("✅", f"ROE {roe:.1f}%，回报率优秀，资金利用效率高"))
        elif roe >= 10:
            good.append(("📊", f"ROE {roe:.1f}%，回报率处于合理区间"))
        elif roe >= 6:
            warn.append(("⚠️", f"ROE {roe:.1f}%，回报率偏低，盈利能力有待提升"))
        elif roe >= 0:
            warn.append(("⚠️", f"ROE {roe:.1f}%，资本回报不及存款利率，价值创造能力较弱"))
        else:
            warn.append(("🚨", f"ROE {roe:.1f}%（负值），当期亏损，净资产遭到侵蚀"))

    # ── 资产负债率 ───────────────────────────────────────
    if debt_ratio is not None:
        if debt_ratio >= 80:
            warn.append(("🚨", f"资产负债率高达 {debt_ratio:.1f}%，财务杠杆风险显著，偿债压力大"))
        elif debt_ratio >= 65:
            warn.append(("⚠️", f"资产负债率 {debt_ratio:.1f}%，接近警戒线，需关注融资和偿债节奏"))
        elif debt_ratio >= 50:
            good.append(("📊", f"资产负债率 {debt_ratio:.1f}%，杠杆适中，处于行业正常水平"))
        else:
            good.append(("✅", f"资产负债率仅 {debt_ratio:.1f}%，财务结构稳健，抗风险能力强"))

    # ── 现金流质量 ───────────────────────────────────────
    if ocf is not None and net_profit is not None and net_profit != 0:
        ratio = ocf / abs(net_profit)
        if net_profit > 0:
            if ratio >= 1.5:
                good.append(("💰", f"现金流质量优异（OCF/净利润 = {ratio:.1f}x），利润含金量高，业绩真实可信"))
            elif ratio >= 1.0:
                good.append(("✅", f"现金流与利润匹配（OCF/净利润 = {ratio:.1f}x），盈利质量良好"))
            elif ratio >= 0.5:
                warn.append(("⚠️", f"现金流回收偏慢（OCF/净利润 = {ratio:.1f}x），应收账款或存货可能积压"))
            elif ratio >= 0:
                warn.append(("🔻", f"现金流质量偏弱（OCF/净利润 = {ratio:.1f}x），需关注资金链紧张风险"))
            else:
                warn.append(("🚨", f"经营现金流为负（OCF/净利润 = {ratio:.1f}x），现金告急，盈利质量存疑"))
        else:
            if ocf > 0:
                good.append(("💡", f"虽然净利润亏损，但经营现金流为正，实际运营状况优于账面"))
            else:
                warn.append(("🚨", "净利润和经营现金流均为负，公司面临双重压力"))

    return good, warn


# ── 主分析函数 ──────────────────────────────────────────────

def analyse_financials(data: dict) -> dict:
    profits   = _parse_profit(data.get("profit",    pd.DataFrame()))
    balances  = _parse_balance(data.get("balance",  pd.DataFrame()))
    cashflows = _parse_cashflow(data.get("cashflow",pd.DataFrame()))
    indicator = _parse_indicator(data.get("indicator", pd.DataFrame()))
    abstract  = _parse_abstract(data.get("abstract",  pd.DataFrame()))

    latest_period = (profits[0]["period"]  if profits  else
                     balances[0]["period"] if balances else "—")

    # ── ROE ─────────────────────────────────────────────
    roe = None
    for key in indicator:
        if "净资产收益率" in key or "ROE" in key.upper():
            roe = indicator[key]; break
    if roe is None:
        for key in abstract:
            if "净资产收益率" in key or "ROE" in key.upper():
                roe = abstract[key]; break

    # ── 毛利率 ───────────────────────────────────────────
    gross_margin = profits[0].get("gross_margin") if profits else None
    if gross_margin is None:
        for key in abstract:
            if "毛利率" in key:
                gross_margin = abstract[key]; break

    prev_margin = profits[1].get("gross_margin") if len(profits) >= 2 else None

    # ── 资产负债率 ───────────────────────────────────────
    debt_ratio = balances[0].get("debt_ratio") if balances else None

    # ── 现金流 & 净利润 ──────────────────────────────────
    ocf        = cashflows[0].get("ocf")       if cashflows else None
    net_profit = profits[0].get("net_profit")  if profits   else None
    revenue    = profits[0].get("revenue")     if profits   else None

    # ── 同比增速（与第 5 期比，约 1 年前同季度）────────────
    rev_growth = profit_growth = None
    if len(profits) >= 5:
        rev_growth    = _growth(profits[0].get("revenue") or 0,    profits[4].get("revenue"))
        profit_growth = _growth(profits[0].get("net_profit") or 0, profits[4].get("net_profit"))
    elif len(profits) >= 2:
        rev_growth    = _growth(profits[0].get("revenue") or 0,    profits[-1].get("revenue"))
        profit_growth = _growth(profits[0].get("net_profit") or 0, profits[-1].get("net_profit"))

    # ── 评分 ─────────────────────────────────────────────
    s_rev    = _score_growth(rev_growth)
    s_profit = _score_growth(profit_growth)
    s_margin = _score_margin(gross_margin, prev_margin)
    s_roe    = _score_roe(roe)
    s_debt   = _score_debt(debt_ratio)
    s_cf     = _score_cashflow(ocf, net_profit)

    total = s_rev + s_profit + s_margin + s_roe + s_debt + s_cf
    scores = {
        "营收增速": s_rev,
        "净利增速": s_profit,
        "毛利趋势": s_margin,
        "ROE":     s_roe,
        "负债安全": s_debt,
        "现金质量": s_cf,
    }

    # ── 深度洞察 ─────────────────────────────────────────
    insights, warnings = _build_insights(
        profits, balances, cashflows,
        roe, gross_margin, prev_margin, debt_ratio,
        ocf, net_profit, rev_growth, profit_growth
    )

    # ── 投资方向 ─────────────────────────────────────────
    if total >= 7:
        direction, icon, color = "强烈推荐", "🌟", "red"
        summary = "基本面全面优质，营收利润双高增，ROE 突出，现金流充裕，强力买入。"
    elif total >= 4:
        direction, icon, color = "推荐买入", "📈", "orange"
        summary = "基本面偏强，核心指标良好，可逢回调布局。"
    elif total >= 1:
        direction, icon, color = "谨慎乐观", "👀", "blue"
        summary = "基本面中性偏好，部分指标待改善，可小仓试探。"
    elif total >= -2:
        direction, icon, color = "中性观望", "⚖️", "grey"
        summary = "基本面一般，无明显亮点，建议等待更清晰的信号。"
    elif total >= -5:
        direction, icon, color = "谨慎减持", "⚠️", "yellow"
        summary = "多项指标走弱，需关注业绩持续性，建议降低仓位。"
    else:
        direction, icon, color = "建议回避", "🚫", "red"
        summary = "基本面明显恶化，营收或利润大幅下滑，现金流告急，不建议持有。"

    return {
        "latest_period":   latest_period,
        "profits":         profits[:4],
        "balances":        balances[:4],
        "cashflows":       cashflows[:4],
        "roe":             roe,
        "gross_margin":    gross_margin,
        "debt_ratio":      debt_ratio,
        "ocf":             ocf,
        "net_profit":      net_profit,
        "revenue":         revenue,
        "rev_growth":      rev_growth,
        "profit_growth":   profit_growth,
        "scores":          scores,
        "total_score":     total,
        "direction":       direction,
        "direction_icon":  icon,
        "direction_color": color,
        "summary":         summary,
        "insights":        insights,
        "warnings":        warnings,
    }


# ── 格式化工具 ──────────────────────────────────────────────

def format_value(v, decimals=2):
    """将大数字转为亿/万/元 可读字符串。"""
    if v is None:
        return "—"
    try:
        v = float(v)
        if abs(v) >= 1e8:
            return f"{v/1e8:.{decimals}f}亿"
        if abs(v) >= 1e4:
            return f"{v/1e4:.{decimals}f}万"
        return f"{v:.{decimals}f}"
    except Exception:
        return "—"


# ── 飞书卡片内容构建 ─────────────────────────────────────────

def build_financial_card_content(res: dict) -> tuple:
    """
    将 analyse_financials 结果组装为 3 段飞书 Markdown 文字。
    Returns: (section_overview, section_detail, section_recommend)
    """
    scores   = res["scores"]
    profits  = res["profits"]
    insights = res.get("insights", [])
    warnings = res.get("warnings", [])

    # ── 评分雷达 ──────────────────────────────────────────
    def _bar(score):
        filled = max(0, score + 2)   # -2~+2 → 0~4
        return "█" * filled + "░" * (4 - filled)

    score_lines = "\n".join(
        f"`{_bar(v)}` {k}  {'＋' if v > 0 else ('－' if v < 0 else '  ')}{abs(v)}"
        for k, v in scores.items()
    )

    overview = (
        f"**{res['direction_icon']} {res['direction']}**　综合评分 **{res['total_score']:+d}** / 12\n\n"
        f"{score_lines}\n\n"
        f"_{res['summary']}_"
    )

    # ── 财务趋势 ──────────────────────────────────────────
    trend_rows = []
    for p in profits:
        period = p["period"][:7]
        rev    = format_value(p.get("revenue"))
        profit = format_value(p.get("net_profit"))
        gm     = p.get("gross_margin")
        margin = f"{gm:.1f}%" if gm is not None else "—"
        trend_rows.append(f"· **{period}**　营收 {rev}　净利 {profit}　毛利率 {margin}")

    rev_g    = _pct_str(res.get("rev_growth"))
    pro_g    = _pct_str(res.get("profit_growth"))
    roe_str  = f"{res['roe']:.1f}%"        if res.get("roe")        is not None else "—"
    debt_str = f"{res['debt_ratio']:.1f}%"  if res.get("debt_ratio") is not None else "—"
    ocf_str  = format_value(res.get("ocf"))

    # 深度洞察 & 风险
    insight_lines = "\n".join(f"{e} {t}" for e, t in insights) if insights else ""
    warning_lines = "\n".join(f"{e} {t}" for e, t in warnings) if warnings else ""

    detail_parts = [
        f"**📊 财务趋势（最新 {len(profits)} 期）**",
        "\n".join(trend_rows) if trend_rows else "暂无数据",
        "",
        f"**关键指标**",
        f"营收同比 **{rev_g}**　净利同比 **{pro_g}**",
        f"ROE **{roe_str}**　资产负债率 **{debt_str}**　经营现金流 **{ocf_str}**",
    ]
    if insight_lines:
        detail_parts += ["", "**✅ 亮点**", insight_lines]
    if warning_lines:
        detail_parts += ["", "**⚠️ 风险**", warning_lines]

    detail = "\n".join(detail_parts)

    # ── 投资建议 ──────────────────────────────────────────
    score = res["total_score"]
    if score >= 7:
        buy_advice  = "积极布局，可逢调整加仓，核心仓位重点持有。"
        hold_advice = "持续持有至估值回归或基本面出现明显拐点。"
        risk_advice = "关注行业政策变化及竞争格局是否发生质变。"
        timing      = "当前基本面强劲，技术面回踩均线是较优买点。"
    elif score >= 4:
        buy_advice  = "分批建仓，等待技术面支撑位介入，避免追高。"
        hold_advice = "持有并跟踪季报，利润增速若持续改善可加仓。"
        risk_advice = "注意毛利率是否趋势性下滑，警惕行业竞争加剧。"
        timing      = "建议在业绩确认期（季报发布前后）重点关注信号。"
    elif score >= 1:
        buy_advice  = "小仓试探，等待财报改善再加仓，严格控制成本。"
        hold_advice = "可持有但不建议加仓，止损设在近期重要支撑下方。"
        risk_advice = "密切跟踪下季财报，若关键指标继续转差须及时止损。"
        timing      = "等待一个清晰的基本面改善信号（营收或利润率回升）。"
    else:
        buy_advice  = "暂不建议买入，等待基本面拐点明确后再介入。"
        hold_advice = "建议逐步减仓或清仓，规避业绩持续下行风险。"
        risk_advice = "重点警惕现金流恶化、商誉减值、债务到期等黑天鹅。"
        timing      = "若有持仓，选择技术反弹机会减仓，勿以补仓摊薄成本。"

    recommend = (
        f"**💡 投资方向建议**\n"
        f"**买入策略** {buy_advice}\n"
        f"**持仓策略** {hold_advice}\n"
        f"**操作时机** {timing}\n"
        f"**风险提示** {risk_advice}"
    )

    return overview, detail, recommend
