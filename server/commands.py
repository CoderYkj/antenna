"""
commands.py - 共享指令解析，供 webhook 和 poll 两种模式复用
"""
import json

ADD_WORDS     = {"添加", "加入", "add", "+"}
REMOVE_WORDS  = {"删除", "移除", "remove", "del", "-"}
LIST_WORDS    = {"列表", "关注", "list", "ls", "show"}
QUOTE_WORDS   = {"行情", "价格", "quote", "q"}
PREDICT_WORDS = {"预测", "分析", "predict", "p"}
SCAN_WORDS    = {"推荐", "扫描", "scan", "s"}
REVIEW_WORDS  = {"复盘", "回测", "review", "r"}
REPORT_WORDS  = {"财报", "年报", "季报", "report", "f"}
STRATEGY_WORDS  = {"策略", "strategy", "st"}
TACTIC_WORDS    = {"战法", "tactic"}
BACKTEST_WORDS  = {"历史", "历史回测", "backtest", "bt"}
PUSH_WORDS    = {"发送", "推送", "广播", "push", "broadcast"}
NEWS_WORDS    = {"新闻", "消息", "资讯", "news", "n"}
TREND_WORDS   = {"趋势", "trend", "k线", "kline"}
HELP_WORDS    = {"帮助", "help", "?", "？"}


def _help_card() -> dict:
    def _row(cmd: str, desc: str, example: str = "") -> dict:
        return {
            "tag": "column_set",
            "flex_mode": "none",
            "background_style": "default",
            "columns": [
                {"tag": "column", "width": "weighted", "weight": 3,
                 "elements": [{"tag": "markdown", "content": f"{cmd}"}]},
                {"tag": "column", "width": "weighted", "weight": 4,
                 "elements": [{"tag": "markdown", "content": desc}]},
                {"tag": "column", "width": "weighted", "weight": 4,
                 "elements": [{"tag": "markdown", "content": example}]},
            ],
        }

    # 表头
    header_row = {
        "tag": "column_set",
        "flex_mode": "none",
        "background_style": "grey",
        "columns": [
            {"tag": "column", "width": "weighted", "weight": 3,
             "elements": [{"tag": "markdown", "content": "**指令**"}]},
            {"tag": "column", "width": "weighted", "weight": 4,
             "elements": [{"tag": "markdown", "content": "**说明**"}]},
            {"tag": "column", "width": "weighted", "weight": 4,
             "elements": [{"tag": "markdown", "content": "**示例**"}]},
        ],
    }

    elements = [
        header_row,
        _row("添加 <代码>", "加入自选股", "添加 600519"),
        _row("删除 <代码>", "移出自选股", "删除 600519"),
        _row("列表", "查看自选股及实时行情", ""),
        {"tag": "hr"},
        _row("行情 <代码>", "实时价格 + 日内走势", "行情 002174"),
        _row("趋势 <代码>", "日线/周线/月线 K 线趋势", "趋势 600519"),
        _row("预测 <代码>", "AI 信号 + 技术分析", "预测 300785"),
        _row("新闻 <代码>", "近期正面/负面消息", "新闻 紫金矿业"),
        _row("财报 <代码>", "季报/年报深度解构与投资建议", "财报 600519"),
        _row("推荐 （N）",         "推荐 N 只买入股票（默认5）",           "推荐 3"),
        _row("战法 <策略> （N）",  "按选股战法筛选 Top N（价值/成长/龙头/逆向）", "战法 价值 5"),
        _row("复盘 （日期）",      "查看历史预测命中率及策略调整",         "复盘 2026-04-09"),
        _row("历史 YYYY-MM",   "Walk-Forward 月度历史回测",            "历史 2026-01"),
        _row("策略",               "查看当前选股策略依据与评判标准",       ""),
        {"tag": "hr"},
        _row("发送 <指令>", "执行指令并广播到所有配置会话", "发送 推荐"),
        {"tag": "hr"},
        _row("帮助", "显示此列表", ""),
    ]

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📡 Antenna 指令列表"},
            "template": "blue",
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


def _search_card(keyword: str, matches: list) -> dict:
    """返回多个匹配股票的选择卡片。"""
    import json
    rows = [{"tag": "markdown", "content": f"关键词「{keyword}」匹配到 **{len(matches)}** 只股票，请补全代码重试："}]
    for code, name in matches[:20]:
        rows.append({"tag": "markdown", "content": f"· **{name.replace('*', chr(92)+'*')}**（{code}）"})
    card = {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "🔍 模糊匹配结果"}, "template": "blue"},
        "elements": rows,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


def _resolve_code(parts: list) -> "str | dict | None":
    """
    解析指令中的股票代码。
    - 纯 6 位数字：直接返回
    - 文字关键词：搜索名称，唯一则返回代码，多个则返回选择卡片，无则返回 None
    """
    if len(parts) < 2:
        return None
    raw = parts[1].strip()

    # 纯数字补零后校验
    padded = raw.zfill(6)
    if padded.isdigit() and len(padded) == 6:
        return padded

    # 名称模糊搜索
    from data.fetcher import search_stocks
    matches = search_stocks(raw)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]   # 唯一命中，直接返回代码
    return _search_card(raw, matches)


def handle_command(text: str, chat_id: str = ""):
    """解析指令，返回字符串或飞书 interactive 卡片 payload（dict）。"""
    text = text.strip()
    parts = text.split()
    if not parts:
        return _help_card()

    cmd = parts[0].lower()

    if cmd in HELP_WORDS:
        return _help_card()

    if cmd in LIST_WORDS:
        from server.watchlist import list_codes
        return list_codes()

    if cmd in QUOTE_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`行情 <代码或名称>`，如：行情 600519 或 行情 游族"
        if isinstance(result, dict):
            return result   # 多匹配选择卡片
        from server.predict_cmd import cmd_quote
        return cmd_quote(result)

    if cmd in PREDICT_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`预测 <代码或名称>`，如：预测 600519 或 预测 游族"
        if isinstance(result, dict):
            return result
        from server.predict_cmd import cmd_predict
        return cmd_predict(result)

    if cmd in NEWS_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`新闻 <代码或名称>`，如：新闻 600519 或 新闻 紫金矿业"
        if isinstance(result, dict):
            return result
        from server.predict_cmd import cmd_news
        return cmd_news(result)

    if cmd in TREND_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`趋势 <代码或名称>`，如：趋势 600519 或 趋势 茅台"
        if isinstance(result, dict):
            return result
        from server.predict_cmd import cmd_trend
        return cmd_trend(result)

    if cmd in REPORT_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`财报 <代码或名称>`，如：财报 600519 或 财报 贵州茅台"
        if isinstance(result, dict):
            return result
        from server.predict_cmd import cmd_financial_report
        return cmd_financial_report(result)

    if cmd in SCAN_WORDS:
        # 扫描 （N），N 默认 5
        n = 5
        if len(parts) >= 2 and parts[1].isdigit():
            n = int(parts[1])
        from server.predict_cmd import cmd_scan_bot
        return cmd_scan_bot(n)

    if cmd in REVIEW_WORDS:
        # 复盘 （YYYY-MM-DD），不传日期则取最近一条记录
        date_arg = parts[1] if len(parts) >= 2 else None
        from server.predict_cmd import cmd_review
        return cmd_review(date_arg)

    if cmd in STRATEGY_WORDS:
        from server.predict_cmd import cmd_strategy
        return cmd_strategy()

    if cmd in TACTIC_WORDS:
        # 战法 <策略名> [N]，如：战法 价值 5
        strategy_key = parts[1] if len(parts) >= 2 else ""
        if not strategy_key:
            return ("用法：`战法 <策略> [N]`\n"
                    "支持战法：价值 | 成长 | 龙头 | 逆向\n"
                    "示例：`战法 价值 5`")
        top_n = 5
        if len(parts) >= 3 and parts[2].isdigit():
            top_n = max(1, min(int(parts[2]), 20))
        from server.predict_cmd import cmd_tactic
        return cmd_tactic(strategy_key, top_n)

    if cmd in BACKTEST_WORDS:
        # 回测历史 YYYY-MM 或 YYYY
        arg = parts[1] if len(parts) >= 2 else None
        from server.predict_cmd import cmd_backtest_bot
        return cmd_backtest_bot(arg)

    if cmd in ADD_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`添加 <代码或名称>`，如：添加 600519 或 添加 游族"
        if isinstance(result, dict):
            return result
        from server.watchlist import add_code
        _, msg = add_code(result)
        return msg

    if cmd in REMOVE_WORDS:
        result = _resolve_code(parts)
        if result is None:
            return "用法：`删除 <代码或名称>`，如：删除 600519 或 删除 游族"
        if isinstance(result, dict):
            return result
        from server.watchlist import remove_code
        _, msg = remove_code(result)
        return msg

    # 纯 6 位数字 → 自动添加
    if parts[0].isdigit() and len(parts[0]) == 6:
        from server.watchlist import add_code
        _, msg = add_code(parts[0])
        return msg

    if cmd in PUSH_WORDS:
        # 发送 <指令>：执行子指令并广播到所有配置会话
        sub_text = " ".join(parts[1:]).strip()
        if not sub_text:
            return "用法：`发送 <指令>`，如：发送 推荐  /  发送 复盘  /  发送 行情 600519"
        import threading
        result = handle_command(sub_text)

        def _do_push():
            from server.feishu_push import push
            push(result)

        threading.Thread(target=_do_push, daemon=True).start()
        return f"已执行「{sub_text}」，正在广播到所有会话…"

    # 未识别指令 → 交给 Claude 闲聊
    from server.predict_cmd import cmd_chat
    return cmd_chat(text, chat_id=chat_id)
