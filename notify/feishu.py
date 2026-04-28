"""
飞书自定义机器人 Webhook 推送
"""
import json
import urllib.request
from datetime import datetime


def _post(webhook_url: str, payload: dict) -> bool:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("code", -1) == 0 or result.get("StatusCode", -1) == 0
    except Exception as e:
        print(f"[飞书推送失败] {e}")
        return False


def send_scan_result(webhook_url: str, scan_result: dict) -> bool:
    """将 cmd_scan 返回的 dict 推送到飞书。"""
    if not webhook_url:
        print("[飞书推送] 未配置 webhook_url，跳过推送。")
        return False

    top = scan_result["top"]
    total = scan_result["total"]
    skipped = scan_result["skipped"]
    thresh_strong = scan_result["thresh_strong"]
    thresh_watch = scan_result["thresh_watch"]
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 信号图标
    icons = {"强势": "🔥", "关注": "👀", "观望": "💤"}

    rows = []
    for r in top:
        icon     = icons.get(r["signal"], "")
        raw_name = r.get("name", r["code"])
        name     = raw_name.replace("*", "＊").replace("_", "\\_")
        trade    = r.get("trade", {})

        buy_str  = f"买入 **{trade['buy_price']}**（{trade['buy_desc']}）" if trade.get("buy_price") else ""
        sell_str = f"止盈 **{trade['sell_price']}**（{trade['sell_desc']}）" if trade.get("sell_price") else ""
        stop_str = f"止损 **{trade['stop_price']}**（{trade['stop_desc']}）" if trade.get("stop_price") else ""
        rr_str   = f"盈亏比 **{trade['rr_ratio']}:1**" if trade.get("rr_ratio") else ""

        price_line = "　".join(filter(None, [buy_str, sell_str, stop_str, rr_str]))

        rows.append(
            f"{icon} **{name}（{r['code']}）** | {r['signal']} | 涨概率 {r['rise_prob']:.1%} | "
            f"置信 {r['confidence']}\n　　{price_line}"
        )

    body = "\n".join(rows) if rows else "（无候选股）"

    content = (
        f"**📡 Antenna 每日扫描 · {date_str}**\n\n"
        f"> 共扫描 **{total}** 只，跳过 {skipped} 只\n"
        f"> 强势线 {thresh_strong:.1%} | 关注线 {thresh_watch:.1%}\n\n"
        f"{body}"
    )

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"📡 Antenna 扫描报告 {date_str}"},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": content,
                }
            ],
        },
    }
    return _post(webhook_url, payload)


def send_predict_results(webhook_url: str, results: list, now: datetime = None,
                         app_token: str = "", chat_ids: list = None) -> bool:
    """推送自选股盘中预测结果到飞书（文字摘要卡片 + 逐只分时图）。"""
    if not webhook_url:
        print("[飞书推送] 未配置 webhook_url，跳过推送。")
        return False

    now = now or datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = now.strftime("%Y-%m-%d")

    signal_order = {"买入": 0, "观望": 1, "回避": 2}
    icons        = {"买入": "🔥", "观望": "👀", "回避": "💤"}
    results_sorted = sorted(results, key=lambda r: (signal_order.get(r["signal"], 9), -r["rise_prob"]))

    # ── 文字摘要卡片（via Webhook）────────────────────────
    rows = []
    for r in results_sorted:
        icon   = icons.get(r["signal"], "")
        reason = r.get("reason", "")
        kline  = r.get("text_kline", "")
        raw_name = r.get("name", "")
        name     = raw_name.replace("*", "＊").replace("_", "\\_")
        label    = f"{name}（{r['code']}）" if name and name != r["code"] else r["code"]

        header = (
            f"{icon} **{label}**　{r['signal']} | "
            f"涨概率 {r['rise_prob']:.1%} | 置信 {r['confidence']}"
        )
        kline_block = "\n".join(f"　　{line}" for line in kline.split("\n")) if kline else ""
        reason_line = f"　　📊 {reason}" if reason else ""

        parts = [header]
        if kline_block:
            parts.append(kline_block)
        if reason_line:
            parts.append(reason_line)
        rows.append("\n".join(parts))

    body    = "\n\n".join(rows) if rows else "（无数据）"
    content = (
        f"**📈 盘中预测 · {date_str} {time_str}**\n"
        f"> 自选股共 {len(results)} 只\n\n"
        f"{body}"
    )
    top_signal = results_sorted[0]["signal"] if results_sorted else "观望"
    color_map  = {"买入": "red", "观望": "orange", "回避": "grey"}

    ok = _post(webhook_url, {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"📈 盘中预测 {time_str}"},
                "template": color_map.get(top_signal, "blue"),
            },
            "elements": [{"tag": "markdown", "content": content}],
        },
    })
    return ok


def send_review_report(webhook_url: str, report: dict) -> bool:
    """推送每日回测 + 策略自优化报告到飞书。"""
    if not webhook_url:
        return False

    date_str   = report["date"]
    dr         = report.get("day_result")
    acc_7d     = report.get("acc_7d", 0)
    acc_30d    = report.get("acc_30d", 0)
    t7         = report.get("samples_7", 0)
    t30        = report.get("samples_30", 0)
    change     = report.get("change_desc", "")
    strategy   = report.get("strategy", {})
    buy_top_pct = strategy.get("buy_top_pct", 0.15)
    target      = strategy.get("target_accuracy", 0.55)

    BAR = 20

    def _acc_bar_line(hits: int, total: int, label: str) -> str:
        if total == 0:
            return f"{label}：暂无买入样本"
        acc = hits / total
        fill = max(0, min(BAR, round(acc * BAR)))
        bar  = "█" * fill + "░" * (BAR - fill)
        icon = "✅" if acc >= target else ("⚠️" if acc >= target - 0.10 else "❌")
        return f"{label} {icon}  `{bar}`  **{acc:.0%}**（{hits}/{total} 命中）"

    # ── 今日明细（与 cmd_review 规则对齐）─────────────────
    elements = []

    if dr and dr.get("details"):
        details  = dr["details"]
        rec_buy  = [d for d in details if d.get("is_buy")]
        rec_watch = [d for d in details if not d.get("is_buy")]

        def _safe_label(d: dict) -> str:
            """返回 '名称（代码）' 标准格式，兼容旧快照缺 name 字段。"""
            code = d["code"]
            name = d.get("name", "")
            if not name or name == code:
                name = code
            safe = name.replace("*", "＊").replace("_", "\\_")
            return f"{safe}（{code}）"

        # 买入信号区
        if rec_buy:
            buy_lines = ["**买入信号（计入精准率）**"]
            for d in rec_buy:
                apct     = d.get("actual_pct")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                hit_icon = "✅" if d.get("hit") is True else ("❌" if d.get("hit") is False else "⬜")
                buy_lines.append(
                    f"· {hit_icon} **{_safe_label(d)}**  "
                    f"涨概率 {d['rise_prob']:.1%}  实际 {apct_str}"
                )
            rec_hits  = sum(1 for d in rec_buy if d.get("hit") is True)
            rec_total = len(rec_buy)
            buy_lines.append("")
            buy_lines.append(_acc_bar_line(rec_hits, rec_total, "今日精准率"))
            elements.append({"tag": "markdown", "content": "\n".join(buy_lines)})

        # 观望/回避区（仅列表，不参与精准率）
        if rec_watch:
            elements.append({"tag": "hr"})
            watch_lines = ["**观望/回避（不计入精准率，仅供参考）**"]
            for d in rec_watch:
                apct     = d.get("actual_pct")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                watch_lines.append(
                    f"· ⬜ {_safe_label(d)}  "
                    f"涨概率 {d['rise_prob']:.1%}  实际 {apct_str}"
                )
            elements.append({"tag": "markdown", "content": "\n".join(watch_lines)})

    else:
        elements.append({"tag": "markdown", "content": "（无今日预测记录）"})

    # ── 准确率趋势 ─────────────────────────────────────────
    elements.append({"tag": "hr"})

    def _bar(acc: float) -> str:
        fill = max(0, min(BAR, round(acc * BAR)))
        bar  = "█" * fill + "░" * (BAR - fill)
        icon = "✅" if acc >= target else ("⚠️" if acc >= target - 0.10 else "❌")
        return f"`{bar}` {acc:.1%} {icon}"

    acc_section = (
        f"近 7 日　{_bar(acc_7d)}　（{t7} 条样本）\n"
        f"近30 日　{_bar(acc_30d)}　（{t30} 条样本）\n"
        f"目标精准率　**{target:.0%}**"
    )
    elements.append({"tag": "markdown", "content": f"**📈 精准率追踪**\n{acc_section}"})

    # ── 策略调整 ───────────────────────────────────────────
    elements.append({"tag": "hr"})
    strategy_md = (
        f"当前买入门槛　涨概率前 **{buy_top_pct:.0%}**\n"
        f"调整说明　{change}"
    )
    elements.append({"tag": "markdown", "content": f"**⚙️ 策略自优化**\n{strategy_md}"})

    header_color = "green" if acc_7d >= target else ("orange" if acc_7d >= target - 0.10 else "red")

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"📊 每日复盘报告　{date_str}"},
                "template": header_color,
            },
            "elements": elements,
        },
    }
    return _post(webhook_url, payload)


def send_train_complete(webhook_url: str, elapsed_seconds: float = 0) -> bool:
    """训练完成后推送简短通知。"""
    if not webhook_url:
        return False

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    elapsed = f"{elapsed_seconds:.0f}s" if elapsed_seconds else "未知"

    payload = {
        "msg_type": "text",
        "content": {
            "text": f"✅ Antenna 模型训练完成\n时间：{date_str}\n耗时：{elapsed}"
        },
    }
    return _post(webhook_url, payload)


send_train_done = send_train_complete   # 兼容 task_train.py
