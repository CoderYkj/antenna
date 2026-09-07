"""
企业微信自建应用消息推送。

公共函数签名与 notify/feishu.py 对称，但不接受 webhook_url 参数——
凭据从 config.yaml.wecom 读取。
"""
import json
import logging
import threading
import time
import urllib.request
from datetime import datetime

log = logging.getLogger(__name__)

_WECOM_API = "https://qyapi.weixin.qq.com/cgi-bin"
_MAX_MD_BYTES = 4096

_TOKEN_CACHE: dict = {"token": "", "expires_at": 0.0}
_TOKEN_LOCK = threading.Lock()


# ── 配置加载 ──────────────────────────────────────────────────────────────────

def _load_cfg() -> tuple[bool, dict]:
    """返回 (wecom_enabled, wecom_config_dict)。失败时返回 (False, {})。"""
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        enabled = cfg.get("channels", {}).get("wecom", False)
        return bool(enabled), cfg.get("wecom", {})
    except Exception as e:
        log.warning("[wecom] 读取 config.yaml 失败: %s", e)
        return False, {}


# ── access_token 管理 ─────────────────────────────────────────────────────────

def _get_access_token(corp_id: str, secret: str) -> str:
    """获取并缓存 access_token，提前 60 秒自动刷新。"""
    now = time.time()
    with _TOKEN_LOCK:
        if _TOKEN_CACHE["token"] and _TOKEN_CACHE["expires_at"] - now > 60:
            return _TOKEN_CACHE["token"]
        url = f"{_WECOM_API}/gettoken?corpid={corp_id}&corpsecret={secret}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("errcode", -1) != 0:
                log.warning("[wecom] gettoken 失败: %s", data)
                return ""
            _TOKEN_CACHE["token"] = data["access_token"]
            _TOKEN_CACHE["expires_at"] = now + data.get("expires_in", 7200)
            return _TOKEN_CACHE["token"]
        except Exception as e:
            log.warning("[wecom] gettoken 请求失败: %s", e)
            return ""


# ── 底层发送 ──────────────────────────────────────────────────────────────────

def _post(payload: dict, _retry: bool = True) -> bool:
    """调用 message/send 接口，token 失效时自动重取一次。"""
    enabled, wcfg = _load_cfg()
    if not enabled:
        return False

    corp_id = wcfg.get("corp_id", "")
    secret = wcfg.get("secret", "")
    agent_id = wcfg.get("agent_id", 0)
    if not corp_id or not secret or not agent_id:
        log.warning("[wecom] corp_id/secret/agent_id 未配置，跳过推送")
        return False

    token = _get_access_token(corp_id, secret)
    if not token:
        return False

    payload.setdefault("agentid", agent_id)
    payload.setdefault("touser", "@all")

    url = f"{_WECOM_API}/message/send?access_token={token}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("errcode", -1) == 0:
            return True
        if result.get("errcode") == 40014 and _retry:
            # token 失效，清除缓存后重试一次
            with _TOKEN_LOCK:
                _TOKEN_CACHE["token"] = ""
                _TOKEN_CACHE["expires_at"] = 0.0
            return _post(payload, _retry=False)
        log.warning("[wecom] message/send 错误: %s", result)
        return False
    except Exception as e:
        log.warning("[wecom] message/send 请求失败: %s", e)
        return False


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _md(text: str) -> dict:
    """包装为企业微信 markdown 消息体。"""
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_MD_BYTES:
        text = encoded[:_MAX_MD_BYTES - 3].decode("utf-8", errors="ignore") + "…"
    return {"msgtype": "markdown", "markdown": {"content": text}}


def _txt(text: str) -> dict:
    return {"msgtype": "text", "text": {"content": text}}


# ── 公共 API（与 notify/feishu.py 对称，去掉 webhook_url 参数）────────────────

def send_text(text: str) -> bool:
    return _post(_txt(text))


def send_scan_result(scan_result: dict) -> bool:
    top = scan_result["top"]
    total = scan_result["total"]
    skipped = scan_result["skipped"]
    thresh_strong = scan_result["thresh_strong"]
    thresh_watch = scan_result["thresh_watch"]
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    icons = {"强势": "🔥", "关注": "👀", "观望": "💤"}
    rows = []
    for r in top:
        icon = icons.get(r["signal"], "")
        name = r.get("name", r["code"]).replace("*", "＊").replace("_", "\\_")
        trade = r.get("trade", {})
        buy_str = f"买入 **{trade['buy_price']}**（{trade['buy_desc']}）" if trade.get("buy_price") else ""
        sell_str = f"止盈 **{trade['sell_price']}**（{trade['sell_desc']}）" if trade.get("sell_price") else ""
        stop_str = f"止损 **{trade['stop_price']}**（{trade['stop_desc']}）" if trade.get("stop_price") else ""
        rr_str = f"盈亏比 **{trade['rr_ratio']}:1**" if trade.get("rr_ratio") else ""
        price_line = "　".join(filter(None, [buy_str, sell_str, stop_str, rr_str]))
        rows.append(
            f"{icon} **{name}（{r['code']}）** | {r['signal']} | "
            f"涨概率 {r['rise_prob']:.1%} | 置信 {r['confidence']}\n　　{price_line}"
        )

    body = "\n".join(rows) if rows else "（无候选股）"
    content = (
        f"**📡 Antenna 扫描报告 · {date_str}**\n\n"
        f"> 共扫描 **{total}** 只，跳过 {skipped} 只\n"
        f"> 强势线 {thresh_strong:.1%} | 关注线 {thresh_watch:.1%}\n\n"
        f"{body}"
    )
    return _post(_md(content))


def send_predict_results(results: list, now: datetime = None) -> bool:
    now = now or datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = now.strftime("%Y-%m-%d")

    signal_order = {"买入": 0, "观望": 1, "回避": 2}
    icons = {"买入": "🔥", "观望": "👀", "回避": "💤"}
    results_sorted = sorted(results, key=lambda r: (signal_order.get(r["signal"], 9), -r["rise_prob"]))

    rows = []
    for r in results_sorted:
        icon = icons.get(r["signal"], "")
        name = r.get("name", "").replace("*", "＊").replace("_", "\\_")
        label = f"{name}（{r['code']}）" if name and name != r["code"] else r["code"]
        kline = r.get("text_kline", "")
        reason = r.get("reason", "")
        header = (
            f"{icon} **{label}**　{r['signal']} | "
            f"涨概率 {r['rise_prob']:.1%} | 置信 {r['confidence']}"
        )
        parts = [header]
        if kline:
            parts.append("\n".join(f"　　{line}" for line in kline.split("\n")))
        if reason:
            parts.append(f"　　📊 {reason}")
        rows.append("\n".join(parts))

    body = "\n\n".join(rows) if rows else "（无数据）"
    content = (
        f"**📈 盘中预测 · {date_str} {time_str}**\n"
        f"> 自选股共 {len(results)} 只\n\n"
        f"{body}"
    )
    return _post(_md(content))


def send_review_report(report: dict) -> bool:
    date_str = report["date"]
    dr = report.get("day_result")
    acc_7d = report.get("acc_7d", 0)
    acc_30d = report.get("acc_30d", 0)
    t7 = report.get("samples_7", 0)
    t30 = report.get("samples_30", 0)
    change = report.get("change_desc", "")
    strategy = report.get("strategy", {})
    buy_top_pct = strategy.get("buy_top_pct", 0.15)
    target = strategy.get("target_accuracy", 0.55)

    BAR = 20

    def _bar(acc: float) -> str:
        fill = max(0, min(BAR, round(acc * BAR)))
        bar = "█" * fill + "░" * (BAR - fill)
        icon = "✅" if acc >= target else ("⚠️" if acc >= target - 0.10 else "❌")
        return f"`{bar}` {acc:.1%} {icon}"

    def _safe_label(d: dict) -> str:
        code = d["code"]
        name = d.get("name", "")
        if not name or name == code:
            try:
                from data.fetcher import _load_name_map
                name = _load_name_map().get(code, "")
            except Exception:
                name = ""
        safe = (name or "").replace("*", "＊").replace("_", "\\_")
        return f"{safe}({code})" if safe else code

    parts = [f"**📊 每日复盘报告　{date_str}**"]

    if dr and dr.get("details"):
        details = dr["details"]
        rec_buy = [d for d in details if d.get("is_buy")]
        rec_watch = [d for d in details if not d.get("is_buy")]

        if rec_buy:
            lines = ["**买入信号（计入精准率）**"]
            for d in rec_buy:
                apct = d.get("actual_pct")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                hit_icon = "✅" if d.get("hit") is True else ("❌" if d.get("hit") is False else "⬜")
                lines.append(f"· {hit_icon} **{_safe_label(d)}**  涨概率 {d['rise_prob']:.1%}  实际 {apct_str}")
            rec_hits = sum(1 for d in rec_buy if d.get("hit") is True)
            if rec_buy:
                acc = rec_hits / len(rec_buy)
                fill = max(0, min(BAR, round(acc * BAR)))
                bar = "█" * fill + "░" * (BAR - fill)
                hi = "✅" if acc >= target else ("⚠️" if acc >= target - 0.10 else "❌")
                lines.append(f"\n今日精准率 {hi}  `{bar}`  **{acc:.0%}**（{rec_hits}/{len(rec_buy)} 命中）")
            parts.append("\n".join(lines))

        if rec_watch:
            parts.append("---")
            lines = ["**观望/回避（不计入精准率）**"]
            for d in rec_watch:
                apct = d.get("actual_pct")
                apct_str = f"{apct:+.2f}%" if apct is not None else "待结算"
                lines.append(f"· ⬜ {_safe_label(d)}  涨概率 {d['rise_prob']:.1%}  实际 {apct_str}")
            parts.append("\n".join(lines))
    else:
        parts.append("（无今日预测记录）")

    parts.append("---")
    parts.append(
        f"**📈 精准率追踪**\n"
        f"近 7 日　{_bar(acc_7d)}　（{t7} 条样本）\n"
        f"近30 日　{_bar(acc_30d)}　（{t30} 条样本）\n"
        f"目标精准率　**{target:.0%}**"
    )
    parts.append("---")
    parts.append(
        f"**⚙️ 策略自优化**\n"
        f"当前买入门槛　涨概率前 **{buy_top_pct:.0%}**\n"
        f"调整说明　{change}"
    )
    return _post(_md("\n\n".join(parts)))


def send_train_complete(elapsed_seconds: float = 0) -> bool:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    elapsed = f"{elapsed_seconds:.0f}s" if elapsed_seconds else "未知"
    return send_text(f"✅ Antenna 模型训练完成\n时间：{date_str}\n耗时：{elapsed}")


send_train_done = send_train_complete


def send_learn_complete(results: dict, elapsed_seconds: float = 0) -> bool:
    _STATUS_ICON = {"ok": "✅", "failed": "❌", "skipped": "⏭", "dry_run": "🔍"}
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    elapsed = f"{elapsed_seconds:.0f}s" if elapsed_seconds else ""

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")

    header = f"{'❌' if failed else '✅'} Antenna 学习完成  {date_str}"
    if elapsed:
        header += f"  耗时 {elapsed}"
    summary = f"ok={ok}  failed={failed}  skipped={skipped}"

    lines = [header, summary, ""]
    for name, r in results.items():
        icon = _STATUS_ICON.get(r["status"], "❓")
        error = f"  → {r['error']}" if r["status"] == "failed" and r.get("error") else ""
        lines.append(f"{icon} {name}{error}")

    return send_text("\n".join(lines))
