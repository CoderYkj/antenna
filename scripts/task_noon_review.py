"""
task_noon_review.py - 午间休市（11:30-13:00）自动执行：

  1. 拍摄 11:30 实时行情快照，与开盘预测对比，生成上午走势反思
  2. 基于最新缓存重训模型（watchlist + 全量，取决于 retrain_mode）
  3. 用新模型重新研判自选股，标注信号变化
  4. 组装综合报告推送飞书

调度：Windows 任务计划程序，11:32 运行（确保上午盘已收）。
  python e:\\antenna\\scripts\\task_noon_review.py
"""
import sys
import io
import os
import time
import json
from datetime import datetime, time as dtime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml


def _load_cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_trading_day(dt: datetime) -> bool:
    try:
        import exchange_calendars as xcals
        import pandas as pd
        return xcals.get_calendar("XSHG").is_session(pd.Timestamp(dt.date()))
    except Exception:
        return dt.weekday() < 5


# ── 上午走势反思 ───────────────────────────────────────────────

def _morning_signal(rise_prob: float) -> str:
    if rise_prob >= 0.60:
        return "买入"
    elif rise_prob <= 0.40:
        return "回避"
    return "观望"


def build_morning_reflection(codes: list, rt_now: dict,
                              model, FEATURE_COLS: list) -> list:
    """
    对每只自选股：
      - 用历史数据跑模型，得到盘前信号
      - 对比上午实际涨跌（open → 11:30）
      - 生成反思文字

    Returns: list of dict per stock
    """
    from data.fetcher import fetch_stock_hist, fetch_intraday_kline
    from features.builder import build_features
    from models.predictor import predict

    items = []
    for code in codes:
        try:
            df = fetch_stock_hist(code, days=365)
            df = build_features(df)
            r  = predict(df, FEATURE_COLS, model=model)

            rt = rt_now.get(code, {})
            raw_name = rt.get("name", code)
            name = raw_name.replace("*", "＊").replace("_", "\\_")

            rise_prob = r.get("rise_prob", 0.5)
            signal    = _morning_signal(rise_prob)

            # 上午实际表现：开盘→11:30 涨幅
            cur  = rt.get("price")
            open_p = rt.get("open")
            if cur and open_p and open_p > 0:
                am_pct = (cur - open_p) / open_p * 100
            else:
                am_pct = None

            # 方向是否一致
            if am_pct is None:
                align = "无行情"
                reflect = "暂无实时数据，无法判断。"
            else:
                dir_up = am_pct >= 0
                pred_up = signal == "买入"
                pred_down = signal == "回避"

                if pred_up and dir_up:
                    align = "✅ 看多正确"
                    reflect = (f"盘前看多（涨概率 {rise_prob:.1%}），"
                               f"上午确实上涨 {am_pct:+.2f}%，判断验证。")
                elif pred_up and not dir_up:
                    align = "❌ 看多偏差"
                    reflect = (f"盘前看多（涨概率 {rise_prob:.1%}），"
                               f"但上午回落 {am_pct:.2f}%，多头预期未兑现，"
                               "午后需重点关注量能与支撑位。")
                elif pred_down and not dir_up:
                    align = "✅ 看空正确"
                    reflect = (f"盘前回避（涨概率 {rise_prob:.1%}），"
                               f"上午确实下跌 {am_pct:.2f}%，判断验证。")
                elif pred_down and dir_up:
                    align = "❌ 看空偏差"
                    reflect = (f"盘前回避（涨概率 {rise_prob:.1%}），"
                               f"但上午反弹 {am_pct:+.2f}%，空头预期落空，"
                               "午后注意是否形成反转信号。")
                else:  # 观望
                    if abs(am_pct) >= 2.0:
                        align = "⚠️ 观望但大幅波动"
                        direction = "上涨" if dir_up else "下跌"
                        reflect = (f"盘前观望，上午却 {direction} {abs(am_pct):.2f}%，"
                                   "信号可能滞后，午后重新评估趋势。")
                    else:
                        align = "➖ 观望震荡"
                        reflect = (f"盘前观望，上午温和波动 {am_pct:+.2f}%，"
                                   "与观望判断一致。")

            items.append({
                "code":      code,
                "name":      name,
                "raw_name":  raw_name,
                "rise_prob": rise_prob,
                "signal":    signal,
                "am_pct":    am_pct,
                "align":     align,
                "reflect":   reflect,
            })
        except Exception as e:
            print(f"  [noon] {code} 反思失败: {e}")

    return items


# ── 模型重训（自选股快速 + 可选全量）────────────────────────────

def retrain_model(config: dict, mode: str = "watchlist") -> tuple:
    """
    重训模型。
    mode="watchlist" 只用自选股（快，数秒），适合午间。
    mode="full"      全量宇宙（慢，数分钟）。
    Returns: (model, auc, elapsed_sec)
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import fetch_stock_hist, cached_codes
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.trainer import build_labels, train, save_model

    target_days = config["model"]["target_days"]
    threshold   = config["model"]["threshold"]
    days        = config["data"]["default_days"]

    if mode == "watchlist":
        codes = config["universe"]["watchlist"]
        label = f"自选股 {len(codes)} 只"
    else:
        codes = load_universe(config)
        label = f"全量 {len(codes)} 只"

    print(f"  [noon/train] 重训 {label} ...")
    t0 = time.time()

    dfs = []
    for code in codes:
        try:
            df = fetch_stock_hist(code, days=days, cache_only=True)
            df = build_features(df)
            df["label"] = build_labels(df, target_days=target_days, threshold=threshold)
            df["code"]  = code
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("无可用缓存数据")

    import pandas as pd
    combined = pd.concat(dfs, ignore_index=True)
    model = train(combined, feature_cols=FEATURE_COLS)
    save_model(model, saved_dir=config["model"]["saved_dir"])
    elapsed = time.time() - t0
    print(f"  [noon/train] 完成，耗时 {elapsed:.0f}s，训练样本 {len(combined)} 行")
    return model, elapsed


# ── 下午重新研判 ───────────────────────────────────────────────

def afternoon_predict(codes: list, model, rt_now: dict,
                      FEATURE_COLS: list) -> list:
    """
    基于新模型重新预测各自选股，标注与上午信号的变化。
    Returns: list of dict per stock
    """
    from data.fetcher import fetch_stock_hist
    from features.builder import build_features
    from features.analyser import analyse, predict_range, suggest_holding, build_commentary
    from models.predictor import predict

    results = []
    for code in codes:
        try:
            df = fetch_stock_hist(code, days=365)
            df = build_features(df)
            r  = predict(df, FEATURE_COLS, model=model)

            rt       = rt_now.get(code, {})
            raw_name = rt.get("name", code)
            name     = raw_name.replace("*", "\\*").replace("_", "\\_")

            last       = df.iloc[-1].to_dict()
            rise_prob  = r.get("rise_prob", 0.5)
            signal     = r.get("signal", "观望")
            confidence = r.get("confidence", "—")

            price_info = predict_range(df, rise_prob)
            if rt:
                price_info.update({k: rt[k] for k in ("price","pct","high","low","open") if k in rt})

            hold_label, hold_reason = suggest_holding(last, rise_prob)
            commentary, basis       = build_commentary(df, r, price_info)
            reason                  = analyse(last)

            cur     = price_info.get("price") or price_info.get("close")
            pct_val = price_info.get("pct")

            results.append({
                "code":       code,
                "name":       name,
                "raw_name":   raw_name,
                "rise_prob":  rise_prob,
                "signal":     signal,
                "confidence": confidence,
                "cur":        cur,
                "pct":        pct_val,
                "hold_label": hold_label,
                "hold_reason": hold_reason,
                "commentary": commentary,
                "basis":      basis,
                "reason":     reason,
            })
        except Exception as e:
            print(f"  [noon/pm] {code} 预测失败: {e}")

    return results


# ── 飞书卡片组装 ───────────────────────────────────────────────

def build_feishu_card(date_str: str, time_str: str,
                      reflection: list, train_elapsed: float,
                      pm_results: list) -> dict:

    elements = []

    # ── 上午走势反思 ─────────────────────────────────────────
    correct   = sum(1 for x in reflection if x["align"].startswith("✅"))
    wrong     = sum(1 for x in reflection if x["align"].startswith("❌"))
    warn      = sum(1 for x in reflection if x["align"].startswith("⚠️"))
    total_ref = len(reflection)
    am_acc    = correct / total_ref if total_ref else 0

    am_header = (
        f"命中 **{correct}/{total_ref}**　"
        f"{'✅' if am_acc >= 0.6 else '⚠️'} 上午准确率 **{am_acc:.0%}**"
    )
    ref_rows = []
    for x in reflection:
        pct_str = f"{x['am_pct']:+.2f}%" if x["am_pct"] is not None else "—"
        arrow   = "🔴▲" if (x["am_pct"] or 0) > 0 else "🟢▼"
        ref_rows.append(
            f"{x['align']}　**{x['name']}（{x['code']}）**　上午 {arrow}{pct_str}\n"
            f"　　{x['reflect']}"
        )
    elements.append({
        "tag": "markdown",
        "content": f"**🔍 上午走势反思　{am_header}**\n" + "\n".join(ref_rows)
    })
    elements.append({"tag": "hr"})

    # ── 模型重训 ─────────────────────────────────────────────
    elements.append({
        "tag": "markdown",
        "content": (
            f"**🤖 午间模型重训**\n"
            f"耗时 **{train_elapsed:.0f}s**　基于自选股最新历史数据重新校准，"
            "信号可信度已更新。"
        )
    })
    elements.append({"tag": "hr"})

    # ── 下午重新研判 ─────────────────────────────────────────
    signal_order = {"买入": 0, "观望": 1, "回避": 2}
    pm_sorted = sorted(pm_results, key=lambda x: (signal_order.get(x["signal"], 9), -x["rise_prob"]))

    pm_rows = []
    for x in pm_sorted:
        icon  = {"买入": "🔥", "观望": "👀", "回避": "💤"}.get(x["signal"], "")
        pct_s = (f"{'🔴▲' if (x['pct'] or 0) > 0 else '🟢▼'}{abs(x['pct'] or 0):.2f}%"
                 if x["pct"] is not None else "")
        pm_rows.append(
            f"{icon} **{x['name']}（{x['code']}）**　{x['cur'] or '—'} {pct_s}\n"
            f"　　{x['signal']}　涨概率 **{x['rise_prob']:.1%}**　置信 {x['confidence']}　⏱ {x['hold_label']}\n"
            f"　　{x['commentary']}\n"
            f"　　**持仓** {x['hold_reason']}"
        )

    elements.append({
        "tag": "markdown",
        "content": f"**📡 下午重新研判　共 {len(pm_results)} 只**\n" + "\n\n".join(pm_rows)
    })

    # 卡片整体颜色
    color = "green" if am_acc >= 0.6 else ("orange" if am_acc >= 0.4 else "red")
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text",
                      "content": f"🔄 午间研判报告　{date_str} {time_str}"},
            "template": color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}


# ── 主流程 ─────────────────────────────────────────────────────

def run():
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    if not _is_trading_day(now):
        print(f"[noon] {date_str} 非交易日，跳过。")
        return

    # 允许 11:25-12:59 之间运行
    t = now.time()
    if not (dtime(11, 25) <= t <= dtime(12, 59)):
        print(f"[noon] {time_str} 非午间时段（11:25-12:59），跳过。")
        return

    cfg = _load_cfg()
    codes       = cfg["universe"]["watchlist"]
    webhook_url = cfg.get("feishu", {}).get("webhook_url", "")

    from data.fetcher import fetch_realtime_prices
    from features.technical import FEATURE_COLS
    from models.predictor import load_model

    print(f"[noon] {time_str} 开始午间研判，自选股 {len(codes)} 只 ...")

    # ── Step 1: 拍摄 11:30 行情快照 ───────────────────────────
    print("[noon] 1/3 拉取实时行情...")
    try:
        rt_now = fetch_realtime_prices(codes)
        print(f"  获取到 {len(rt_now)}/{len(codes)} 只")
    except Exception as e:
        print(f"  实时行情失败: {e}")
        rt_now = {}

    # ── Step 2: 上午走势反思（用当前模型） ────────────────────
    print("[noon] 2/3 生成上午走势反思...")
    try:
        old_model  = load_model(cfg["model"]["saved_dir"])
        reflection = build_morning_reflection(codes, rt_now, old_model, FEATURE_COLS)
        print(f"  反思完成，{len(reflection)} 只")
    except Exception as e:
        print(f"  反思失败: {e}")
        reflection = []

    # ── Step 3: 模型重训（自选股快速模式）─────────────────────
    print("[noon] 3/4 模型重训（watchlist 模式）...")
    try:
        new_model, train_elapsed = retrain_model(cfg, mode="watchlist")
    except Exception as e:
        print(f"  重训失败: {e}，使用旧模型")
        new_model     = old_model if "old_model" in dir() else None
        train_elapsed = 0.0
        if new_model is None:
            print("[noon] 无可用模型，终止。")
            return

    # ── Step 4: 下午重新研判 ──────────────────────────────────
    print("[noon] 4/4 下午重新研判...")
    pm_results = afternoon_predict(codes, new_model, rt_now, FEATURE_COLS)
    print(f"  研判完成，{len(pm_results)} 只")

    # ── Step 5: 推送飞书 ──────────────────────────────────────
    if not webhook_url:
        print("[noon] 未配置 webhook_url，跳过推送。")
        return

    card = build_feishu_card(date_str, time_str, reflection, train_elapsed, pm_results)

    import urllib.request
    data = json.dumps(
        {"msg_type": card["msg_type"],
         "card": json.loads(card["content"])},
        ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            ok = result.get("code", -1) == 0 or result.get("StatusCode", -1) == 0
        print("[noon] 飞书推送成功。" if ok else f"[noon] 飞书推送异常: {result}")
    except Exception as e:
        print(f"[noon] 飞书推送失败: {e}")


if __name__ == "__main__":
    run()
