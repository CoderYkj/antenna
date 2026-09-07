"""
周一至周五 9:15 定时任务：扫描全市场，Top 结果推送飞书。
由 Windows 任务计划程序调用：
  python e:\\antenna\\scripts\\task_scan.py
"""
import sys
import io
import os

# 强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    top_n = config.get("feishu", {}).get("scan_top", 20)

    print(f"[task_scan] 开始扫描 Top {top_n} ...")

    # 预热 alt_data 缓存（bot 扫描路径 cache_only=True，依赖此处提前拉取）
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        from data.fetcher import cached_codes
        from data.alt_fetcher import fetch_alt_features
        all_codes = cached_codes()
        if all_codes:
            print(f"[task_scan] 预热 alt_data 缓存 {len(all_codes)} 只 ...")
            fetch_alt_features(all_codes, today_str)
            print(f"[task_scan] alt_data 缓存预热完成")
    except Exception as _e:
        print(f"[task_scan] alt_data 预热失败（不影响扫描）: {_e}")

    import argparse
    from cli import cmd_scan
    args = argparse.Namespace(top=top_n, workers=8)
    scan_result = cmd_scan(args, config)

    if scan_result is None:
        print("[task_scan] cmd_scan 无返回值，推送跳过。")
        return

    print(f"[task_scan] 扫描完成，推送通知 ...")
    from notify import send_scan_result
    results = send_scan_result(scan_result)
    ok = any(results.values()) if results else False
    if ok:
        print(f"[task_scan] 推送成功: {results}")
    else:
        print(f"[task_scan] 推送失败或未配置通道: {results}")

    # ── 写入扫描预测快照（与自选股合并，供复盘使用）────────
    from learning.tracker import log_predictions
    from learning.optimizer import load_strategy
    buy_threshold = load_strategy().get("buy_threshold", 0.60)
    snapshot = []
    for r in scan_result.get("top", []):
        pi = r.get("price_info", {})
        snapshot.append({
            "code":       r["code"],
            "name":       r.get("name", r["code"]),
            "signal":     "买入" if r["rise_prob"] >= buy_threshold else "观望",
            "rise_prob":  round(r["rise_prob"], 4),
            "confidence": r.get("confidence", ""),
            "pred_high":  pi.get("pred_high"),
            "pred_low":   pi.get("pred_low"),
            "open":       pi.get("open"),
            "scene":      "scan",
        })
    if snapshot:
        log_predictions(datetime.now().strftime("%Y-%m-%d"), snapshot)
        print(f"[task_scan] 已写入 {len(snapshot)} 只扫描预测快照（复盘基准）")


if __name__ == "__main__":
    main()
