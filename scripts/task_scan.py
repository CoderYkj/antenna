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

    webhook_url = config.get("feishu", {}).get("webhook_url", "")
    top_n = config.get("feishu", {}).get("scan_top", 20)

    print(f"[task_scan] 开始扫描 Top {top_n} ...")

    import argparse
    from cli import cmd_scan
    args = argparse.Namespace(top=top_n, workers=8)
    scan_result = cmd_scan(args, config)

    if scan_result is None:
        print("[task_scan] cmd_scan 无返回值，推送跳过。")
        return

    print(f"[task_scan] 扫描完成，推送飞书 ...")
    from notify.feishu import send_scan_result
    ok = send_scan_result(webhook_url, scan_result)
    if ok:
        print("[task_scan] 飞书通知已发送。")
    elif webhook_url:
        print("[task_scan] 飞书通知发送失败。")
    else:
        print("[task_scan] 未配置 webhook_url，跳过推送。")

    # ── 写入扫描预测快照（与自选股合并，供复盘使用）────────
    from learning.tracker import log_predictions
    from learning.optimizer import load_strategy
    from datetime import datetime
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
        })
    if snapshot:
        log_predictions(datetime.now().strftime("%Y-%m-%d"), snapshot)
        print(f"[task_scan] 已写入 {len(snapshot)} 只扫描预测快照（复盘基准）")


if __name__ == "__main__":
    main()
