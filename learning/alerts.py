"""
alerts.py - 学习系统告警:飞书 webhook + 本地 alerts.jsonl + stdout。

所有告警都三路同时落:
  1. 写 learning/alerts.jsonl(permanent,人工复盘用)
  2. 尝试发飞书(若 webhook 配置)
  3. 打印到 stdout

用法:
  try:
      module.run()
  except Exception as e:
      alerts.send_alert(module.__name__, f"{e}", traceback=True)
"""
import json
import traceback as tb_mod
from datetime import datetime
from pathlib import Path

ALERTS_FILE = Path("learning/alerts.jsonl")


def _get_webhook() -> str:
    """读 config.yaml 拿飞书 webhook。"""
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("feishu", {}).get("webhook_url", "") or ""
    except Exception:
        return ""


def _push_feishu(webhook: str, msg: str) -> bool:
    """发送纯文本告警到飞书 webhook。失败不抛。"""
    try:
        from notify.feishu import send_text
        return send_text(webhook, msg)
    except Exception:
        return False


def record_alert(module: str, message: str, severity: str = "error") -> None:
    """只写 alerts.jsonl + stdout,不发飞书。"""
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":       datetime.now().isoformat(timespec="seconds"),
        "module":   module,
        "severity": severity,
        "message":  message,
    }
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[alert:{severity}] {module}: {message}")


def send_alert(module: str, message: str, severity: str = "error", traceback: bool = False) -> None:
    """完整告警:alerts.jsonl + 飞书 + stdout。"""
    tb_str = ""
    if traceback:
        tb_str = "\n" + tb_mod.format_exc()

    full_msg = f"⚠️ [{module}] {message}{tb_str}"
    record_alert(module, message + tb_str, severity)

    webhook = _get_webhook()
    if webhook:
        _push_feishu(webhook, full_msg)
