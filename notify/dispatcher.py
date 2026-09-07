"""
多通道推送扇出 — 飞书 + 企业微信。

调用方从 notify.feishu 迁移到此模块后，无需关心通道细节：
  from notify.dispatcher import send_scan_result
  send_scan_result(scan_result)

通道开关由 config.yaml.channels 控制；任一通道失败不影响其他。
"""
import logging
import threading
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


# ── 配置加载 ──────────────────────────────────────────────────────────────────

def _load_config() -> tuple[dict, str]:
    """返回 (channels_dict, feishu_webhook_url)。"""
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        channels = cfg.get("channels", {"feishu": True, "wecom": False})
        webhook = cfg.get("feishu", {}).get("webhook_url", "") or ""
        return channels, webhook
    except Exception as e:
        log.warning("[dispatcher] 读取 config.yaml 失败: %s", e)
        return {"feishu": True, "wecom": False}, ""


# ── 并行扇出 ──────────────────────────────────────────────────────────────────

def _fanout(tasks: list[tuple[str, Any, tuple, dict]]) -> dict[str, bool]:
    """并行执行 (name, fn, args, kwargs)，返回各通道结果。"""
    results: dict[str, bool] = {}
    lock = threading.Lock()

    def _run(name: str, fn: Any, args: tuple, kwargs: dict) -> None:
        try:
            r = fn(*args, **kwargs)
            with lock:
                results[name] = bool(r)
        except Exception as e:
            log.warning("[dispatcher] 通道 %s 异常: %s", name, e)
            with lock:
                results[name] = False

    threads = [
        threading.Thread(target=_run, args=(n, f, a, k), daemon=True)
        for n, f, a, k in tasks
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    return results


# ── 公共 API ──────────────────────────────────────────────────────────────────

def send_scan_result(scan_result: dict) -> dict[str, bool]:
    from notify import feishu, wecom
    channels, webhook = _load_config()
    tasks = []
    if channels.get("feishu", True):
        tasks.append(("feishu", feishu.send_scan_result, (webhook, scan_result), {}))
    if channels.get("wecom", False):
        tasks.append(("wecom", wecom.send_scan_result, (scan_result,), {}))
    return _fanout(tasks)


def send_predict_results(results: list, now: datetime = None) -> dict[str, bool]:
    from notify import feishu, wecom
    channels, webhook = _load_config()
    tasks = []
    if channels.get("feishu", True):
        tasks.append(("feishu", feishu.send_predict_results, (webhook, results, now), {}))
    if channels.get("wecom", False):
        tasks.append(("wecom", wecom.send_predict_results, (results, now), {}))
    return _fanout(tasks)


def send_review_report(report: dict) -> dict[str, bool]:
    from notify import feishu, wecom
    channels, webhook = _load_config()
    tasks = []
    if channels.get("feishu", True):
        tasks.append(("feishu", feishu.send_review_report, (webhook, report), {}))
    if channels.get("wecom", False):
        tasks.append(("wecom", wecom.send_review_report, (report,), {}))
    return _fanout(tasks)


def send_train_complete(elapsed_seconds: float = 0) -> dict[str, bool]:
    from notify import feishu, wecom
    channels, webhook = _load_config()
    tasks = []
    if channels.get("feishu", True):
        tasks.append(("feishu", feishu.send_train_complete, (webhook, elapsed_seconds), {}))
    if channels.get("wecom", False):
        tasks.append(("wecom", wecom.send_train_complete, (elapsed_seconds,), {}))
    return _fanout(tasks)


send_train_done = send_train_complete


def send_learn_complete(results: dict, elapsed_seconds: float = 0) -> dict[str, bool]:
    from notify import feishu, wecom
    channels, webhook = _load_config()
    tasks = []
    if channels.get("feishu", True):
        tasks.append(("feishu", feishu.send_learn_complete, (webhook, results, elapsed_seconds), {}))
    if channels.get("wecom", False):
        tasks.append(("wecom", wecom.send_learn_complete, (results, elapsed_seconds), {}))
    return _fanout(tasks)


def send_text(text: str) -> dict[str, bool]:
    from notify import feishu, wecom
    channels, webhook = _load_config()
    tasks = []
    if channels.get("feishu", True):
        tasks.append(("feishu", feishu.send_text, (webhook, text), {}))
    if channels.get("wecom", False):
        tasks.append(("wecom", wecom.send_text, (text,), {}))
    return _fanout(tasks)
