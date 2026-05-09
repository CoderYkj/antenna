"""
feishu_push.py - 主动推送消息到飞书会话（供后台任务完成后回调）。

用法：
    from server.feishu_push import push
    push("回测完成！")          # 纯文本
    push({"msg_type": ...})    # interactive 卡片 payload
"""
import json
import time
import threading
import requests
import yaml

BASE = "https://open.feishu.cn/open-apis"

_token      = ""
_token_exp  = 0.0
_token_lock = threading.Lock()


def _load_cfg() -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_token() -> str:
    global _token, _token_exp
    with _token_lock:
        if time.time() < _token_exp:
            return _token
        full_cfg = _load_cfg()
        if "feishu" not in full_cfg:
            raise RuntimeError(
                "config.yaml 缺 'feishu' 配置块。请参考 config.example.yaml 复制并填入凭据。"
            )
        cfg = full_cfg["feishu"]
        resp = requests.post(
            f"{BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": cfg["app_id"], "app_secret": cfg["app_secret"]},
            timeout=10,
        ).json()
        _token     = resp.get("tenant_access_token", "")
        _token_exp = time.time() + resp.get("expire", 7200) - 120
        return _token


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type":  "application/json",
    }


def push(reply, chat_ids: list | None = None):
    """
    向配置中的所有 chat_ids 推送消息。
    reply: str（纯文本）或 dict（飞书 interactive 卡片 payload）
    """
    cfg      = _load_cfg()
    targets  = chat_ids or cfg.get("feishu", {}).get("chat_ids", [])
    if not targets:
        print("[feishu_push] 未配置 chat_ids，无法推送")
        return

    if isinstance(reply, dict):
        payload = reply
    else:
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "📡 Antenna"},
                "template": "blue",
            },
            "elements": [{"tag": "markdown", "content": str(reply)}],
        }
        payload = {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}

    for chat_id in targets:
        try:
            resp = requests.post(
                f"{BASE}/im/v1/messages",
                headers=_headers(),
                params={"receive_id_type": "chat_id"},
                json={
                    "receive_id": chat_id,
                    "msg_type":   payload["msg_type"],
                    "content":    payload["content"],
                },
                timeout=10,
            ).json()
            if resp.get("code", -1) != 0:
                print(f"[feishu_push] 推送失败 {chat_id}: {resp.get('msg')}")
        except Exception as e:
            print(f"[feishu_push] 推送异常 {chat_id}: {e}")
