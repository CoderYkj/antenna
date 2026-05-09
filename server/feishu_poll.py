"""
feishu_poll.py - 飞书轮询机器人（无需公网地址/任何账号）

原理：每 5 秒主动调用飞书 API 拉取新消息，处理指令后回复。
支持指令与 feishu_bot.py 相同，见 server/commands.py。

启动：python server/feishu_poll.py
"""
import sys
import io
import os
import json
import re
import time
import threading
import logging

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import requests
import yaml

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot_poll.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

POLL_INTERVAL = 5   # 轮询间隔（秒）
MAX_CMD_SLOTS = 3   # 最大并发重型指令数

# 重型指令：耗时较长（调 LLM / 全量扫描 / 复杂计算），受并发槽位限制
_ALL_HEAVY = frozenset({
    "预测", "分析", "predict", "p",
    "推荐", "扫描", "scan", "s",
    "复盘", "回测", "review", "r",
    "财报", "年报", "季报", "report", "f",
    "历史", "历史回测", "backtest", "bt",
    "新闻", "消息", "资讯", "news", "n",
    "发送", "推送", "广播", "push", "broadcast",
    "学习", "自学习", "learn", "l",
})

BASE = "https://open.feishu.cn/open-apis"


def _load_cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FeishuPoller:
    def __init__(self):
        full_cfg = _load_cfg()
        if "feishu" not in full_cfg:
            raise RuntimeError(
                "config.yaml 缺 'feishu' 配置块。请参考 config.example.yaml 复制并填入凭据 "
                "(app_id / app_secret / chat_ids),详见 README §配置文件。"
            )
        cfg = full_cfg["feishu"]
        self.app_id     = cfg["app_id"]
        self.app_secret = cfg["app_secret"]
        self._cfg_chat_ids = cfg.get("chat_ids", [])   # 启动时读一次，后续直接复用
        self._token        = ""
        self._token_expire = 0.0
        self._seen: set[str] = set()          # 已处理的 message_id
        self._chat_cursor: dict[str, str] = {}  # chat_id -> 最新消息 create_time(ms)
        self.bot_open_id = ""
        self._cmd_lock   = threading.Lock()
        self._cmd_active = 0  # 当前正在处理的重型指令数

    # ── Token ────────────────────────────────────────────────

    def _get_token(self) -> str:
        if time.time() < self._token_expire:
            return self._token
        resp = requests.post(
            f"{BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=10,
        ).json()
        self._token = resp.get("tenant_access_token", "")
        self._token_expire = time.time() + resp.get("expire", 7200) - 120
        return self._token

    def _h(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    # ── Bot Info ─────────────────────────────────────────────

    def _init_bot_info(self):
        resp = requests.get(f"{BASE}/bot/v3/info", headers=self._h(), timeout=10).json()
        self.bot_open_id = resp.get("bot", {}).get("open_id", "")
        log.info(f"Bot open_id: {self.bot_open_id}")

    # ── Chats ────────────────────────────────────────────────

    def _get_chats(self) -> list:
        """优先使用 config.yaml 中指定的 chat_ids（启动时已缓存），否则自动发现。"""
        if self._cfg_chat_ids:
            return [{"chat_id": cid, "chat_type": "group" if cid.startswith("oc_") else "p2p"}
                    for cid in self._cfg_chat_ids]
        try:
            resp = requests.get(
                f"{BASE}/im/v1/chats",
                headers=self._h(),
                params={"page_size": 50},
                timeout=10,
            ).json()
        except Exception as e:
            log.warning(f"获取会话列表失败: {e}")
            return []
        data = resp.get("data")
        if not isinstance(data, dict):
            return []
        items = data.get("items")
        return items if isinstance(items, list) else []

    # ── Messages ─────────────────────────────────────────────

    def _get_messages(self, chat_id: str, since_ms: str) -> list:
        """拉取 chat 中 since_ms 之后的消息（飞书 API 用秒），超时自动重试一次。"""
        since_s = str(int(since_ms) // 1000)
        params  = {
            "container_id_type": "chat",
            "container_id": chat_id,
            "start_time": since_s,
            "page_size": 20,
            "sort_type": "ByCreateTimeAsc",
        }
        for attempt in range(2):
            try:
                resp = requests.get(
                    f"{BASE}/im/v1/messages",
                    headers=self._h(),
                    params=params,
                    timeout=20,
                ).json()
                break
            except Exception as e:
                if attempt == 0:
                    log.debug(f"消息拉取超时，重试 {chat_id}: {e}")
                    time.sleep(1)
                else:
                    log.warning(f"消息拉取请求失败 {chat_id}: {e}")
                    return []
        if resp.get("code", -1) != 0:
            log.debug(f"消息拉取 API 错误 {chat_id}: code={resp.get('code')} msg={resp.get('msg')}")
            return []
        data = resp.get("data")
        if not isinstance(data, dict):
            return []
        items = data.get("items")
        return items if isinstance(items, list) else []

    # ── Reply ────────────────────────────────────────────────

    def _reply(self, msg_id: str, reply):
        """回复消息。reply 为字符串时自动包装为 interactive 卡片，为 dict 时直接发送。"""
        try:
            if isinstance(reply, dict):
                payload = reply
            else:
                # 纯文本统一包装为带 header 的 interactive 卡片
                card = {
                    "config": {"wide_screen_mode": True},
                    "header": {
                        "title": {"tag": "plain_text", "content": "📡 Antenna"},
                        "template": "blue",
                    },
                    "elements": [{"tag": "markdown", "content": str(reply)}],
                }
                payload = {
                    "msg_type": "interactive",
                    "content": json.dumps(card, ensure_ascii=False),
                }
            resp = requests.post(
                f"{BASE}/im/v1/messages/{msg_id}/reply",
                headers=self._h(),
                json=payload,
                timeout=10,
            ).json()
        except Exception as e:
            log.warning(f"回复请求失败: {e}")
            return
        if resp.get("code", -1) != 0:
            log.warning(f"回复失败: code={resp.get('code')} msg={resp.get('msg')}")

    # ── Command Slots ────────────────────────────────────────

    def _is_heavy(self, text: str) -> bool:
        """判断是否为重型指令（需要占用并发槽位）。"""
        cmd = text.strip().split()[0].lower() if text.strip() else ""
        return cmd in _ALL_HEAVY

    # ── Message Processing ───────────────────────────────────

    def _extract_text(self, msg: dict) -> str:
        """从消息体中提取纯文本，去掉 @mention 标记。"""
        try:
            body = msg.get("body", {})
            raw = body.get("content", "{}")
            content = json.loads(raw)
            text = content.get("text", "")
        except Exception:
            return ""
        # 去掉 @_user_xxx 形式的 mention
        text = re.sub(r"@[^\s]+", "", text).strip()
        return text

    def _process(self, msg: dict, msg_id: str, chat_type: str, chat_id: str = ""):
        """处理单条消息（由主循环线程去重后分发，此处不再修改 _seen）。"""
        # 跳过 bot 自己发的消息
        sender_type = msg.get("sender", {}).get("sender_type", "")
        if sender_type == "app":
            return

        # 群聊中只响应 @机器人 的消息
        if chat_type == "group":
            mentions = msg.get("mentions", []) or []
            bot_mentioned = self.bot_open_id and self.bot_open_id in json.dumps(mentions)
            if not bot_mentioned:
                return

        if msg.get("msg_type") != "text":
            return

        text = self._extract_text(msg)
        if not text:
            return

        log.info(f"[{chat_type}] 收到: {text}")
        from server.commands import handle_command

        if self._is_heavy(text):
            # 重型指令：受并发槽位限制
            with self._cmd_lock:
                if self._cmd_active >= MAX_CMD_SLOTS:
                    self._reply(
                        msg_id,
                        f"🚦 当前已有 {MAX_CMD_SLOTS} 条指令在处理中，请稍候再发～\n闲聊随时可用，不占用槽位。",
                    )
                    return
                self._cmd_active += 1
            try:
                reply = handle_command(text, chat_id=chat_id)
            except Exception as e:
                log.error(f"指令执行出错 [{text!r}]: {e}", exc_info=True)
                reply = f"❌ 指令执行出错：{e}"
            finally:
                with self._cmd_lock:
                    self._cmd_active -= 1
        else:
            # 轻型指令 / 闲聊：直接处理，不占槽位
            try:
                reply = handle_command(text, chat_id=chat_id)
            except Exception as e:
                log.error(f"指令执行出错 [{text!r}]: {e}", exc_info=True)
                reply = f"❌ 指令执行出错：{e}"

        self._reply(msg_id, reply)
        log.info(f"已回复: {str(reply)[:80]}")

    # ── Main Loop ────────────────────────────────────────────

    def run(self):
        log.info("初始化...")
        self._init_bot_info()

        # 初始化各 chat 的游标为当前时间，避免处理历史消息
        now_ms = str(int(time.time() * 1000))
        chats = self._get_chats()
        for chat in chats:
            self._chat_cursor[chat["chat_id"]] = now_ms
        log.info(f"监控 {len(chats)} 个会话，轮询间隔 {POLL_INTERVAL}s")
        log.info("已就绪，等待飞书消息 ...")

        while True:
            try:
                chats = self._get_chats()
                for chat in chats:
                    chat_id   = chat.get("chat_id", "")
                    chat_type = chat.get("chat_type", "p2p")
                    since     = self._chat_cursor.get(chat_id, now_ms)

                    msgs = self._get_messages(chat_id, since)
                    for msg in msgs:
                        ct = msg.get("create_time", "0")
                        if int(ct) > int(self._chat_cursor.get(chat_id, "0")):
                            self._chat_cursor[chat_id] = ct

                        # 去重在主循环（单线程）中完成，保证原子性，不存在竞态
                        msg_id = msg.get("message_id", "")
                        if not msg_id or msg_id in self._seen:
                            continue
                        self._seen.add(msg_id)
                        if len(self._seen) > 2000:
                            self._seen = set(list(self._seen)[-1000:])

                        threading.Thread(
                            target=self._process,
                            args=(msg, msg_id, chat_type),
                            kwargs={"chat_id": chat_id},
                            daemon=True,
                        ).start()

            except Exception as e:
                log.error(f"轮询出错: {e}", exc_info=True)

            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    FeishuPoller().run()
