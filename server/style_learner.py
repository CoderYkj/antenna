"""
style_learner.py - 从飞书 P2P 聊天记录采集目标同事消息，LLM 提炼语言风格，写入 learning/persona.txt。

工作流：
  1. 从 config.yaml colleague.p2p_chat_id 指定的私聊中拉取近 N 天消息
  2. 过滤掉 bot 自身发送的消息，只保留对方的文本发言
  3. 将发言交给 LLM 分析口语风格
  4. 写入 learning/persona.txt
  5. predict_cmd.py 的 cmd_chat 启动时加载该文件，追加到 system prompt

配置（config.yaml）：
  colleague:
    p2p_chat_id: "p2p_xxx"   # 与目标同事的飞书 P2P 会话 ID
    fetch_days: 60            # 采集最近 N 天
    min_messages: 30          # 样本不足则不更新

用法：
  python cli.py style             # 采集 + 分析 + 更新
  python cli.py style --dry-run   # 只分析，不写文件
  python cli.py style --days 90   # 覆盖采集天数
"""
import json
import logging
import re
import time
import requests
import yaml
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

BASE         = "https://open.feishu.cn/open-apis"
PERSONA_FILE = Path("learning/persona.txt")


# ── 配置 & Token ──────────────────────────────────────────────

def _load_cfg() -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_token(cfg: dict) -> str:
    if "feishu" not in cfg:
        raise RuntimeError(
            "config.yaml 缺 'feishu' 配置块。请参考 config.example.yaml 复制并填入凭据。"
        )
    fc = cfg["feishu"]
    resp = requests.post(
        f"{BASE}/auth/v3/tenant_access_token/internal",
        json={"app_id": fc["app_id"], "app_secret": fc["app_secret"]},
        timeout=10,
    ).json()
    return resp.get("tenant_access_token", "")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ── P2P 会话发现 ──────────────────────────────────────────────

def list_p2p_chats(cfg: dict | None = None) -> list[dict]:
    """
    列举 bot 参与的所有 P2P 私聊，返回 [{"chat_id", "name", "msg_count"}, ...]。
    调用 GET /im/v1/chats?chat_type=p2p，P2P 会话的 name 字段通常即对方姓名。
    """
    cfg   = cfg or _load_cfg()
    token = _get_token(cfg)
    chats = []
    page_token = None

    while True:
        params = {"chat_type": "p2p", "page_size": 100}
        if page_token:
            params["page_token"] = page_token
        try:
            resp = requests.get(
                f"{BASE}/im/v1/chats",
                headers=_headers(token),
                params=params,
                timeout=10,
            ).json()
        except Exception as e:
            log.warning(f"获取会话列表失败: {e}")
            break

        if resp.get("code", -1) != 0:
            log.warning(f"API 错误: {resp.get('msg')}")
            break

        for item in resp.get("data", {}).get("items") or []:
            chats.append({
                "chat_id":   item.get("chat_id", ""),
                "name":      item.get("name", ""),
            })

        if not resp.get("data", {}).get("has_more"):
            break
        page_token = resp.get("data", {}).get("page_token")

    return chats


def print_p2p_chats():
    """打印所有 P2P 会话，方便用户找到目标同事的 chat_id。"""
    chats = list_p2p_chats()
    if not chats:
        print("未找到任何 P2P 私聊（bot 需要先与对方有过消息往来）")
        return
    print(f"找到 {len(chats)} 个 P2P 私聊：\n")
    print(f"  {'姓名':<20}  chat_id")
    print(f"  {'-'*20}  {'-'*40}")
    for c in chats:
        print(f"  {c['name']:<20}  {c['chat_id']}")
    print(f"\n将目标同事的 chat_id 填入 config.yaml → colleague.p2p_chat_id")


# ── 用户查找 ──────────────────────────────────────────────────

def find_user_by_name(name_keyword: str, cfg: dict | None = None) -> list[dict]:
    """
    在所有已配置群聊的成员列表中按姓名关键词查找用户。
    调用 GET /im/v1/chats/{chat_id}/members，只需 im:chat:readonly 权限（bot 已有）。
    返回 [{"open_id", "name", "chat_name"}, ...]。
    """
    cfg      = cfg or _load_cfg()
    token    = _get_token(cfg)
    chat_ids = cfg.get("feishu", {}).get("chat_ids", [])
    if not chat_ids:
        log.warning("feishu.chat_ids 为空，无法从群聊成员中搜索用户")
        return []

    found: dict[str, dict] = {}   # open_id → result，去重

    for chat_id in chat_ids:
        # 先获取群名称
        try:
            chat_info = requests.get(
                f"{BASE}/im/v1/chats/{chat_id}",
                headers=_headers(token),
                timeout=10,
            ).json()
            chat_name = chat_info.get("data", {}).get("name", chat_id)
        except Exception:
            chat_name = chat_id

        # 翻页获取群成员
        page_token = None
        while True:
            params = {"member_id_type": "open_id", "page_size": 100}
            if page_token:
                params["page_token"] = page_token
            try:
                resp = requests.get(
                    f"{BASE}/im/v1/chats/{chat_id}/members",
                    headers=_headers(token),
                    params=params,
                    timeout=10,
                ).json()
            except Exception as e:
                log.warning(f"获取群成员失败 {chat_id}: {e}")
                break

            if resp.get("code", -1) != 0:
                log.warning(f"群成员 API 错误 {chat_id}: {resp.get('msg')}")
                break

            for member in resp.get("data", {}).get("items") or []:
                name    = member.get("name", "")
                open_id = member.get("member_id", "")
                if name_keyword.lower() in name.lower() and open_id:
                    if open_id not in found:
                        found[open_id] = {"open_id": open_id, "name": name, "chat_name": chat_name}

            if not resp.get("data", {}).get("has_more"):
                break
            page_token = resp.get("data", {}).get("page_token")

    return list(found.values())


def print_find_user(name_keyword: str):
    """打印按姓名搜索到的用户列表，方便用户确认并复制 open_id。"""
    users = find_user_by_name(name_keyword)
    if not users:
        print(f"未在群聊成员中找到「{name_keyword}」")
        return
    print(f"在群聊成员中找到 {len(users)} 个匹配「{name_keyword}」的用户：\n")
    print(f"  {'姓名':<16}  {'所在群':<20}  open_id")
    print(f"  {'-'*16}  {'-'*20}  {'-'*40}")
    for u in users:
        print(f"  {u['name']:<16}  {u['chat_name']:<20}  {u['open_id']}")
    print(f"\n将目标同事的 open_id 填入 config.yaml → colleague.open_id")


# ── 消息采集 ──────────────────────────────────────────────────

def _pull_messages_from_chat(
    token: str,
    chat_id: str,
    since_s: str,
    target_open_id: str = "",
    limit: int = 300,
) -> list[str]:
    """
    从单个会话中拉取消息，可选按 open_id 过滤发言人。
    返回纯文本列表。
    """
    collected: list[str] = []
    page = 0
    cur_since = since_s

    while len(collected) < limit and page < 100:
        page += 1
        try:
            resp = requests.get(
                f"{BASE}/im/v1/messages",
                headers=_headers(token),
                params={
                    "container_id_type": "chat",
                    "container_id": chat_id,
                    "start_time": cur_since,
                    "page_size": 50,
                    "sort_type": "ByCreateTimeAsc",
                },
                timeout=10,
            ).json()
        except Exception as e:
            log.warning(f"消息拉取失败 {chat_id} 第{page}页: {e}")
            break

        if resp.get("code", -1) != 0:
            log.warning(f"API 错误 {chat_id}: code={resp.get('code')} msg={resp.get('msg')}")
            break

        data  = resp.get("data") or {}
        items = data.get("items") or []
        if not items:
            break

        for msg in items:
            sender      = msg.get("sender", {})
            sender_type = sender.get("sender_type", "")
            sender_id   = sender.get("id", "")

            # 只要真人消息
            if sender_type != "user":
                continue
            # 若指定了 open_id，只取该人发言
            if target_open_id and sender_id != target_open_id:
                continue
            if msg.get("msg_type") != "text":
                continue

            try:
                content = json.loads(msg.get("body", {}).get("content", "{}"))
                text = content.get("text", "").strip()
            except Exception:
                continue

            text = re.sub(r"@[^\s]+", "", text).strip()
            if text and len(text) >= 2:
                collected.append(text)

        last_ct = items[-1].get("create_time", "")
        if last_ct:
            cur_since = str(int(last_ct) // 1000 + 1)

        if not data.get("has_more"):
            break

        time.sleep(0.15)

    return collected


def fetch_colleague_messages(
    cfg: dict | None = None,
    days: int = 60,
    limit: int = 300,
) -> list[str]:
    """
    采集目标同事的文本消息，支持三种来源（优先级从高到低）：

    1. colleague.p2p_chat_id 非空 → 从 P2P 私聊采集（所有真人消息均为目标同事）
    2. colleague.open_id 非空    → 从 feishu.chat_ids 群聊中按 open_id 过滤
    3. 以上均未配置              → 报错提示

    返回纯文本列表（已去除 @mention），最多 limit 条。
    """
    cfg   = cfg or _load_cfg()
    token = _get_token(cfg)
    cc    = cfg.get("colleague", {})

    p2p_chat_id = cc.get("p2p_chat_id", "").strip()
    open_id     = cc.get("open_id", "").strip()
    since_s     = str(int((datetime.now() - timedelta(days=days)).timestamp()))

    if p2p_chat_id:
        # 方式 1：P2P 私聊，所有真人消息都是目标同事的
        log.info(f"采集来源：P2P 私聊 {p2p_chat_id}")
        msgs = _pull_messages_from_chat(token, p2p_chat_id, since_s, limit=limit)

    elif open_id:
        # 方式 2：群聊 + open_id 过滤
        chat_ids = cfg.get("feishu", {}).get("chat_ids", [])
        if not chat_ids:
            raise ValueError("colleague.open_id 已配置，但 feishu.chat_ids 为空，无法从群聊采集")
        log.info(f"采集来源：{len(chat_ids)} 个群聊，过滤 open_id={open_id}")
        msgs = []
        for cid in chat_ids:
            part = _pull_messages_from_chat(token, cid, since_s, target_open_id=open_id, limit=limit)
            msgs.extend(part)
            if len(msgs) >= limit:
                break
        msgs = msgs[:limit]

    else:
        raise ValueError(
            "请在 config.yaml 中配置 colleague.p2p_chat_id 或 colleague.open_id。\n"
            "  - 有 P2P 私聊记录：运行 python cli.py style --list-chats\n"
            "  - 无私聊记录：运行 python cli.py style --find-user <姓名> 获取 open_id"
        )

    log.info(f"共采集 {len(msgs)} 条消息")
    return msgs


# ── 风格分析 ──────────────────────────────────────────────────

def analyse_style(messages: list[str], cfg: dict) -> str:
    """
    将消息列表交给 LLM 分析，返回风格描述文本（150 字以内）。
    优先 Qwen，失败回退 Claude Haiku。
    """
    sample  = messages[:150]
    joined  = "\n".join(f"· {m}" for m in sample)
    prompt  = (
        f"以下是某人在工作群里的真实聊天记录（共 {len(sample)} 条）：\n\n"
        f"{joined}\n\n"
        "请分析其说话风格，输出一段人设补充描述（150字以内），要求：\n"
        "- 用第二人称「你」描述（如「你习惯…」「你遇到…会…」）\n"
        "- 列出 3-5 条最鲜明的语言习惯或常用词\n"
        "- 说明回复长短偏好\n"
        "- 只输出风格描述本身，不要分析过程"
    )

    # 优先 Qwen
    qw = cfg.get("qwen", {})
    if qw.get("api_key"):
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=qw["api_key"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            resp = client.chat.completions.create(
                model=qw.get("model", "qwen3-8b"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"Qwen 分析失败，回退 Claude: {e}")

    # 回退 Claude
    ant = cfg.get("anthropic", {})
    if ant.get("api_key"):
        import anthropic
        client = anthropic.Anthropic(api_key=ant["api_key"])
        resp = client.messages.create(
            model=ant.get("model", "claude-haiku-4-5-20251001"),
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    raise RuntimeError("未配置任何可用的 LLM，无法分析风格（请配置 qwen.api_key 或 anthropic.api_key）")


# ── 读写 persona ──────────────────────────────────────────────

def update_persona(style_desc: str):
    """将风格描述写入 learning/persona.txt（附带时间戳注释）。"""
    PERSONA_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    PERSONA_FILE.write_text(
        f"# 自动生成风格描述（{ts}，由 python cli.py style 更新）\n{style_desc}\n",
        encoding="utf-8",
    )
    log.info(f"风格描述已写入 {PERSONA_FILE}")


def load_persona() -> str:
    """
    加载 learning/persona.txt，跳过 # 注释行，返回风格描述。
    文件不存在或为空时返回 ""。
    """
    if not PERSONA_FILE.exists():
        return ""
    lines = [
        line for line in PERSONA_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return "\n".join(lines).strip()


# ── 主流程 ────────────────────────────────────────────────────

def run(dry_run: bool = False, days: int | None = None) -> str:
    """
    完整流程：加载配置 → 采集消息 → 分析风格 → 写入文件。

    dry_run=True 时只打印分析结果，不写入 persona.txt。
    返回摘要字符串（供 CLI 打印或飞书推送）。
    """
    cfg   = _load_cfg()
    cc    = cfg.get("colleague", {})
    days  = days or cc.get("fetch_days", 60)
    min_n = cc.get("min_messages", 30)

    source = "P2P 私聊" if cc.get("p2p_chat_id") else f"群聊（open_id={cc.get('open_id', '?')}）"
    print(f"[style] 采集最近 {days} 天消息（来源：{source}）...")
    messages = fetch_colleague_messages(cfg, days=days)

    if len(messages) < min_n:
        return (
            f"采集到 {len(messages)} 条消息，样本不足（需 ≥{min_n} 条）。\n"
            "可调整 config.yaml colleague.fetch_days 延长采集范围，或降低 min_messages 阈值。"
        )

    print(f"[style] 共 {len(messages)} 条消息，正在分析风格...")
    style = analyse_style(messages, cfg)

    summary = f"采集：{len(messages)} 条消息\n\n【提炼结果】\n{style}"

    if dry_run:
        summary = "[dry-run] " + summary
        print(summary)
    else:
        update_persona(style)
        print(summary)
        print(f"\n已写入 {PERSONA_FILE}，下次闲聊时自动生效。")

    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _dry = "--dry-run" in sys.argv
    _days_arg = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
    print(run(dry_run=_dry, days=_days_arg))
