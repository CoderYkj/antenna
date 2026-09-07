# 微信通道集成计划

> 状态：Phase 1 + Phase 2 待实现；Phase 3 待公网 IP 后执行

## 需求概述

在现有飞书通道基础上，新增企业微信通道：
- **单向推送**：定时任务报告（扫描/预测/复盘/训练/学习/告警）同时推送到企业微信
- **双向对话**：用户在企业微信发指令（推荐/战法/历史等），与飞书 bot 等价（Phase 3，需公网 HTTPS）

飞书通道保持不变，两套通道并存，按 `config.yaml` 开关控制。

---

## 架构

```
定时任务/学习报告
      ↓
notify/dispatcher.py
      ↙              ↘
notify/feishu      notify/wecom
（现状不变）        （新增）

企业微信用户发指令（Phase 3）
      ↓
server/wecom_bot.py  ← 需公网 HTTPS callback
      ↓
server/commands.py（复用，不改）
      ↓
notify/wecom.py → 回推结果给用户
```

---

## 新增文件

| 文件 | 职责 | 阶段 |
|------|------|------|
| `notify/render.py` | Feishu 卡片 → WeCom markdown 翻译层 | Phase 1 |
| `notify/wecom.py` | 企业微信主动消息 API（access_token 缓存） | Phase 2 |
| `notify/dispatcher.py` | 飞书 + 企业微信多通道扇出 | Phase 2 |
| `server/wecom_crypto.py` | 回调签名校验 + AES-256-CBC 解密 | Phase 3 |
| `server/wecom_bot.py` | 回调服务（GET 验证 + POST 收消息） | Phase 3 |
| `docs/wecom_setup.md` | 部署指南（申请应用、ngrok/Nginx/Caddy） | Phase 3 |
| `tests/test_render.py` | 渲染层单测 | Phase 1 |
| `tests/test_wecom.py` | WeCom 客户端单测 | Phase 2 |
| `tests/test_dispatcher.py` | 多通道扇出单测 | Phase 2 |
| `tests/test_wecom_crypto.py` | 加解密单测（官方测试向量） | Phase 3 |
| `tests/test_wecom_bot.py` | 回调端到端单测 | Phase 3 |

---

## 修改文件

| 文件 | 改动 |
|------|------|
| `config.example.yaml` | 新增 `channels:` + `wecom:` 配置块 |
| `notify/__init__.py` | re-export `dispatcher.broadcast` |
| `scripts/task_scan.py` | `from notify.feishu import send_xxx` → `from notify import broadcast` |
| `scripts/task_predict.py` | 同上 |
| `scripts/task_daily_review.py` | 同上 |
| `scripts/task_train.py` | 同上 |
| `learning/alerts.py` | 同上 |
| `cli.py`（line 323） | 同上（learn 完成推送） |
| `ecosystem.config.cjs` | Phase 3：新增 `wecom_bot` 进程项 |

**注意：`notify/feishu.py` 和 `server/feishu_push.py` 不改动。**

---

## 配置 Schema

```yaml
# config.example.yaml 新增部分

# ── 多通道总开关 ─────────────────────────────────────
channels:
  feishu:  true    # 飞书（默认开）
  wecom:   false   # 企业微信（配置完凭据后开）

# ── 企业微信自建应用 ─────────────────────────────────
wecom:
  corp_id:          ""    # 我的企业 → 企业信息 → 企业ID
  agent_id:         0     # 自建应用 → AgentId（数字）
  secret:           ""    # 自建应用 → Secret
  # Phase 3 专属（需公网 HTTPS）
  callback_token:   ""    # 接收消息 → Token
  encoding_aes_key: ""    # 接收消息 → EncodingAESKey（43位）
  callback_port:    5000
  callback_path:    /wecom/callback
  allowed_users:    []    # 指令白名单，空=不限制
```

---

## Phase 1：渲染层（无外部依赖）

**目标**：建立 Feishu 卡片 → WeCom markdown 的翻译层，作为后续所有通道的基础。

### 步骤

1. **扩展 config schema**（`config.example.yaml`）
   - 新增 `channels:` 和 `wecom:` 块（Phase 3 字段先留空注释）
   - Risk: Low

2. **创建渲染层** (`notify/render.py`)
   - `to_wecom_markdown(reply: dict | str) -> str`
     - 输入兼容 `str` 与 Feishu interactive `dict`
     - 遍历 `elements`：`markdown` → 原文，`hr` → `---`，`column_set` → 多列拼 `|` 分隔
     - 超 4096 字节自动截断加省略号
     - `img` 类型降级为占位文字（与「不发图片」决策一致）
   - Risk: Medium（Feishu 卡片结构多样，需覆盖常见 element 类型）

3. **单测渲染层** (`tests/test_render.py`)
   - 用项目中真实卡片（如 `_scan_ack_card`）做样例
   - 断言关键字段保留、超长截断、img 降级
   - Risk: Low

---

## Phase 2：WeCom 单向推送（无公网要求）

**目标**：定时任务报告同时推送到企业微信，飞书行为不变。

### 步骤

4. **创建 WeCom 客户端** (`notify/wecom.py`)
   - `_get_access_token()` — 带锁缓存，7200 秒有效，自动刷新
   - `send_markdown(content, touser="@all")` — 调 `message/send` API
   - `send_text(content, touser="@all")`
   - 返回 `bool`，失败只记日志不抛异常
   - Risk: Medium（access_token 5分钟内频繁刷新会被腾讯限流，复用双 lock 缓存模式）

5. **单测 WeCom 客户端** (`tests/test_wecom.py`)
   - mock token 缓存命中 / 过期
   - mock 错误码 `40014`（token 失效自动重取）
   - Risk: Low

6. **创建 dispatcher** (`notify/dispatcher.py`)
   - `broadcast(reply, channels=None)` — 从 config 读启用通道
   - 每通道独立线程（`daemon=True`），任一失败不影响其他
   - 返回 `dict[str, bool]`
   - Risk: Medium（失败隔离要严格）

7. **单测 dispatcher** (`tests/test_dispatcher.py`)
   - 通道开关、单通道异常时其他仍执行
   - Risk: Low

8. **接入定时任务**（`scripts/task_*.py`、`learning/alerts.py`、`cli.py`）
   - 改 import：`from notify.feishu import send_xxx` → `from notify import broadcast`
   - 旧 `notify.feishu.*` API 保留，加 `DeprecationWarning`，下一版本清理
   - Risk: Medium（逐个验证 payload 类型）

---

## Phase 3：WeCom 双向对话（需公网 HTTPS）

> **前置条件**：有公网 IP + 域名，配置 Caddy/Nginx + HTTPS 证书

### 步骤

9. **创建加解密模块** (`server/wecom_crypto.py`)
   - 实现腾讯官方 `WXBizMsgCrypt`：URL 验证 + 消息 AES-256-CBC 解密
   - 参考 `wechatpy` 开源实现，不从零手写
   - Risk: High（一处错则全部 callback 失败，必须用官方测试向量验证）

10. **单测 wecom_crypto** (`tests/test_wecom_crypto.py`)
    - 用腾讯文档官方测试向量做断言（`token=QDG6eK` 系列样例）
    - Risk: Low

11. **创建回调服务** (`server/wecom_bot.py`)
    - GET：URL 验证（解密 echostr 原样返回）
    - POST：解密 XML → 提取 `<Content>` → `handle_command()` → **立即返回空 200**
    - 异步线程调 `notify.wecom.send_markdown(reply, touser=from_user)` 回推
    - `MsgId` LRU 去重（防腾讯 5 秒超时重发 3 次）
    - Risk: High（5 秒响应窗口 + 重发去重 + 签名校验三关须同时正确）

12. **单测 wecom_bot** (`tests/test_wecom_bot.py`)
    - GET 验证 / POST 解密 / 去重 / ack 行为
    - Risk: Medium

13. **PM2 接入** (`ecosystem.config.cjs`)
    - 新增 `wecom_bot` 进程，与 `feishu_poll` 同等地位
    - Risk: Low

14. **部署文档** (`docs/wecom_setup.md`)
    - 申请企业微信组织 + 自建应用步骤
    - 凭据填入 `config.yaml` 说明
    - Caddy/Nginx HTTPS 配置样例
    - ngrok 临时测试方法
    - 排错：token 错 / aeskey 长度 / IP 白名单
    - Risk: Low

---

## 风险汇总

| 风险 | 严重度 | 缓解措施 |
|------|--------|----------|
| WeCom 回调加解密实现错误 | **High** | 官方测试向量单测；参考 wechatpy 实现 |
| 腾讯 5 秒超时重发导致重复执行 | **High** | MsgId LRU 去重，立即返回 200 |
| access_token 频繁刷新被限流 | Medium | 双 lock 缓存，复用 feishu_push 模式 |
| Feishu 卡片渲染降级丢信息 | Medium | column_set 拼 `\|`，测试固化期望输出 |
| dispatcher 某通道阻塞拖慢主任务 | Medium | 每通道 daemon 线程，主线程立即返回 |

---

## 成功标准

- [ ] `channels.wecom: false` 时行为与现在完全一致，现有测试全部通过
- [ ] `channels.wecom: true` 时定时报告同时推送飞书和企业微信
- [ ] 渲染层将飞书卡片正确转为 WeCom markdown（关键字段不丢失）
- [ ] dispatcher 中任一通道失败不影响另一通道
- [ ] 新增模块单测覆盖率 ≥ 80%
- [ ] Phase 3：企业微信发"推荐 3"，5 秒内收到 ACK，60 秒内收到完整结果

---

## 不在范围内

- PushPlus（已排除）
- 企业微信群机器人 Webhook（仅做自建应用）
- 微信公众号
- 图片推送（与项目「不发图片」决策一致）
- itchat / 非官方个人微信客户端
