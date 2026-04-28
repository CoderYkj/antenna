# Antenna Bot — 代码修改后自测清单

> 每次修改代码后，在提交前必须逐项核查。重点关注股票代码↔名称转换链路。

---

## 一、股票代码↔名称转换（最高优先级）

| # | 检查点 | 检查方法 |
|---|--------|---------|
| 1 | `_resolve_code()` 返回的是 6 位数字字符串，而不是名称或 None | 在 `commands.py` 中追踪每条 `if cmd in XXX_WORDS` 分支 |
| 2 | 调用 `cmd_*()` 时传入的是 `code`（纯数字），而不是 `name` | 检查每个 `cmd_*(result)` 调用，`result` 必须是 str 且全为数字 |
| 3 | `cmd_news(code)` 内部通过 `_load_name_map()[code]` 获取名称，调用 `fetch_cls_news_for(name=..., code=code)` | 检查 `name_map` 取值有无 `.get(code, code)` 兜底 |
| 4 | `cmd_quote(code)` 从 `fetch_realtime_price` 返回值中取 `name` 字段，不能硬编码 | 检查 `price_info.get("name", code)` 是否有兜底 |
| 5 | `cmd_predict(code)` 展示卡片标题含 `name（code）`，名称来自实时行情 | 检查 `name` 变量赋值行 |
| 6 | `cmd_scan_bot()` 扫描结果中 `r.get("name", "")` 是否可能为空串 | 检查空名称时新闻过滤是否会误过滤所有内容 |
| 7 | `list_codes()` 在 `watchlist.py` 中展示时，名称从 `fetch_realtime_prices` 获取，不能缺失 | 检查实时行情失败时是否有 code 兜底显示 |

---

## 二、函数返回类型一致性

每个 `cmd_*` 函数的返回值必须符合以下规范：

| 函数 | 预期返回类型 | 注意 |
|------|------------|------|
| `cmd_quote` | `dict`（飞书 interactive 卡片） | 有分时数据时不重复输出 summary |
| `cmd_predict` | `dict`（飞书 interactive 卡片） | — |
| `cmd_news` | `dict`（飞书 interactive 卡片） | — |
| `cmd_scan_bot` | `dict`（飞书 interactive 卡片） | — |
| `cmd_review` | `dict`（飞书 interactive 卡片） | — |
| `cmd_strategy` | `dict`（飞书 interactive 卡片） | — |
| `cmd_financial_report` | `dict`（飞书 interactive 卡片） | — |
| `cmd_backtest_bot` | `dict`（飞书 interactive 卡片） | — |
| `cmd_chat` | `dict`（`msg_type: "text"`，纯文本，非卡片） | 闲聊专用，不走 markdown |
| `list_codes` | `dict`（飞书 interactive 卡片）或 `str`（空列表提示） | — |
| `add_code` / `remove_code` | `tuple[bool, str]` | — |
| `handle_command` | `str` 或 `dict` | 均可，由 `_reply()` 统一处理 |

**检查方法**：在修改任何 `cmd_*` 函数时，追踪所有 `return` 语句，确认类型未变化。

---

## 三、飞书消息格式

| # | 检查点 |
|---|--------|
| 1 | `interactive` 卡片中 `elements` 列表不能为空 |
| 2 | `markdown` 元素中的内容若含 `*` `_` `[` `]` 需转义（`_escape_md`） |
| 3 | `cmd_chat` 返回 `{"msg_type": "text", "content": json.dumps({"text": ...})}` 格式，不走卡片 |
| 4 | `push()` 接受 `str` 或 `dict` 均可，不需要额外转换 |
| 5 | 新闻类日期格式统一为 `[YYYY-MM-DD]`，不含反引号，不含 Markdown 格式符 |

---

## 四、commands.py 路由完整性

每次修改 `commands.py` 或新增指令词后：

| # | 检查点 |
|---|--------|
| 1 | 新增的 `XXX_WORDS` 集合是否在 `handle_command` 中有对应 `if cmd in XXX_WORDS` 分支 |
| 2 | 需要股票代码的指令是否都先调用 `_resolve_code()` 并处理 `None` 和 `dict`（多匹配）两种情况 |
| 3 | `_resolve_code` 返回 `dict` 时（多匹配选择卡片）是否直接 `return result`，不继续执行 |
| 4 | 末尾兜底 `cmd_chat(text, chat_id=chat_id)` 是否保留，`chat_id` 是否正确传递 |
| 5 | `PUSH_WORDS` 分支中 `handle_command(sub_text)` 递归调用是否正常（无死循环） |

---

## 五、feishu_poll.py 消息处理链

| # | 检查点 |
|---|--------|
| 1 | `_process(msg, chat_type, chat_id=chat_id)` 调用时 `chat_id` 是否传入 |
| 2 | `handle_command(text, chat_id=chat_id)` 调用时 `chat_id` 是否传入 |
| 3 | 群聊过滤（`bot_mentioned` 检查）不影响私聊（`p2p`）消息 |
| 4 | `_seen` 集合超出 2000 条后有截断，不会无限增长 |
| 5 | `_reply()` 对 `str` 和 `dict` 两种类型均有处理 |

---

## 六、新闻与相关性过滤

| # | 检查点 |
|---|--------|
| 1 | `_is_relevant()` 使用的 `name_keywords` 包含：完整名称、简称（去掉"股份/集团/网络/科技"等后缀）、6位代码 |
| 2 | 过滤后若两类（正面/负面）新闻均为空，卡片应显示"暂无相关消息" |
| 3 | 新闻按 `date_str` 降序排列（`reverse=True`），最多显示 10 条 |
| 4 | `<em>` 标签已清除（`replace("<em>","").replace("</em>","")` ） |

---

## 七、快速自测脚本（CLI 验证）

修改后可在项目根目录运行以下命令快速验证：

```bash
# 1. 按代码查询（数字输入）
python cli.py bot 行情 000001

# 2. 按名称查询（模糊匹配）
python cli.py bot 行情 平安银行

# 3. 预测（验证名称展示）
python cli.py bot 预测 600519

# 4. 新闻（验证相关性过滤 + 日期格式）
python cli.py bot 新闻 002174

# 5. 推荐（验证扫描结果含名称）
python cli.py bot 推荐 3

# 6. 复盘（验证推荐股/自选股分开统计）
python cli.py bot 复盘

# 7. 闲聊（验证纯文本、无 markdown）
python cli.py bot 今天大盘怎么样

# 8. 帮助（验证卡片完整性）
python cli.py bot 帮助
```

---

## 八、修改影响范围速查

| 修改的文件 | 需要重点检查的地方 |
|-----------|-----------------|
| `predict_cmd.py` | 函数返回类型、名称变量赋值、新闻过滤逻辑 |
| `commands.py` | 路由完整性、`_resolve_code` 调用、`chat_id` 传递 |
| `fetcher.py` | `_load_name_map` 格式、`fetch_realtime_prices` 返回字段 |
| `analyser.py` | `fetch_cls_news_for` 参数顺序、返回值解包 |
| `feishu_poll.py` | `chat_id` 传递链、`_process` 参数 |
| `watchlist.py` | `add_code/remove_code` 返回 `tuple[bool, str]` |
| `optimizer.py` | 推荐股/自选股精准率分开统计（`is_watchlist` 分支） |
| `config.yaml` | `anthropic.api_key` 不为空、`feishu.chat_ids` 格式正确 |
