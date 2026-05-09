# Antenna — A 股量化分析与飞书交易助手

基于机器学习的 A 股量化交易辅助系统。支持历史数据拉取、LightGBM 模型训练、个股预测、全市场扫描、四大选股战法、Walk-Forward 回测、策略自优化，并通过飞书机器人提供双向交互式操控与 AI 闲聊。

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置文件](#配置文件)
- [CLI 命令](#cli-命令)
- [定时任务](#定时任务)
- [飞书集成](#飞书集成)
- [飞书机器人指令](#飞书机器人指令)
- [选股战法](#选股战法)
- [策略自优化](#策略自优化)
- [AI 闲聊](#ai-闲聊)

---

## 项目结构

```
antenna/
├── cli.py                      # 命令行入口
├── config.yaml                 # 全局配置（数据源、飞书、AI 密钥）
├── requirements.txt
│
├── data/
│   ├── fetcher.py              # 行情数据拉取（akshare + 新浪实时）
│   ├── universe.py             # 股票池管理（watchlist / 全 A 股）
│   ├── name_map.json           # 代码→名称映射表
│   ├── sector_map.json         # 代码→行业映射表
│   └── cache/                  # 本地 Parquet 缓存（自动创建）
│
├── features/
│   ├── builder.py              # 特征工程流水线
│   ├── technical.py            # 技术指标（MA / MACD / RSI / KDJ / 布林带 / ATR 等）
│   ├── analyser.py             # 技术分析描述、双周期买卖价位、财联社资讯抓取
│   └── fundamental.py          # 财务报表解构与投资建议
│
├── models/
│   ├── trainer.py              # LightGBM 模型训练
│   ├── predictor.py            # 模型推理 + 全局信号分配
│   └── saved/                  # 训练好的模型文件（自动创建）
│
├── learning/
│   ├── tracker.py              # 预测快照记录 & 实际结果写入
│   ├── optimizer.py            # 买入精准率统计 + 策略门槛自动调整
│   ├── backtest_history.py     # Walk-Forward 回测结果存档
│   ├── strategy.json           # 当前策略状态（buy_top_pct 等，自动维护）
│   └── persona.txt             # AI 闲聊人设（风格采集后生成）
│
├── server/
│   ├── feishu_poll.py          # 飞书轮询机器人（每 5s 拉取消息，支持并发指令）
│   ├── commands.py             # 指令路由解析
│   ├── predict_cmd.py          # 所有飞书指令的执行逻辑
│   ├── watchlist.py            # 自选股增删查（线程安全）
│   ├── style_learner.py        # 从飞书 P2P 记录采集风格，提炼 AI 闲聊人设
│   └── feishu_push.py          # 主动推送消息到飞书会话
│
├── notify/
│   └── feishu.py               # Webhook 单向推送（扫描/复盘报告）
│
├── scripts/
│   ├── task_scan.py            # 定时扫描推荐任务
│   ├── task_predict.py         # 盘中预测任务
│   ├── task_daily_review.py    # 收盘后自动复盘任务
│   ├── task_noon_review.py     # 午间复盘任务
│   ├── task_train.py           # 定时训练任务
│   ├── run_poll.bat            # 飞书机器人启动
│   ├── run_scan.bat / run_train.bat / ...
│   └── setup_tasks.ps1         # Windows 任务计划一键注册
│
├── reports/
│   └── output/                 # HTML 报告输出目录
│
└── logs/                       # 日志目录（自动创建）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 填写配置

编辑 `config.yaml`，至少填入：

- `feishu.app_id` / `feishu.app_secret` / `feishu.chat_ids`（飞书双向交互）
- `anthropic.api_key` 或 `qwen.api_key`（AI 闲聊，二选一）

### 3. 拉取行情数据

```bash
# 更新自选股数据（默认 365 天）
python cli.py fetch

# 首次全量拉取（约 5500 只，需数小时）
python cli.py fetch --all --workers 16
```

### 4. 训练模型

```bash
python cli.py train
```

### 5. 预测单只股票

```bash
python cli.py predict 600519
```

### 6. 启动飞书机器人

```bash
scripts\run_poll.bat
```

---

## 配置文件

`config.yaml` 是系统唯一配置入口：

```yaml
data:
  source: akshare          # 数据源
  cache_dir: data/cache    # 本地缓存目录
  default_days: 365        # 拉取天数

universe:
  watchlist:               # 自选股（飞书机器人可动态增删）
    - "600519"
    - "000858"
  scan_pool: all           # 扫描范围：watchlist 或 all（全 A 股缓存）

model:
  target_days: 5           # 预测未来 N 日涨跌
  threshold: 0.02          # 涨跌幅判定阈值（2%）
  train_years: 3           # 训练数据年限
  saved_dir: models/saved

feishu:
  webhook_url: ""          # 自定义机器人 Webhook（单向推送，可选）
  scan_top: 20             # 扫描结果推送 Top N
  app_id: ""               # 自建应用 App ID（双向交互必填）
  app_secret: ""           # 自建应用 App Secret
  chat_ids:                # 机器人监控的飞书会话 ID（oc_ 开头为群聊）
    - "oc_xxx"

anthropic:
  api_key: ""              # Anthropic API Key（闲聊备用）
  model: claude-haiku-4-5-20251001

qwen:
  api_key: ""              # 阿里云 DashScope API Key（闲聊首选）
  model: qwen3-8b          # qwen3-8b / qwen3-14b / qwen3-235b-a22b

colleague:                 # AI 闲聊风格采集（可选）
  p2p_chat_id: ""          # 与目标同事的飞书 P2P 会话 ID
  fetch_days: 60           # 采集最近 N 天消息
  min_messages: 30         # 样本不足此数量则不更新
```

---

## CLI 命令

### `fetch` — 拉取行情数据

```bash
python cli.py fetch                        # 更新自选股
python cli.py fetch --code 600519          # 更新指定股票
python cli.py fetch --all --workers 16     # 全量拉取（首次建议）
python cli.py fetch --days 730             # 指定天数
```

### `train` — 训练预测模型

```bash
python cli.py train
```

从本地缓存读取所有股票数据，训练 LightGBM 分类模型，保存至 `models/saved/`。

### `predict` — 预测单只股票

```bash
python cli.py predict 600519
```

输出信号（买入 / 观望 / 回避）、涨跌概率、置信度，并生成 HTML 报告。

### `scan` — 全市场扫描

```bash
python cli.py scan              # Top 20（默认）
python cli.py scan --top 50    # 自定义数量
```

对所有已缓存股票批量预测，按涨概率 + 技术动量综合排序，输出候选股清单。

### `backtest` — Walk-Forward 历史回测

```bash
python cli.py backtest --month 2026-01    # 月度回测
python cli.py backtest --year 2026        # 年度回测
```

逐日滚动预测，统计买入精准率并更新策略门槛。

### `style` — 采集同事风格（AI 闲聊人设）

```bash
python cli.py style               # 采集 + 分析 + 更新 persona.txt
python cli.py style --dry-run     # 只分析，不写文件
python cli.py style --days 90     # 覆盖采集天数
```

从 `config.yaml` 中 `colleague.p2p_chat_id` 指定的飞书私聊拉取消息，用 LLM 提炼语言风格，写入 `learning/persona.txt`，供 AI 闲聊时参考。

---

## 定时任务

一键注册 Windows 任务计划：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_tasks.ps1
```

| 任务 | 执行时间 | 说明 |
|------|---------|------|
| `Antenna-Train` | 每天 08:50 | 训练模型，完成后推送飞书 |
| `Antenna-Scan` | 工作日 09:15 | 全市场扫描，Top N 推荐推送飞书 |
| `Antenna-Predict` | 工作日 09:30、10:00、11:00、14:00 | 自选股盘中预测推送 |
| `Antenna-NoonReview` | 工作日 12:00 | 午间阶段性复盘 |
| `Antenna-DailyReview` | 工作日 15:30 | 收盘后自动复盘，触发策略门槛更新 |

> 通过交易所日历自动识别节假日，节假日直接跳过不执行。

日志位于 `logs/` 目录。

---

## 飞书集成

### 主动推送（单向 Webhook）

在 `config.yaml` 填入群机器人 Webhook：

```yaml
feishu:
  webhook_url: "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
```

获取方式：飞书群 → 群设置 → 机器人 → 添加自定义机器人 → 复制 Webhook。

### 双向交互（自建应用机器人）

1. 打开 [open.feishu.cn](https://open.feishu.cn) → 创建企业自建应用
2. 添加能力：**机器人**
3. 开启权限：`im:message:send_as_bot`、`im:message:readonly`、`im:chat:readonly`
4. 发布应用，将 `app_id`、`app_secret` 填入 `config.yaml`
5. 将机器人加入飞书群，复制群 ID（群设置 → 群信息，以 `oc_` 开头）填入 `chat_ids`

启动机器人（无需公网地址，每 5 秒主动轮询）：

```bash
scripts\run_poll.bat
```

**并发支持**：最多同时处理 3 条重型指令（预测、扫描、回测等），超出自动回复等待提示；闲聊不占并发槽位。

---

## 飞书机器人指令

在飞书群中 **@机器人** 发送以下指令（私聊直接发送，无需 @）。支持股票名称模糊搜索，如 `预测 茅台` 会自动匹配 `600519`；多个结果时返回候选列表。

### 自选股管理

| 指令 | 示例 | 说明 |
|------|------|------|
| `添加 <代码或名称>` | `添加 600519` / `添加 贵州茅台` | 加入自选股 |
| `删除 <代码或名称>` | `删除 600519` | 移出自选股 |
| `列表` | `列表` | 查看当前自选股实时行情 |

### 行情与分析

| 指令 | 示例 | 说明 |
|------|------|------|
| `行情 <代码或名称>` | `行情 002174` | 实时价格 + 日内温度计 + 分时走势 |
| `预测 <代码或名称>` | `预测 300785` | AI 信号 + 技术分析 + 双周期买卖价位 |
| `新闻 <代码或名称>` | `新闻 紫金矿业` | 近期正面 / 负面资讯（按情绪分类）|
| `财报 <代码或名称>` | `财报 600519` | 季报 / 年报深度解构与投资方向 |

### 扫描与回测

| 指令 | 示例 | 说明 |
|------|------|------|
| `推荐 (N)` | `推荐 3` | 全市场扫描推荐 N 只买入候选，附战法共振标签（默认 5）|
| `战法 <策略> (N)` | `战法 价值 5` | 按选股战法筛选 Top N，附双周期买卖价位 |
| `复盘 (日期)` | `复盘 2026-04-09` | 查看历史预测命中率及策略调整记录 |
| `历史 YYYY-MM` | `历史 2026-01` | 月度 Walk-Forward 回测（后台运行）|
| `历史 YYYY` | `历史 2026` | 年度回测（后台运行）|
| `策略` | `策略` | 查看当前选股门槛、精准率、评判标准 |

### 广播

| 指令 | 示例 | 说明 |
|------|------|------|
| `发送 <子指令>` | `发送 推荐` | 执行子指令并广播到所有配置会话 |

### 帮助

```
帮助
```

---

## 选股战法

`战法` 指令基于财务 + 技术双重筛选，分为四种策略，均支持**上涨概率**、**短线/长线目标涨幅**及**双周期买卖价位**输出。

```
战法 价值 5    # 价值投资法，筛选 Top 5
战法 成长 10   # 成长股投资法，筛选 Top 10
战法 龙头      # 行业龙头战法，默认 5 只
战法 逆向 3    # 逆向投资法，筛选 3 只
```

| 战法 | 英文 | 核心筛选条件 |
|------|------|------------|
| 📊 价值投资法 | value | PE 低·PB 低·ROE 高·低负债·现金流稳健 |
| 🚀 成长股投资法 | growth | 营收高增速·净利高增速·成长性突出 |
| 👑 行业龙头战法 | leader | ROE 优秀·毛利率高·技术强势·综合基本面强 |
| 🔄 逆向投资法 | contra | 技术超跌 >15%·基本面稳健·等待修复机会 |

### 每只股票输出内容

```
No.1 名称（代码）　方向　综合评分 9.5
  💹 当前股价 1520　涨跌 +1.23%　今日区间 1490～1525
  📈 上涨概率 73.5%　短线目标 +2.1%　长线目标 +5.8%
  📅 短线 1-5日　买入 99.3（MA5支撑）　止盈 103.5　止损 97.3　RR 2.1
  📅 长线 2-4周　买入 96.7（MA20支撑）　止盈 105.0　止损 90.2　RR 1.3
  ⏱ 建议持仓 中期 5-10日
  📊 ROE 29.4%　毛利率 91.5%　负债率 18.2%
```

### 双周期价位说明

| 周期 | 买入参考 | 止盈参考 | 止损参考 |
|------|---------|---------|---------|
| 短线（1-5 日） | MA5 支撑 / 布林下轨 | 布林上轨 / 预测高位 | ATR×1.5 或 2% |
| 长线（2-4 周） | MA20 支撑 | MA60 / 幅度×3 | MA60 下方 5% |

### 推荐指令中的战法共振

`推荐` 指令会在扫描结果上自动运行四大战法评分，命中 ≥ 2 个战法的股票显示共振标签：

```
1. 炜冈科技（001256）　买入　全市场第2/5193 [高]　★价值+龙头+逆向
　💹  现价 24.07  涨概率 62.4%  短线 +16.0%  长线 +12.7%
```

---

## 策略自优化

系统内置自适应选股门槛，每日收盘后自动运行。

**核心指标**：买入精准率 = 被标记「买入」且实际涨幅 ≥ 1% 的股票数 / 全部「买入」信号数

> 观望 / 回避信号不参与精准率计算，避免震荡市中「观望占多」导致的虚高精准率。

| 近 7 日精准率 | 调整动作 |
|-------------|---------|
| < 35% | 门槛收严（−1%）|
| 35%–55% | 维持当前门槛 |
| 55%–65% | 门槛微放（+1%）|
| > 65% | 门槛放宽（+2%）|
| 连续 7 天触底（8%） | 自动重置为 15%，重新探索 |

目标精准率：**55%**（A 股短线现实水平）。

策略状态持久化在 `learning/strategy.json`，原子写入，多线程安全。

---

## 学习系统（Learning）

系统分阶段升级"自主学习"能力，每日由飞书「学习」指令或编排器定时驱动。

### P0 — 地基层（✅ 已上线）

- `learning/market_state.py`：大盘状态打标（bull/bear/range），3 日防抖
- `learning/tracker.py`：pred/outcome JSONL 归档，含 scene 场景标签
- `learning/outcome_metrics.py`：`hit_tier`（miss/weak/good/great）与 5 日指标
- `learning/orchestrator.py`：瘦编排器，模块失败隔离，dry-run 支持
- `learning/optimizer.py`：精准率监控 + `buy_top_pct` 日级自动调参
- `learning/feedback_io.py`：原子写 + 7 份历史快照

### P1 — 模型层（✅ 已上线）

目标：从"纯分位切信号"升级为"概率校准 + 错样本加权重训 + 绝对阈值 gate"。

| 组件 | 职责 |
|------|------|
| `learning/model_learner.py` | isotonic 概率校准（按 market_state 分 3 桶）+ abs_threshold 月度自校 |
| `learning/model_learner.yaml` | 可配置参数：sample_weights / calibration / absolute_threshold |
| `models/predictor.assign_global_signals()` | 双门槛：`rank_pct < buy_top_pct AND prob_cal >= abs_threshold` |
| `scripts/backfill_hit_tier.py` | 历史 outcome 批量回填 hit_tier（幂等） |
| `scripts/backfill_market_state.py` | 沪深 300 历史回放，重塑 200+ 天状态序列 |
| `scripts/replay_learn.py --with-p1` | 198 天验收：baseline vs P1 精准率与 Brier 对比 |

**P1 投产流程**：

```bash
# 1. 一次性回填（幂等,可重复运行）
python scripts/backfill_hit_tier.py
python scripts/backfill_market_state.py

# 2. 首次触发(学习指令 or CLI)
python cli.py learn          # → 生成 learning/model_learner.json
python cli.py learn --check  # → 验证所有学习产物 JSON 合法

# 3. 加权重训(可选;默认日级不触发,由 Antenna-WeeklyTrain 周日 20:00 运行)
python cli.py train --weighted

# 4. 回放验收
python scripts/replay_learn.py --with-p1 --abs-threshold 0.30
```

**P1 实测验收**（2025-07-04 ~ 2026-05-08 共 205 天回放，`abs_threshold=0.30`）：

| 指标 | 基线 | P1 | Δ | 硬指标 |
|------|------|------|-----|--------|
| 买入精准率 | 21.95% | 30.48% | **+8.52pt** | ≥ +2pt ✅ |
| Mean Brier | 0.248 | 0.173 | ratio 0.697 | ≤ 0.95 ✅ |
| Calibrator 拟合率 | — | 98% (200/205 天) | — | — |

> 绝对精准率 30.48% 未达 spec 原设 35% 硬指标——因实际 baseline 22% 而非假设的 33%（当前模型在此数据期偏弱），isotonic 输出上限就是历史实际命中率 36.2%。P1 的核心价值是**把用户看到的概率校准到真实命中率**，避免"模型说 70% 实际只中 30%"的过度乐观。

### P2/P3/P4 — 路线图（⏳ 未启动）

- P2 战法层：`tactic_learner`（阈值/权重自适应）+ AI 深度理由
- P3 特征层：`alt_data` 资金面/情绪面 + `feature_learner`（指标贡献度）
- P4 价位层：`price_learner`（ATR/振幅系数网格搜索）
- 横向：连错股票黑名单，推荐路径过滤

---

## AI 闲聊

飞书群中无法识别的消息会转交给内置 AI 处理。

AI 角色设定为**方木木**：游戏公司后端开发，带着程序员的懒散口吻，说话简洁直接，每 5 分钟自动切换当前状态（改 bug 中 / 等构建 / 快下班了……）以保持回复多样性。

**模型调用链**（优先级从高到低）：

1. **阿里云 Qwen3**（`qwen.api_key` 非空时）
2. **Anthropic Claude Haiku**（Qwen 不可用时自动回退）
3. 内置兜底回复（两者均不可用时）

对话历史按 `chat_id` 独立保留，最多 20 轮上下文。

### 风格采集（可选）

通过 `python cli.py style` 采集指定同事的飞书聊天记录，用 LLM 提炼其真实语言习惯，写入 `learning/persona.txt`。之后 AI 闲聊时会在 system prompt 中追加这份风格描述，让角色更贴近本人。
