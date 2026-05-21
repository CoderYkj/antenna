# Antenna — A 股量化分析与飞书交易助手

基于 LightGBM + isotonic 概率校准的 A 股交易辅助系统，提供：

- 🤖 **AI 选股**：全市场扫描 → 双门槛信号（分位 + 校准概率）
- 📊 **四大战法**：价值 / 成长 / 龙头 / 逆向，财务 + 技术双重筛选
- 📈 **双周期价位**：每只候选股自动生成短线 1-5 日 + 长线 2-4 周买卖止损位
- 🧠 **自主学习**：每日按市场状态拟合 isotonic 校准器，把模型概率校准到真实命中率
- 💬 **飞书机器人**：双向交互、自然语言指令、AI 闲聊兜底
- ♻️ **Walk-Forward 回测** + **策略自优化**（精准率监控 → 选股门槛动态调整）

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置文件](#配置文件)
- [CLI 命令](#cli-命令)
- [飞书机器人](#飞书机器人)
- [选股战法](#选股战法)
- [学习系统](#学习系统-learning)
- [策略自优化](#策略自优化)
- [PM2 进程管理](#pm2-进程管理)
- [定时任务](#定时任务)
- [AI 闲聊](#ai-闲聊)

---

## 项目结构

```
antenna/
├── cli.py                         # 命令行入口（fetch/train/predict/scan/learn 等）
├── config.example.yaml            # 配置模板（含完整字段说明）
├── config.yaml                    # 本地配置（gitignored，含真实凭据）
├── requirements.txt
│
├── data/
│   ├── fetcher.py                 # 行情数据拉取（akshare + 新浪实时）
│   ├── universe.py                # 股票池管理
│   ├── name_map.json              # 代码 → 名称映射
│   ├── sector_map.json            # 代码 → 行业映射
│   └── cache/                     # Parquet 缓存（自动）
│
├── features/
│   ├── builder.py                 # 特征工程流水线
│   ├── technical.py               # 25 个技术指标
│   ├── analyser.py                # 技术分析描述、双周期价位、新闻抓取
│   └── fundamental.py             # 财务报表解构与评分
│
├── models/
│   ├── trainer.py                 # LightGBM 训练（含 train_weighted）
│   ├── predictor.py               # 推理 + 全市场信号双门槛分配
│   └── saved/                     # 模型 pkl + calibrator pkl（gitignored）
│
├── learning/                      # 学习系统(P0/P1/P2 + 横向黑名单)
│   ├── tracker.py                 # pred/outcome JSONL 归档（含 scene 标签）
│   ├── outcome_metrics.py         # hit_tier(miss/weak/good/great) + 5 日指标
│   ├── market_state.py            # 大盘状态打标(bull/bear/range)，3 日防抖
│   ├── orchestrator.py            # 瘦编排器，模块失败隔离
│   ├── feedback_io.py             # 原子写 + 7 份历史快照
│   ├── alerts.py                  # 飞书 + JSONL 三路告警
│   ├── optimizer.py               # buy_top_pct 日级自动调参
│   ├── scene_bucket.py            # 按 scene/state 分桶工具
│   ├── backtest_history.py        # Walk-Forward 结果存档
│   ├── model_learner.py           # ★ P1 isotonic 校准 + abs_threshold 自校 + 加权重训
│   ├── model_learner.yaml         # P1 可配置参数
│   ├── tactic_learner.py          # ★ P2 4战法×3状态=12桶阈值自适应 + 权重学习
│   ├── tactic_learner.yaml        # P2 defaults / bear override / bounds / step
│   ├── blacklist.py               # ★ 横向黑名单：连错股票自动拉黑 30 天
│   ├── blacklist.yaml             # streak_threshold / block_days / per_state_threshold
│   ├── strategy.json              # 当前选股策略状态（自动维护）
│   ├── model_learner.json         # P1 校准状态 + threshold 历史
│   ├── tactic_params.json         # P2 战法参数 + 权重（自动维护）
│   ├── blacklist.json             # 当前黑名单条目（自动维护）
│   ├── market_state.json          # 大盘状态历史（含 365 天容量）
│   └── persona.txt                # AI 闲聊人设
│
├── server/
│   ├── feishu_poll.py             # 飞书轮询（每 5s，并发指令支持）
│   ├── commands.py                # 指令词集合 + 路由
│   ├── predict_cmd.py             # 全部飞书指令执行体（含 cmd_learn）
│   ├── ai_reason.py               # ★ P2 LLM 深度推荐理由（Qwen3→Claude Haiku 三级降级）
│   ├── watchlist.py               # 自选股管理
│   ├── style_learner.py           # 风格采集 → persona.txt
│   ├── feishu_push.py             # 主动推送
│   └── pm2_monitor.py             # PM2 状态监控面板
│
├── notify/
│   └── feishu.py                  # Webhook 单向推送
│
├── scripts/
│   ├── backfill_hit_tier.py       # ★ P1 历史 hit_tier 批量回填（幂等）
│   ├── backfill_market_state.py   # ★ P1 沪深 300 历史回放
│   ├── replay_learn.py            # 学习系统回放 / P1 验收
│   ├── task_scan.py               # 定时扫描
│   ├── task_predict.py            # 盘中预测
│   ├── task_daily_review.py       # 收盘复盘
│   ├── task_noon_review.py        # 午间复盘
│   ├── task_train.py              # 定时训练
│   ├── run_train_weekly.bat       # ★ P1 周日 20:00 加权重训
│   ├── run_*.bat                  # 任务计划入口
│   └── setup_tasks.ps1            # Windows 任务计划一键注册
│
└── reports/output/                # HTML 报告输出（gitignored）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备配置

```bash
cp config.example.yaml config.yaml
```

至少填入以下字段（详见 [配置文件](#配置文件)）：

- `feishu.app_id` / `feishu.app_secret` / `feishu.chat_ids` — 飞书双向交互
- `qwen.api_key` 或 `anthropic.api_key` — AI 闲聊（二选一）

### 3. 拉取行情

```bash
# 仅自选股（约 30 秒）
python cli.py fetch

# 全 A 股首次全量（约 5500 只，需数小时）
python cli.py fetch --all --workers 16
```

### 4. 训练模型

```bash
python cli.py train                  # 标准训练
python cli.py train --weighted       # P1 加权重训（按历史 signal × hit_tier 加权）
```

### 5. 启动飞书机器人

```bash
# 前台启动（开发期）
python server/feishu_poll.py

# PM2 托管启动（生产推荐）
pm2 start ecosystem.config.cjs   # 同时启动 antenna-bot + pm2-monitor
pm2 save                          # 持久化进程列表，重启系统后自动恢复

# 或注册为 Windows 任务计划（无 PM2 时的替代方案）
powershell -File scripts/setup_tasks.ps1
```

---

## 配置文件

`config.yaml` 已脱离 git 跟踪（避免凭据外泄）。模板见 `config.example.yaml`，必填项：

| 段 | 字段 | 说明 |
|---|---|---|
| `data.default_days` | int | 拉取天数（默认 365） |
| `universe.watchlist` | list | 自选股代码列表 |
| `universe.scan_pool` | `watchlist` / `all` | 扫描范围 |
| `model.target_days` | int | 预测窗口（默认 5 日） |
| `model.threshold` | float | 涨幅判定阈值（默认 0.02） |
| `feishu.app_id` | str | 自建应用 App ID（双向必填） |
| `feishu.app_secret` | str | 自建应用 Secret |
| `feishu.chat_ids` | list | 监控的会话 ID（`oc_` 为群） |
| `feishu.webhook_url` | str | 单向推送 webhook（可选） |
| `qwen.api_key` | str | 阿里云 DashScope（AI 闲聊首选） |
| `anthropic.api_key` | str | Claude Haiku（AI 闲聊兜底） |

---

## CLI 命令

```bash
python cli.py <subcommand> [options]
```

| 子命令 | 功能 | 关键选项 |
|--------|------|---------|
| `fetch` | 拉取行情数据 | `--code` / `--all` / `--days` / `--workers` |
| `train` | 训练 LightGBM 模型 | `--weighted`（P1 加权）/ `--workers` |
| `predict <code>` | 单股预测 + HTML 报告 | — |
| `scan` | 全市场扫描 Top N | `--top` / `--workers` |
| `report <code>` | 生成 HTML 报告（不重训） | — |
| `backtest` | Walk-Forward 历史回测 | `--month YYYY-MM` / `--year YYYY` / `--top` |
| `learn` | 触发学习系统编排器 | `--date YYYY-MM-DD` / `--dry-run` / `--check` |
| `style` | 采集飞书同事聊天风格 | `--list-chats` / `--find-user` / `--dry-run` |
| `bot <cmd>` | 执行任意飞书指令并广播 | 例: `bot 推荐 5` |

---

## 飞书机器人

### 主动推送（单向 Webhook）

`notify/feishu.py` 通过 `webhook_url` 推送，定时任务的扫描结果与复盘报告由此发出。无需自建应用，配置 `feishu.webhook_url` 即可。

### 双向交互（自建应用）

`server/feishu_poll.py` 每 5 秒轮询消息，支持自然语言模糊匹配（`预测 茅台` → 自动解析到 `600519`）。需要自建应用 + 填入 `app_id` / `app_secret` / `chat_ids`。

**并发**：最多同时处理 3 条重型指令（预测 / 扫描 / 回测 / 学习），其余排队；闲聊不占槽位。

### 指令清单

群中需要 `@机器人`，私聊直接发送即可。

#### 自选股管理

| 指令 | 示例 | 说明 |
|------|------|------|
| `添加 <代码或名称>` | `添加 600519` | 加入自选 |
| `删除 <代码或名称>` | `删除 茅台` | 移出自选 |
| `列表` | `列表` | 自选股实时行情 |

#### 行情与分析

| 指令 | 示例 | 说明 |
|------|------|------|
| `行情 <code>` | `行情 002174` | 实时价格 + 日内温度计 + 分时 |
| `预测 <code>` | `预测 300785` | AI 信号 + 技术分析 + 双周期价位 |
| `趋势 <code>` | `趋势 600519` | 日 / 周 / 月线文字趋势图 |
| `新闻 <code>` | `新闻 紫金矿业` | 财联社近期资讯（按情绪分类） |
| `财报 <code>` | `财报 600519` | 季报 / 年报深度解构 |

#### 扫描与战法

| 指令 | 示例 | 说明 |
|------|------|------|
| `推荐 (N)` | `推荐 3` | 全市场扫描，含战法共振标签 |
| `战法 <策略> (N)` | `战法 价值 5` | 按战法筛选 Top N |
| `复盘 (日期)` | `复盘 2026-04-09` | 历史命中率 + 策略调整记录 |
| `历史 YYYY-MM` | `历史 2026-01` | 月度 Walk-Forward 回测 |
| `历史 YYYY` | `历史 2026` | 年度回测 |

#### 学习系统

| 指令 | 示例 | 说明 |
|------|------|------|
| `学习 (日期)` | `学习` / `学习 2026-04-29` | 跑 market_state + model_learner 编排器 |
| `学习 dry-run` | `学习 演练` | 演练，不写盘 |
| `策略` | `策略` | 当前选股门槛、精准率、abs_threshold |

#### 其他

| 指令 | 说明 |
|------|------|
| `发送 <子指令>` | 执行子指令并广播到所有会话 |
| `帮助` / `help` / `?` | 完整指令列表 |

无法识别的消息自动转交 [AI 闲聊](#ai-闲聊)。

---

## 选股战法

`战法` 指令基于**财务 + 技术**双重筛选，4 种策略均输出上涨概率、双周期目标涨幅与买卖止损位。

| 战法 | 英文 | 核心条件 |
|------|------|---------|
| 📊 价值投资法 | `value` / `价值` | PE 低 · PB 低 · ROE 高 · 低负债 |
| 🚀 成长股投资法 | `growth` / `成长` | 营收高增速 · 净利高增速 · ROE 优秀 |
| 👑 行业龙头战法 | `leader` / `龙头` | ROE > 15% · 毛利 > 30% · 技术强势 |
| 🔄 逆向投资法 | `contra` / `逆向` | 技术超跌 > 15% · 基本面稳健 |

**示例输出**：

```
No.1 名称（代码）　方向　综合评分 9.5
  💹 当前股价 152.0　涨跌 +1.23%　今日区间 149.0～152.5
  📈 上涨概率 73.5%　短线目标 +2.1%　长线目标 +5.8%
  📅 短线 1-5日　买入 99.3（MA5 支撑）　止盈 103.5　止损 97.3　RR 2.1
  📅 长线 2-4周　买入 96.7（MA20 支撑）　止盈 105.0　止损 90.2　RR 1.3
  ⏱ 建议持仓 中期 5-10 日
  📊 ROE 29.4%　毛利率 91.5%　负债率 18.2%
```

### 双周期价位

| 周期 | 买入 | 止盈 | 止损 |
|------|------|------|------|
| 短线（1-5 日） | MA5 / 布林下轨 | 布林上轨 / 预测高位 | ATR×1.5 或 2% |
| 长线（2-4 周） | MA20 | MA60 / 振幅×3 | MA60 下方 5% |

### 战法共振

`推荐` 指令会自动对扫描结果运行四大战法评分，命中 ≥ 2 个战法的股票显示共振标签：

```
1. 炜冈科技（001256）　买入　全市场第 2/5193 [高]　★价值+龙头+逆向
　💹 现价 24.07  涨概率 62.4%  短线 +16.0%  长线 +12.7%
```

---

## 学习系统 (Learning)

系统分阶段升级"自主学习"能力，由飞书 `学习` 指令、定时任务或 `cli.py learn` 触发。

### P0 — 地基层（✅ 已上线）

| 模块 | 职责 |
|------|------|
| `learning/market_state.py` | 大盘状态打标（bull/bear/range），3 日防抖 |
| `learning/tracker.py` | pred / outcome JSONL 归档，含 scene 场景标签 |
| `learning/outcome_metrics.py` | hit_tier（miss / weak / good / great）+ 5 日指标 |
| `learning/orchestrator.py` | 瘦编排器，模块失败隔离，dry-run 支持 |
| `learning/optimizer.py` | 精准率监控 + `buy_top_pct` 日级自动调参 |
| `learning/feedback_io.py` | 原子写 + 7 份历史快照 |
| `learning/alerts.py` | 飞书 + JSONL 三路告警 |

### P1 — 模型层（✅ 已上线）

**目标**：从"纯分位切信号"升级为"概率校准 + 错样本加权重训 + 绝对阈值 gate"。

| 组件 | 职责 |
|------|------|
| `learning/model_learner.py` | isotonic 校准（按 market_state 分 3 桶）+ abs_threshold 月度自校 |
| `learning/model_learner.yaml` | sample_weights / calibration / absolute_threshold 三组参数 |
| `models/predictor.assign_global_signals()` | 双门槛：`rank_pct < buy_top_pct AND prob_cal >= abs_threshold` |
| `scripts/backfill_hit_tier.py` | 历史 outcome 批量回填 hit_tier（幂等） |
| `scripts/backfill_market_state.py` | 沪深 300 历史回放，重塑 200+ 天状态序列 |
| `scripts/replay_learn.py --with-p1` | 198 天验收：baseline vs P1 精准率 + Brier 对比 |

### P2 — 战法层（✅ 已上线）

**目标**：4 战法阈值自适应 + 战法权重学习 + LLM 深度推荐理由。

| 组件 | 职责 |
|------|------|
| `learning/tactic_learner.py` | 4 战法 × 3 状态 = 12 桶独立统计 90 日精准率；阈值按 direction 标记微调；权重 = 精准率归一化 |
| `learning/tactic_learner.yaml` | defaults / bear override / bounds / step / evaluation / weights |
| `server/predict_cmd._enrich_tactic_scores` | 读 `tactic_params.json[state]` 替代硬编码阈值 |
| `server/predict_cmd._apply_resonance_boost` | 共振股(≥2 战法命中)调 `global_rank_pct`,不动 `rise_prob_cal` |
| `server/ai_reason.py` | Qwen3 → Claude Haiku → None 三级降级；24h 缓存；日封顶 50 |

**P2 关键设计决策**(详见 [spec](docs/superpowers/specs/2026-05-09-antenna-learning-p2-tactic-layer-design.md))：

- **D3 共振机制**：不动 `rise_prob_cal`（保留 calibrator 真实命中率承诺），改为调整 `global_rank_pct`
- **D4 bear 桶兜底**：当前 history bear=0 天，`defaults_bear_override` 手工调严（ROE/负债要求更高）
- **D5 阈值收严方向**：每阈值附 `direction: tighten_up / tighten_down`，修复旧版"全部 +step"对 max 类阈值方向错误的 bug

**成本**：ai_reason 预期 0.5 元/月（缓存命中 ≥50%），日调用硬封顶 50 次。

### P3/P4 — 路线图（⏳ 未启动）

**P1 投产流程**：

```bash
# 1. 一次性回填（幂等，可重复执行）
python scripts/backfill_hit_tier.py
python scripts/backfill_market_state.py

# 2. 首次触发
python cli.py learn          # 生成 learning/model_learner.json
python cli.py learn --check  # 验证学习产物 JSON 合法

# 3. 加权重训（默认日级不触发，由 Antenna-WeeklyTrain 周日 20:00 运行）
python cli.py train --weighted

# 4. 回放验收
python scripts/replay_learn.py --with-p1 --abs-threshold 0.30
```

**实测效果**（最近 27 天 range 市场）：

| 指标 | 基线 | P1 | Δ | 硬指标 |
|------|------|----|---|--------|
| 买入精准率 | 21.66% | **37.50%** | +15.84pt | ≥ +2pt ✅ |
| Mean Brier | 0.234 | 0.168 | ratio 0.719 | ≤ 0.95 ✅ |
| Calibrator 拟合率 | — | 100% | — | — |

P1 的核心价值是**把用户看到的概率校准到真实命中率**——避免"模型说 70% 实际只中 30%"的过度乐观。

### P3/P4 — 路线图（⏳ 未启动）

- **P3 特征层**：`feature_learner`（Permutation Importance + 8 周淘汰）+ `alt_data`（主力净流入/龙虎榜/北向资金）
- **P4 价位层**：`price_learner`（ATR / 振幅系数网格搜索）

### 横向 — 黑名单（✅ 已上线）

| 组件 | 职责 |
|------|------|
| `learning/blacklist.py` | 同 `(code, state)` 连续 N 次 buy+miss/weak → 拉黑 30 天 |
| `learning/blacklist.yaml` | streak_threshold / block_days / per_state_threshold（bear 更严格 2 次即拉黑）|
| `server/predict_cmd.cmd_scan_bot` | `assign_global_signals` 之前过滤黑名单股票 |
| **白名单 override** | watchlist 内的自选股永不参与拉黑判定 |
| **自动过期** | 编排器每次跑都清理 `expires < today` 条目 |

---

## 策略自优化

P0 内置自适应选股门槛，每日收盘后自动运行（独立于 P1 校准链路）。

**核心指标**：买入精准率 = `signal=买入 且 actual_pct ≥ 1%` 的股票数 / 全部"买入"信号数。

| 近 7 日精准率 | 调整动作 |
|-------------|---------|
| < 35% | 门槛收严（−1%） |
| 35%–55% | 维持 |
| 55%–65% | 门槛微放（+1%） |
| > 65% | 门槛放宽（+2%） |
| 连续 7 天触底（8%） | 自动重置为 15%，重新探索 |

目标精准率 **55%**（A 股短线现实水平），状态持久化在 `learning/strategy.json`。

P1 在此之上追加 `abs_threshold`（绝对概率门槛，月度自校），形成双门槛体系。

---

## PM2 进程管理

生产环境推荐通过 PM2 托管，崩溃后自动重启、日志持久化、Web 监控面板一体化。

### 前置条件

```bash
# 安装 PM2（需要 Node.js）
npm install -g pm2
```

### ecosystem.config.cjs 进程清单

| 进程名 | 启动脚本 | 说明 |
|--------|---------|------|
| `antenna-bot` | `server/start.cjs` → `feishu_poll.py` | 飞书机器人（主进程） |
| `pm2-monitor` | `server/pm2_monitor.py` | Web 监控面板（Flask） |

### 常用命令

```bash
# 启动全部
pm2 start ecosystem.config.cjs

# 查看进程状态
pm2 status

# 重启飞书机器人
pm2 restart antenna-bot

# 停止全部
pm2 stop all

# 持久化进程列表（系统重启后自动拉起）
pm2 save
pm2 startup           # 生成自启动脚本（Linux/macOS）
# Windows 下改用任务计划调用 `pm2 resurrect`

# 查看日志
pm2 logs antenna-bot --lines 100
pm2 logs pm2-monitor  --lines 50
```

日志文件默认路径：

```
logs/pm2-out.log           # antenna-bot 标准输出
logs/pm2-error.log         # antenna-bot 错误
logs/pm2-monitor-out.log   # pm2-monitor 标准输出
logs/pm2-monitor-error.log # pm2-monitor 错误
```

### Web 监控面板（pm2-monitor）

`server/pm2_monitor.py` 提供 Basic Auth 保护的 Web 面板，支持查看进程状态、Restart / Stop / Start 操作、查看最近日志。

在 `config.yaml` 中配置：

```yaml
pm2_monitor:
  host:      "0.0.0.0"    # 对外开放改为 0.0.0.0；仅本机访问保持 127.0.0.1
  port:      9615
  username:  "admin"
  password:  "your_password"   # 必填，留空则拒绝启动
  log_lines: 200               # /logs 页面显示最近多少行
```

面板地址：`http://<host>:9615`（每 10 秒自动刷新）。健康检查端点（无鉴权）：`http://<host>:9615/healthz`。

> **注意**：`password` 未配置时 pm2-monitor 会拒绝启动，防止面板裸奔。

---

## 定时任务

`scripts/setup_tasks.ps1` 一键注册以下 Windows 任务计划（需管理员）：

| 任务 | 时间 | 功能 |
|------|------|------|
| `Antenna-Train` | 每日 09:00 | 重训练模型 |
| `Antenna-Scan` | 工作日 09:15 | 全市场扫描推送 |
| `Antenna-Predict-AM` | 工作日 09:30 起每 15 min（135 min） | 上午盘中预测 |
| `Antenna-Predict-PM` | 工作日 13:00 起每 15 min（135 min） | 下午盘中预测 |
| `Antenna-Noon-Review` | 工作日 11:32 | 午间复盘 |
| `Antenna-Daily-Review` | 工作日 15:32 | 收盘复盘 + 策略调整 |
| `Antenna-WeeklyTrain` | **周日 20:00** | ★ P1 加权重训 |
| `Antenna-LearnWeekly` | **周日 02:30** | ★ P3/P4 学习编排（特征剪枝 + 价位网格搜索） |

```powershell
# 注册所有任务
powershell -File scripts/setup_tasks.ps1

# 手动触发
schtasks /run /tn Antenna-Scan
```

---

## AI 闲聊

无法识别的消息自动转交内置 AI。角色设定为**方木木**：游戏公司后端开发，懒散口吻、简洁直接，每 5 分钟自动切换"当前状态"以保持回复多样性。

**模型调用链**（优先级从高到低）：

1. **阿里云 Qwen3**（`qwen.api_key` 非空）
2. **Anthropic Claude Haiku**（Qwen 不可用时回退）
3. 内置兜底回复

对话历史按 `chat_id` 独立保留，最多 20 轮上下文。

### 风格采集（可选）

```bash
# 列出可采集的 P2P 私聊
python cli.py style --list-chats

# 按姓名搜索 open_id
python cli.py style --find-user 木

# 采集并提炼到 learning/persona.txt
python cli.py style --dry-run    # 演练
python cli.py style              # 真实写入
```

---

## 技术栈

- **数据**：akshare（A 股行情 / 财报）+ 新浪 HQ API（实时报价）
- **特征**：pandas / numpy / pandas_ta（25 个技术指标）
- **模型**：LightGBM 4.x（二分类）+ scikit-learn IsotonicRegression（概率校准）
- **配置**：PyYAML
- **机器人**：requests（飞书 Open Platform）
- **进程管理**：PM2（Node.js）+ Windows 任务计划
- **测试**：pytest + pytest-cov（200+ 用例覆盖学习系统全链路）

---

## 许可

本项目为个人学习与交易辅助工具，**不构成任何投资建议**。模型输出仅供参考，使用前请理解风险并自行验证。
