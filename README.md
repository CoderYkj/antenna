# Antenna — A 股量化分析与飞书交易助手

基于 LightGBM + isotonic 概率校准的 A 股交易辅助系统，提供：

- **AI 选股**：全市场扫描 → 双门槛信号（分位排名 + 校准概率）
- **战法过滤**：四大战法共振评分，仅推荐同时获 AI 买入信号 + 战法认可的股票
- **双周期价位**：每只候选股自动生成短线 1-5 日 + 长线 2-4 周买卖止损位（ATR/振幅系数自学习）
- **自主学习**：6 模块编排器每日/每周自动运行，覆盖概率校准、战法参数、特征剪枝、价位优化
- **多通道推送**：飞书双向交互 + 企业微信推送，通道并行扇出，任一失败不影响其他
- **Walk-Forward 回测** + **策略自优化**（精准率监控 → 选股门槛动态调整）

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置文件](#配置文件)
- [CLI 命令p](#cli-命令)
- [飞书机器人](#飞书机器人)
- [企业微信](#企业微信)
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
│   ├── fetcher.py                 # 行情数据拉取（akshare + 新浪实时，name_map 进程内缓存）
│   ├── alt_fetcher.py             # P3 alt 数据（主力净流入/龙虎榜/北向资金），缓存到 cache/alt/
│   ├── universe.py                # 股票池管理
│   ├── name_map.json              # 代码 → 名称映射（24h 文件缓存 + 1h 进程内缓存）
│   ├── sector_map.json            # 代码 → 行业映射
│   └── cache/
│       ├── *.parquet              # 行情 Parquet 缓存（自动）
│       └── alt/                   # P3 alt 特征日缓存（YYYY-MM-DD.parquet）
│
├── features/
│   ├── builder.py                 # 特征工程流水线（支持注入 alt 特征）
│   ├── technical.py               # 27 个技术指标 + get_active_feature_cols()
│   ├── analyser.py                # 技术分析描述、双周期价位（动态 ATR/振幅系数）、新闻抓取
│   └── fundamental.py             # 财务报表解构与评分
│
├── models/
│   ├── trainer.py                 # LightGBM 训练（含 train_weighted）
│   ├── predictor.py               # 推理 + 全市场信号双门槛分配
│   └── saved/                     # 模型 pkl + calibrator pkl（gitignored）
│
├── learning/                      # 学习系统（P0 / P1 / P2 / P3 / P4 + 横向黑名单）
│   ├── orchestrator.py            # 编排器：6 模块依赖调度 + check() 语义校验
│   ├── tracker.py                 # pred / outcome JSONL 归档（含 scene / market_state 标签）
│   ├── outcome_metrics.py         # hit_tier（miss / weak / good / great）+ 5 日指标
│   ├── market_state.py            # 大盘状态打标（bull / bear / range），3 日防抖
│   ├── feedback_io.py             # 原子写 + 7 份历史快照
│   ├── alerts.py                  # 飞书 + JSONL 三路告警
│   ├── optimizer.py               # buy_top_pct 日级自动调参
│   ├── scene_bucket.py            # 按 scene / state 分桶工具
│   ├── backtest_history.py        # Walk-Forward 结果存档
│   ├── model_learner.py           # P1 isotonic 校准 + abs_threshold 自校 + 加权重训
│   ├── model_learner.yaml         # P1 可配置参数
│   ├── tactic_learner.py          # P2 4战法×3状态=12桶阈值自适应 + 权重学习
│   ├── tactic_learner.yaml        # P2 defaults / bear override / bounds / step
│   ├── blacklist.py               # 横向黑名单：连错股票自动拉黑 30 天
│   ├── blacklist.yaml             # streak_threshold / block_days / per_state_threshold
│   ├── feature_learner.py         # P3 Permutation Importance 特征剪枝 + alt IC 筛选
│   ├── feature_learner.yaml       # P3 ic_threshold / min_samples / decay 参数
│   ├── price_learner.py           # P4 ATR/振幅系数网格搜索，按 market_state 分桶
│   ├── price_learner.yaml         # P4 网格范围 / cold_start 默认值
│   ├── strategy.json              # 当前选股策略状态（自动维护）
│   ├── model_learner.json         # P1 校准状态 + threshold 历史
│   ├── tactic_params.json         # P2 战法参数 + 权重（自动维护）
│   ├── blacklist.json             # 当前黑名单条目（自动维护）
│   ├── feature_weights.json       # P3 特征重要性 + active 列列表（自动维护）
│   ├── price_params.json          # P4 各状态最优 ATR/振幅系数（自动维护）
│   ├── market_state.json          # 大盘状态历史（含 365 天容量）
│   └── persona.txt                # AI 闲聊人设
│
├── server/
│   ├── feishu_poll.py             # 飞书轮询（每 5s，并发指令支持）
│   ├── commands.py                # 指令词集合 + 路由
│   ├── predict_cmd.py             # 全部飞书指令执行体（含 alt_data 注入）
│   ├── ai_reason.py               # P2 LLM 深度推荐理由（Qwen3 → Claude Haiku 三级降级）
│   ├── watchlist.py               # 自选股管理
│   ├── style_learner.py           # 风格采集 → persona.txt
│   ├── feishu_push.py             # 主动推送
│   └── pm2_monitor.py             # PM2 状态监控面板（Flask + Basic Auth）
│
├── notify/
│   ├── dispatcher.py              # 多通道扇出：并行推送飞书 + 企业微信，任一失败不影响其他
│   ├── feishu.py                  # 飞书 Webhook 单向推送（含学习完成通知）
│   ├── wecom.py                   # 企业微信自建应用推送（access_token 缓存、自动刷新）
│   └── render.py                  # 消息渲染工具（Markdown 格式化）
│
├── scripts/
│   ├── backfill_hit_tier.py       # P1 历史 hit_tier 批量回填（幂等）
│   ├── backfill_market_state.py   # P1 沪深 300 历史回放
│   ├── replay_learn.py            # 学习系统回放 / P1 验收
│   ├── check_alt_ic.py            # P3 alt 特征 IC 有效性离线验证
│   ├── task_scan.py               # 定时扫描（含 alt_data 批量拉取）
│   ├── task_predict.py            # 盘中预测（动态特征列 + alt_data 注入）
│   ├── task_daily_review.py       # 收盘复盘
│   ├── task_noon_review.py        # 午间复盘
│   ├── task_train.py              # 定时训练
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
- `qwen.api_key` 或 `anthropic.api_key` 或 `deepseek.api_key` — AI 闲聊（三选一）

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

# PM2 托管启动（生产推荐；Windows / Linux 通用）
pm2 start ecosystem.config.cjs   # 同时启动 antenna-bot + pm2-monitor
pm2 save                          # 持久化进程列表，重启系统后自动恢复

# 或注册为 Windows 任务计划（无 PM2 时的替代方案）
powershell -File scripts/setup_tasks.ps1
```

### 6. 触发首次学习

```bash
python cli.py learn          # 运行全部 6 个学习模块
python cli.py learn --check  # 验证产物 JSON 合法性（含语义校验）
```

---

## 配置文件

`config.yaml` 已脱离 git 跟踪（避免凭据外泄）。模板见 `config.example.yaml`，必填项：

| 段                     | 字段                    | 说明                              |
| ---------------------- | ----------------------- | --------------------------------- |
| `data.default_days`  | int                     | 拉取天数（默认 365）              |
| `universe.watchlist` | list                    | 自选股代码列表                    |
| `universe.scan_pool` | `watchlist` / `all` | 扫描范围                          |
| `model.target_days`  | int                     | 预测窗口（默认 5 日）             |
| `model.threshold`    | float                   | 涨幅判定阈值（默认 0.02）         |
| `feishu.app_id`      | str                     | 自建应用 App ID（双向必填）       |
| `feishu.app_secret`  | str                     | 自建应用 Secret                   |
| `feishu.chat_ids`    | list                    | 监控的会话 ID（`oc_` 为群）     |
| `feishu.webhook_url` | str                     | 单向推送 webhook（可选）          |
| `qwen.api_key`       | str                     | 阿里云 DashScope（AI 闲聊首选）   |
| `deepseek.api_key`   | str                     | DeepSeek API（AI 闲聊备选）       |
| `anthropic.api_key`  | str                     | Claude Haiku（AI 闲聊兜底）       |
| `channels.feishu`    | bool                    | 飞书推送开关（默认 true）         |
| `channels.wecom`     | bool                    | 企业微信推送开关（默认 false）    |
| `wecom.corp_id`      | str                     | 企业微信企业 ID                   |
| `wecom.agent_id`     | int                     | 自建应用 AgentId                  |
| `wecom.secret`       | str                     | 自建应用 Secret                   |
| `ai_reason.enable`   | bool                    | LLM 深度推荐理由开关（默认 true） |

---

## CLI 命令

```bash
python cli.py <subcommand> [options]
```

| 子命令             | 功能                     | 关键选项                                            |
| ------------------ | ------------------------ | --------------------------------------------------- |
| `fetch`          | 拉取行情数据             | `--code` / `--all` / `--days` / `--workers` |
| `train`          | 训练 LightGBM 模型       | `--weighted`（P1 加权）/ `--workers`            |
| `predict <code>` | 单股预测 + HTML 报告     | —                                                  |
| `scan`           | 全市场扫描 Top N         | `--top` / `--workers`                           |
| `report <code>`  | 生成 HTML 报告（不重训） | —                                                  |
| `backtest`       | Walk-Forward 历史回测    | `--month YYYY-MM` / `--year YYYY` / `--top`   |
| `learn`          | 触发学习系统编排器       | `--date YYYY-MM-DD` / `--dry-run` / `--check` |
| `style`          | 采集飞书同事聊天风格     | `--list-chats` / `--find-user` / `--dry-run`  |
| `bot <cmd>`      | 执行任意飞书指令并广播   | 例:`bot 推荐 5`                                   |

---

## 飞书机器人

### 主动推送（单向 Webhook）

`notify/feishu.py` 通过 `webhook_url` 推送，定时任务的扫描结果、复盘报告和学习完成通知由此发出。无需自建应用，配置 `feishu.webhook_url` 即可。

### 双向交互（自建应用）

`server/feishu_poll.py` 每 5 秒轮询消息，支持自然语言模糊匹配（`预测 茅台` → 自动解析到 `600519`）。需要自建应用 + 填入 `app_id` / `app_secret` / `chat_ids`。

**即时反馈**：重型指令（`推荐` / `战法`）收到后**立即返回进度卡片**（含步骤说明和预估时长），后台处理完成后自动推送结果卡片。

**并发**：最多同时处理 3 条重型指令（预测 / 扫描 / 回测 / 学习），其余排队；闲聊不占槽位。

### 指令清单

群中需要 `@机器人`，私聊直接发送即可。

#### 自选股管理

| 指令                  | 示例            | 说明           |
| --------------------- | --------------- | -------------- |
| `添加 <代码或名称>` | `添加 600519` | 加入自选       |
| `删除 <代码或名称>` | `删除 茅台`   | 移出自选       |
| `列表`              | `列表`        | 自选股实时行情 |

#### 行情与分析

| 指令            | 示例              | 说明                                         |
| --------------- | ----------------- | -------------------------------------------- |
| `行情 <code>` | `行情 002174`   | 实时价格 + 日内温度计 + 分时（黑名单警告）   |
| `预测 <code>` | `预测 300785`   | AI 信号 + 技术分析 + 双周期价位 + 学习上下文 |
| `趋势 <code>` | `趋势 600519`   | 日 / 周 / 月线文字趋势图                     |
| `新闻 <code>` | `新闻 紫金矿业` | 财联社近期资讯（按情绪分类）                 |
| `财报 <code>` | `财报 600519`   | 季报 / 年报深度解构                          |

#### 扫描与战法

| 指令                | 示例                | 说明                                            |
| ------------------- | ------------------- | ----------------------------------------------- |
| `推荐 (N)`        | `推荐 3`          | 全市场扫描，仅推荐 AI 买入信号 + 战法认可的股票 |
| `战法 <策略> (N)` | `战法 价值 5`     | 按战法筛选 Top N                                |
| `复盘 (日期)`     | `复盘 2026-04-09` | 历史命中率 + 策略调整记录                       |
| `历史 YYYY-MM`    | `历史 2026-01`    | 月度 Walk-Forward 回测                          |
| `历史 YYYY`       | `历史 2026`       | 年度回测                                        |

#### 学习系统

| 指令             | 示例                           | 说明                                  |
| ---------------- | ------------------------------ | ------------------------------------- |
| `学习 (日期)`  | `学习` / `学习 2026-04-29` | 跑全部 6 个学习模块                   |
| `学习 dry-run` | `学习 演练`                  | 演练，不写盘                          |
| `策略`         | `策略`                       | 当前选股门槛、精准率、7天/30天看板核心指标 |

#### 其他

| 指令                        | 说明                       |
| --------------------------- | -------------------------- |
| `发送 <子指令>`           | 执行子指令并广播到所有会话 |
| `帮助` / `help` / `?` | 完整指令列表               |

无法识别的消息自动转交 [AI 闲聊](#ai-闲聊)。

---

## 企业微信

`notify/wecom.py` 通过**自建应用**推送消息，与飞书 webhook 并列为第二推送通道。

### 启用步骤

1. 在 [企业微信管理后台](https://work.weixin.qq.com/wework_admin/loginpage_wx) 创建自建应用，获取 `corp_id` / `agent_id` / `secret`
2. 填入 `config.yaml`：

```yaml
channels:
  feishu: true
  wecom:  true          # 改为 true 启用

wecom:
  corp_id:   "ww..."   # 企业 ID
  agent_id:  1000001   # 自建应用 AgentId（整数）
  secret:    "..."     # 应用 Secret
```

3. 重启进程即生效，无需修改代码

### 双向对话（可选 Phase 3）

如需企业微信接收消息（双向对话），还需公网 HTTPS 回调地址：

```yaml
wecom:
  callback_token:    "..."   # 接收消息 → Token
  encoding_aes_key:  "..."   # 接收消息 → EncodingAESKey（43 位）
  callback_port:     5000
  callback_path:     /wecom/callback
  allowed_users:     []      # 指令白名单，空列表=不限制
```

### 多通道扇出架构

`notify/dispatcher.py` 将推送请求**并行扇出**到所有已启用通道：

```
send_scan_result(result)
    ├── feishu（channels.feishu=true）
    └── wecom（channels.wecom=true）
```

任一通道异常（网络超时、凭据错误等）均不影响其他通道正常推送，每通道超时 15 秒。

---

## 选股战法

### 推荐指令过滤逻辑

`推荐 N` 指令采用**三重过滤**，确保只展示真正值得买入的股票：

1. **候选池**：AI 高分前 `N×4` 只（按涨幅概率 + 动量降序）
2. **战法门**：`_tier_split` 按战法共振度分层
   - tier1：`tactic_resonance ≥ 2`（多战法共振，优先展示）
   - tier2：`tactic_resonance = 1`（单战法命中）
   - tactic = 0：直接排除
3. **信号门**：仅保留 AI `signal = "买入"` 的股票（双门槛：分位排名 + 校准概率）

**空结果处理**：

- 无买入+战法共振股 → 展示**黄色「值得关注」卡片**（观望信号 + 战法认可，附说明"等待信号确认后操作"）
- 两类均为空 → 灰色「今日暂无推荐」卡片

**示例输出**（正常结果）：

```
共扫描 5193 只，AI 买入信号 12 只。
战法筛选后推荐 3 只（★共振 2 / ★单战法 1）
市场状态 range　活跃特征 24

Top 3 推荐
1. 炜冈科技（001256）　买入　全市场前 0.04% [高]　★价值+龙头
　💹  现价 24.07  涨概率 62.4%  短线 +16.0%  长线 +12.7%
```

### 四大战法

`战法` 指令基于**财务 + 技术**双重筛选，均输出上涨概率、双周期目标涨幅与买卖止损位。

| 战法            | 英文                  | 核心条件                             |
| --------------- | --------------------- | ------------------------------------ |
| 📊 价值投资法   | `value` / `价值`  | PE 低 · PB 低 · ROE 高 · 低负债   |
| 🚀 成长股投资法 | `growth` / `成长` | 营收高增速 · 净利高增速 · ROE 优秀 |
| 👑 行业龙头战法 | `leader` / `龙头` | ROE > 15% · 毛利 > 30% · 技术强势  |
| 🔄 逆向投资法   | `contra` / `逆向` | 技术超跌 > 15% · 基本面稳健         |

### 双周期价位

| 周期           | 买入           | 止盈                | 止损                   |
| -------------- | -------------- | ------------------- | ---------------------- |
| 短线（1-5 日） | MA5 / 布林下轨 | 布林上轨 / 预测高位 | ATR×系数（P4 自学习） |
| 长线（2-4 周） | MA20           | MA60 / 振幅×系数   | MA60 下方（P4 自学习） |

止损/止盈系数由 `price_learner`（P4）每周日网格搜索最优值，按 bull/bear/range 三状态分别存储。

---

## 学习系统 (Learning)

系统分阶段升级"自主学习"能力，由飞书 `学习` 指令、定时任务或 `cli.py learn` 触发。编排器 `orchestrator.py` 按依赖顺序运行 6 个模块，任一模块失败不影响其他模块。**学习完成后自动推送飞书通知**，含各模块状态与耗时。

### P0 — 地基层（已上线）

| 模块                            | 职责                                                                 |
| ------------------------------- | -------------------------------------------------------------------- |
| `learning/market_state.py`    | 大盘状态打标（bull/bear/range），3 日防抖                            |
| `learning/tracker.py`         | pred / outcome JSONL 归档，含 scene / market_state 场景标签          |
| `learning/outcome_metrics.py` | hit_tier（miss / weak / good / great）+ 5 日指标                     |
| `learning/orchestrator.py`    | 编排器：6 模块依赖调度，模块失败隔离，dry-run 支持，check() 语义校验 |
| `learning/optimizer.py`       | 精准率监控 +`buy_top_pct` 日级自动调参                             |
| `learning/feedback_io.py`     | 原子写 + 7 份历史快照                                                |
| `learning/alerts.py`          | 飞书 + JSONL 三路告警                                                |

### P1 — 模型层（已上线）

**目标**：从"纯分位切信号"升级为"概率校准 + 错样本加权重训 + 绝对阈值 gate"。

| 组件                                         | 职责                                                             |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `learning/model_learner.py`                | isotonic 校准（按 market_state 分 3 桶）+ abs_threshold 月度自校 |
| `learning/model_learner.yaml`              | sample_weights / calibration / absolute_threshold 三组参数       |
| `models/predictor.assign_global_signals()` | 双门槛：`rank_pct < buy_top_pct AND prob_cal >= abs_threshold` |
| `scripts/backfill_hit_tier.py`             | 历史 outcome 批量回填 hit_tier（幂等）                           |
| `scripts/backfill_market_state.py`         | 沪深 300 历史回放，重塑 200+ 天状态序列                          |
| `scripts/replay_learn.py --with-p1`        | 198 天验收：baseline vs P1 精准率 + Brier 对比                   |

**实测效果**（最近 27 天 range 市场）：

| 指标       | 基线   | P1               | Δ          |
| ---------- | ------ | ---------------- | ----------- |
| 买入精准率 | 21.66% | **37.50%** | +15.84pt    |
| Mean Brier | 0.234  | 0.168            | ratio 0.719 |

### P2 — 战法层（已上线）

**目标**：4 战法阈值自适应 + 战法权重学习 + LLM 深度推荐理由。

| 组件                                          | 职责                                                                    |
| --------------------------------------------- | ----------------------------------------------------------------------- |
| `learning/tactic_learner.py`                | 4 战法 × 3 状态 = 12 桶独立统计 90 日精准率；阈值按 direction 标记微调 |
| `server/predict_cmd._enrich_tactic_scores`  | 读 `tactic_params.json[state]` 替代硬编码阈值                         |
| `server/predict_cmd._apply_resonance_boost` | 共振股（≥2 战法命中）调 `global_rank_pct`，不动 `rise_prob_cal`    |
| `server/predict_cmd._tier_split`            | 按战法共振度分层（tier1≥2, tier2=1, 0战法排除）                        |
| `server/ai_reason.py`                       | Qwen3 → Claude Haiku → None 三级降级；24h 缓存；日封顶 50 次          |

**关键设计决策**：

- **共振机制**：不动 `rise_prob_cal`（保留 calibrator 真实命中率承诺），改为调整 `global_rank_pct`
- **推荐过滤**：tier_split 后再加 signal == "买入" 后置过滤，回避/观望股不进推荐
- **bear 桶兜底**：history bear=0 天时，`defaults_bear_override` 手工调严

### P3 — 特征层（已上线）

**目标**：动态特征剪枝（淘汰无 IC 特征）+ alt_data 补充外部信号。

| 组件                            | 职责                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `data/alt_fetcher.py`         | 每日拉取 5 个 alt 特征，缓存到 `data/cache/alt/YYYY-MM-DD.parquet`          |
| `features/builder.py`         | `build_features(df, alt=None)` 支持注入 alt 特征                            |
| `features/technical.py`       | `get_active_feature_cols()` 读 `feature_weights.json` 返回剪枝后的列列表  |
| `learning/feature_learner.py` | Permutation Importance 评估所有特征，IC < 0.02 或 8 周连续末位的列移出 active |
| `scripts/check_alt_ic.py`     | 离线验证 alt 特征 IC 分布                                                     |

**5 个 alt 特征**：

| 特征                   | 来源               | 说明                       |
| ---------------------- | ------------------ | -------------------------- |
| `main_net_in_1d`     | akshare 主力净流入 | 当日主力净买入额           |
| `main_net_in_5d`     | akshare 主力净流入 | 近 5 日主力净买入额        |
| `dragon_top_cnt_10d` | akshare 龙虎榜     | 近 10 日上榜次数           |
| `sector_heat_rank`   | akshare 行业热度   | 所属行业热度排名（归一化） |
| `north_hold_chg_5d`  | akshare 北向持仓   | 近 5 日北向持仓变化        |

### P4 — 价位层（已上线）

**目标**：止损/止盈系数不再硬编码，由历史命中率数据网格搜索最优值。

| 组件                                               | 职责                                                                                           |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `learning/price_learner.py`                      | 4 参数（short_atr_mult / short_gain_mult / long_amp_mult / long_ma60_buffer）× 3 状态网格搜索 |
| `learning/price_learner.yaml`                    | 网格范围 + cold_start 默认值                                                                   |
| `features/analyser.suggest_dual_period_trades()` | 读 `price_params.json[state]` 动态加载系数，fallback 到 yaml 默认值                          |

网格搜索目标函数：`(止盈命中次数 − 止损触发次数) / 样本数`，每周日按 bull/bear/range 分别寻优。

### 横向 — 黑名单（已上线）

| 组件                                | 职责                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `learning/blacklist.py`           | 同 `(code, state)` 连续 N 次 buy+miss/weak → 拉黑 30 天                    |
| `learning/blacklist.yaml`         | streak_threshold / block_days / per_state_threshold（bear 更严格 2 次即拉黑） |
| `server/predict_cmd.cmd_scan_bot` | `assign_global_signals` 之前过滤黑名单股票                                  |
| `server/predict_cmd.cmd_quote`    | 行情指令展示黑名单警告                                                        |
| **白名单 override**           | watchlist 内的自选股永不参与拉黑判定                                          |
| **自动过期**                  | 编排器每次跑都清理 `expires < today` 条目                                   |

### 学习系统状态面板

`策略` 指令展示完整学习状态面板：

```
P1 校准　市场 range　abs_threshold 0.38　上次校准 2026-05-18
P2 战法　value 61%（10 样本）· growth 55%（8 样本）· ...
P3 特征　活跃特征 24　上次更新 2026-05-18
P4 价位　short_atr_mult 1.3　short_gain_mult 1.8（range 市场）
```

`预测` 指令在结果中追加单行学习上下文（市场状态 / 校准门槛 / 活跃特征数 / ATR 系数）。

### 投产流程

```bash
# 一次性回填（幂等，可重复执行）
python scripts/backfill_hit_tier.py
python scripts/backfill_market_state.py

# 首次触发全量学习
python cli.py learn

# 验证产物合法性（含 feature_weights.json active 列语义校验）
python cli.py learn --check

# 加权重训（由 Antenna-WeeklyTrain 周日 20:00 自动运行）
python cli.py train --weighted

# 验收回放
python scripts/replay_learn.py --with-p1 --abs-threshold 0.30

# 验证 alt 特征 IC（积累 10+ 交易日后运行）
python scripts/check_alt_ic.py
```

---

## 策略自优化

P0 内置自适应选股门槛，每日收盘后自动运行（独立于 P1 校准链路）。

**核心指标**：买入精准率 = `signal=买入 且 actual_pct ≥ 1%` 的股票数 / 全部"买入"信号数。

| 近 7 日精准率       | 调整动作                 |
| ------------------- | ------------------------ |
| < 35%               | 门槛收严（−1%）         |
| 35%–55%            | 维持                     |
| 55%–65%            | 门槛微放（+1%）          |
| > 65%               | 门槛放宽（+2%）          |
| 连续 7 天触底（8%） | 自动重置为 15%，重新探索 |

目标精准率 **55%**，状态持久化在 `learning/strategy.json`。P1 在此之上追加 `abs_threshold`（绝对概率门槛，月度自校），形成双门槛体系。

另外新增了**风险护栏**（同样写入 `learning/strategy.json`）：

- 按市场状态动态限制 `buy_top_pct` 范围（bull/range/bear 不同上限与下限）
- 当 30 日买入精准率低于阈值（默认 40%）时，进一步收紧 `buy_top_pct` 上限（默认 12%）
- 策略面板会同步展示当前状态下的护栏范围与仓位建议（bull 100% / range 60% / bear 30%）
- `推荐` 扫描结果卡片会同步展示护栏提示（当前门槛、状态区间、建议仓位、是否触发 30 日收敛上限）
- `复盘` / `task_daily_review.py` 会落地并展示护栏追踪（门槛变更前后、状态区间、触发原因），用于审计与回溯
- 护栏触发原因会自动转换为中文标签（如“状态区间限幅”“30日低精准率上限收敛”），便于运营复盘
- `策略` / `复盘` 页面会展示近 30 日护栏触发统计与主要原因分布，便于观察风控压力变化
- `策略` / `复盘` 新增 7 天与 30 天监控看板核心字段：命中率、样本量、均笔收益、最大回撤、Top5 命中率
- `复盘` 卡片会基于 7d/30d 对比自动给出“提精风险告警”（如过度降级风险、短期命中率漂移、短期回撤恶化）
- `策略` 指令页也会同步展示“提精风险告警”，便于在会话内即时决策参数回调
- `推荐` 新增“提精闸门”：当 7/30 日命中率走弱且样本充足时，自动将低质量买入候选降级为观望（可通过 `scan.recommend_precision_gate` 配置）
- 提精闸门新增“概率分桶可靠性”检查：若某概率区间在近 30 日命中率偏低且样本充足，会优先降级该区间买入候选，减少系统性误报
- 提精闸门新增“置信度分层可靠性”检查：若高/中/低置信度桶近期命中率低于阈值，会自动收紧对应桶的买入候选
- 提精闸门新增“战法可靠性”检查：若命中战法在当前市场状态下样本充足但历史精准率偏弱，会降级相关买入候选以降低误报
- 提精闸门新增“个股近期命中记忆”检查：若个股近 30 日买入样本命中率持续偏弱，会自动降级该标的买入信号
- 提精闸门新增“收益/风险性价比”检查：若预期涨幅不足或相对历史回撤的收益风险比偏低，会降级对应买入候选
- 提精闸门新增“技术共振确认”检查：当 RSI / MACD / 量比 / 价格趋势确认数不足时，会在弱势期自动降级买入候选
- 提精闸门新增“个股回撤风险记忆”检查：若个股近阶段 5 日深回撤发生率过高，会自动降级对应买入候选
- 提精闸门新增“流动性质量”检查：当换手率过低/过热或量比不足时，会自动降级对应买入候选
- 提精闸门新增“趋势一致性”检查：当均线结构与收盘位置不满足趋势对齐条件时，会自动降级对应买入候选
- 提精闸门新增“趋势过热偏离”检查：当收盘价相对 MA20 偏离过大（追高风险）时，会自动降级对应买入候选
- 提精闸门新增“极端波动”检查：当 ATR/价格波动比异常偏高时，会自动降级对应买入候选
- 提精闸门新增“概率边际”检查：当个股涨概率仅略高于全市场买入门槛时，会自动降级边际信号候选
- 提精闸门新增“双周期目标一致性”检查：当短线/长线目标涨幅不足或长线相对短线失衡时，会自动降级对应买入候选
- 提精闸门新增“ATR收益比”检查：当预期涨幅相对 ATR 波动补偿不足时，会自动降级对应买入候选
- 冲刺后复盘新增“自适应阈值调优”：在弱势边缘自动轻放松（soft），在显著失压时自动收紧（strict），并按样本量自动衰减调优强度，减少小样本过调
- 参数收敛细化：弱模式默认下调了边际/流动性/ATR收益比等阈值的“过严”程度（并保留 hard 模式收紧），降低误杀高质量候选

---

## PM2 进程管理

生产环境推荐通过 PM2 托管，崩溃后自动重启、日志持久化、Web 监控面板一体化。

### 前置条件

```bash
npm install -g pm2
```

`ecosystem.config.cjs` 现已使用**相对路径 + 自动选择 Python 解释器**，Windows / Linux 均可直接复用；若你的环境没有 `python`/`python3` 命令，可在启动前指定：

```bash
export ANTENNA_PYTHON=/usr/bin/python3
pm2 start ecosystem.config.cjs
```

### 进程清单（ecosystem.config.cjs）

| 进程名 | 启动脚本 | 说明 |
|--------|---------|------|
| `antenna-bot` | `server/feishu_poll.py` | 飞书机器人（主进程） |
| `pm2-monitor` | `server/pm2_monitor.py` | Web 监控面板（Flask） |

> 默认会设置 `TQDM_DISABLE=1`，抑制第三方库进度条控制字符，减少 PM2 日志噪音。

### 常用命令

```bash
pm2 start ecosystem.config.cjs   # 启动全部
pm2 status                        # 查看进程状态
pm2 restart antenna-bot           # 重启飞书机器人
pm2 stop all                      # 停止全部
pm2 save                          # 持久化进程列表
pm2 logs antenna-bot --lines 100  # 查看日志
```

日志文件路径：`logs/pm2-out.log` / `logs/pm2-error.log`。

### Linux 更新部署（推荐）

仓库内置一键脚本 `scripts/deploy_linux.sh`，用于：

- 拉取远端最新代码（`git pull --ff-only`）
- 创建/复用虚拟环境并安装依赖
- 重启 PM2 进程并保存
- 可选执行 `scripts/smoke_test.py` 做健康检查

运行手册见：`docs/linux-deploy-runbook.md`

如果你是在本地机器上直接发布到远端 Linux（当前默认目标：`115.29.240.130:/data/antenna`），可使用固定入口：

```bash
# 建议用环境变量提供密码（避免写在命令行历史）
export ANTENNA_REMOTE_PASSWORD='***'
python scripts/deploy_remote.py
```

该命令会自动执行：打包当前 `HEAD` → 上传并解压到远端 → 运行 `deploy_linux.sh`（`SKIP_GIT_PULL=1`）→ `pm2` 重启 → `healthz` 与日志实时页检查。

首次执行：

```bash
cd /path/to/antenna
chmod +x scripts/deploy_linux.sh
DEPLOY_BRANCH=main APP_DIR=/path/to/antenna ./scripts/deploy_linux.sh
```

常用参数：

- `PYTHON_BIN`：Python 命令（默认 `python3`）
- `VENV_DIR`：虚拟环境目录（默认 `.venv`）
- `SKIP_SMOKE_TEST=1`：跳过 smoke test
- `ALLOW_DIRTY=1`：允许有未提交改动时继续部署（默认禁止）
- `SKIP_GIT_PULL=1`：跳过 `git fetch/pull`（仅重装依赖并重启服务）
- `DEPLOY_REF=<commit/tag>`：部署指定提交或标签（detach 模式）
- `ROLLBACK_ON_FAILURE=1`：部署失败自动回滚到部署前版本并尝试拉起 PM2
- `DEPLOY_LOCK_FILE`：部署锁文件路径（默认 `.deploy.lock`，防并发部署）
- `AUTO_TRAIN_IF_MISSING=1`：若未发现 `models/saved/model_*.pkl`，自动训练（默认开启）
- `TRAIN_WEIGHTED_IF_MISSING=1`：模型缺失时使用 `train --weighted` 自动训练

示例（跳过 smoke test）：

```bash
DEPLOY_BRANCH=main SKIP_SMOKE_TEST=1 ./scripts/deploy_linux.sh
```

示例（部署指定 tag，失败自动回滚）：

```bash
DEPLOY_REF=v2.1.0 ROLLBACK_ON_FAILURE=1 ./scripts/deploy_linux.sh
```

如果服务器目录不是 git 仓库（例如通过压缩包/rsync 发布），请使用：

```bash
SKIP_GIT_PULL=1 ./scripts/deploy_linux.sh
```

若你遇到“`No model found in models/saved`”报错，直接执行：

```bash
cd /path/to/antenna
source .venv/bin/activate
python cli.py train
pm2 restart antenna-bot
```

### Web 监控面板

`server/pm2_monitor.py` 提供 Basic Auth 保护的 Web 面板，支持查看进程状态、Restart / Stop / Start 操作、查看最近日志。
日志页现已支持前端每 2 秒自动拉取 `/api/logs/<name>`，无需手动刷新即可近实时查看新日志。

```yaml
pm2_monitor:
  host:      "0.0.0.0"
  port:      9615
  username:  "admin"
  password:  "your_password"   # 必填，留空则拒绝启动
  log_lines: 200               # /logs 页面显示最近多少行
  access_log: false            # 默认关闭访问日志，减少 /api/logs 轮询噪音
```

面板：`http://<host>:9615`（每 10 秒自动刷新）。健康检查（无鉴权）：`http://<host>:9615/healthz`。

---

## 定时任务

`scripts/setup_tasks.ps1` 一键注册以下 Windows 任务计划（需管理员）：

| 任务                     | 时间                     | 功能                                      |
| ------------------------ | ------------------------ | ----------------------------------------- |
| `Antenna-Train`        | 每日 09:00               | 重训练模型                                |
| `Antenna-Scan`         | 工作日 09:15             | 全市场扫描推送（含 alt_data）             |
| `Antenna-Predict-AM`   | 工作日 09:30 起每 10 min | 上午盘中预测（动态特征列 + alt）          |
| `Antenna-Predict-PM`   | 工作日 13:00 起每 10 min | 下午盘中预测                              |
| `Antenna-Noon-Review`  | 工作日 11:32             | 午间复盘                                  |
| `Antenna-Daily-Review` | 工作日 15:32             | 收盘复盘 + 策略调整                       |
| `Antenna-WeeklyTrain`  | 周日 20:00               | P1 加权重训                               |
| `Antenna-LearnWeekly`  | 周日 02:30               | P3/P4 学习编排（特征剪枝 + 价位网格搜索） |

```powershell
powershell -File scripts/setup_tasks.ps1   # 注册所有任务
schtasks /run /tn Antenna-Scan             # 手动触发
```

---

## AI 闲聊

无法识别的消息自动转交内置 AI。角色设定为**方木木**：游戏公司后端开发，懒散口吻、简洁直接，每 5 分钟自动切换"当前状态"以保持回复多样性。

**模型调用链**（优先级从高到低）：

1. **阿里云 Qwen3**（`qwen.api_key` 非空）
2. **DeepSeek**（`deepseek.api_key` 非空，Qwen 不可用时回退）
3. **Anthropic Claude Haiku**（上述均不可用时回退）
4. 内置兜底回复

对话历史按 `chat_id` 独立保留，最多 20 轮上下文。

### 风格采集（可选）

```bash
python cli.py style --list-chats       # 列出可采集的 P2P 私聊
python cli.py style --find-user 木     # 按姓名搜索 open_id
python cli.py style --dry-run          # 演练
python cli.py style                    # 真实写入 learning/persona.txt
```

---

## 技术栈

- **数据**：akshare（A 股行情 / 财报 / alt_data）+ 新浪 HQ API（实时报价）
- **特征**：pandas / numpy / pandas_ta（27 个技术指标 + 5 个 alt 特征）
- **模型**：LightGBM 4.x（二分类）+ scikit-learn IsotonicRegression（概率校准）+ Permutation Importance（P3 特征剪枝）
- **配置**：PyYAML
- **机器人**：requests（飞书 Open Platform / 企业微信 API）
- **推送**：多通道扇出（飞书 Webhook + 企业微信自建应用，并行执行）
- **进程管理**：PM2（Node.js）+ Windows 任务计划
- **存储**：Parquet（行情缓存 + alt 缓存）+ JSONL（pred / outcome 归档）
- **测试**：pytest（402 用例，覆盖学习系统全链路）

---

## 许可

本项目为个人学习与交易辅助工具，**不构成任何投资建议**。模型输出仅供参考，使用前请理解风险并自行验证。
