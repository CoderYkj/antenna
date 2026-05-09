# Antenna 学习系统 Spec Review 报告

> **日期：** 2026-05-09
> **范围：** docs/superpowers/specs/ 下全部 4 份设计文档
> **目的：** 落地 P2 之前，盘点已实施部分与设计文档的偏差、未拍板决策、跨文档矛盾

---

## 1. Spec 地图

| Spec | 行数 | 时点 | 状态 | 主题 |
|------|-----|------|-----|------|
| `2026-04-01-astock-analysis-design.md` | 164 | 项目伊始 | 🟡 **过时** | v0.1.0 项目最初设计：CLI + HTML 报告，无飞书、无学习 |
| `2026-04-27-antenna-learning-system-design.md` | 556 | 学习系统总图 | 🟡 **部分过时** | P0-P4 + 横向（blacklist/alt_data/ai_reason）路线总图 |
| `2026-04-29-antenna-learning-p1-model-layer-design.md` | 530 | P1 详细 | ✅ **已实施** | isotonic 校准 + 加权重训 + 双门槛 |
| `2026-05-09-antenna-learning-p2-tactic-layer-design.md` | 581 | P2 详细 | 📐 **设计完成，待实施** | 战法阈值自适应 + ai_reason |

**总文档量**：1831 行。

---

## 2. 已实施 vs Spec 偏差（最关键，避免基于过时假设决策）

### 2.1 与 P1 spec 的偏差（4 处）

P1 spec 写于 4-29，5 个 Sprint 实施过程中出现以下偏离：

| 偏差点 | spec 原计划 | 实际实施 | 原因 |
|--------|------------|---------|------|
| `abs_threshold.initial` | 0.45 | **0.30**（抛光阶段下调） | 实测 isotonic 输出上限 0.36，0.45 太严导致买入信号 0 |
| `_collect_calibration_samples` 用最新模型重打分 | spec §5.1 要求 | **回放脚本简化为直接用 pred.rise_prob** | 回放 198 天 × 数千股票重打分不现实；正式 fit_calibrators 仍按 spec 走 |
| 验收硬指标"精准率 ≥ 35%" | spec §9 | **未达成（30.48%/全量 / 37.50%/最近 27 天）** | 实际 baseline 22% 而非假设的 33%，isotonic 输出上限 36.2% |
| `models/predictor.py` 状态 | spec 假设已有 load_model/assign_global_signals | **实际只有 37 行简单版，Sprint B 重写到 185 行** | spec 反映"设计意图"而非"代码现状" |

**影响**：spec 的核心算法 100% 落地，但**部分硬指标和默认值需要按实际数据调整**。这种"实施反哺 spec"的修订需要回写到 spec 里，否则下次启动 P3 时又会踩 staleness 坑。

### 2.2 与早期总体 spec（4-27）的偏差（3 处）

| 偏差点 | 早期 spec | 实际 | 备注 |
|--------|---------|-----|------|
| 学习成果文件位置 | spec 列出 `learning/feedback/` `learning/history/` `learning/alerts.jsonl` | ✅ 完全一致 | feedback_io / alerts P0 已实现 |
| 学习产物列表 | spec 列出 5 个 json | 实际产生 4 个（无 blacklist.json） | blacklist 模块尚未启动 |
| `tactic_params.json` 路径 | spec 指 `learning/tactic_params.json` | P2 spec 沿用 | 一致 |

**影响**：早期总图与实际**基本一致**，可以继续作为路线图主参考。

### 2.3 与最早 spec（4-01）的偏差（重大）

`2026-04-01-astock-analysis-design.md` 是 v0.1.0 个人 CLI 工具设计，**与当前系统几乎无关**：
- ❌ 无飞书机器人（实际有）
- ❌ 无学习系统（实际 P0+P1 已上线）
- ❌ 无四大战法（实际有）
- ❌ 无 Walk-Forward 回测（实际有）

**建议**：归档到 `docs/superpowers/specs/archive/`，保留作为项目历史，但不作为决策参考。

---

## 3. P2 Spec 决策点 review

P2 spec 已固化的关键决策（spec §1.3），逐项审视：

### 3.1 已拍板（无需调整）

| 决策 | 选择 | 评估 |
|------|------|------|
| 阈值调整算法 | 固定步长 ±1（±2 触底回弹） | ✅ 与 P1 abs_threshold 风格一致；保守稳定 |
| 代码落位 | `tactic_learner.py` + yaml | ✅ 对齐 P1 风格 |
| 冷启动 | 单桶 <30 → 维持上版（不重置） | ✅ 比 P0 阈值重置更保守，吸取 P1 经验 |
| ai_reason 缓存 | (code, state, prob_bucket=0.05) 24h | ✅ 成本可控（≤0.5 元/月） |
| 共振权重 boost_per_tactic | 0.02 上限 | ✅ 全市场 rank 排序后会被压缩，安全 |

### 3.2 待用户拍板（spec 已选默认但有讨论空间）

#### 决策点 D1：交付范围

| 选项 | spec 当前 | 替代 | 权衡 |
|------|----------|-----|-----|
| 完整 P2 = tactic_learner + ai_reason | ✅ 当前 | 仅 tactic_learner，ai_reason 后做 | 完整版 4.5 天 vs 仅前者 ~2 天；ai_reason 成本低且独立性强，建议保留完整版 |

**风险**：ai_reason 涉及外部 LLM，单独 Sprint C（1.5 天）出问题不影响 Sprint A/B 的 tactic_learner。可以进度到 Sprint B 时再决定 C 是否实施。

#### 决策点 D2：触发频率

| 选项 | spec 当前 | 替代 | 权衡 |
|------|----------|-----|-----|
| 跟 model_learner 同步（每次"学习"指令 + 周日 15:30） | ✅ 当前 | 仅周三/周日 15:30 | 同步触发：每天有人发"学习"会反复重算，浪费；但实测 90 日窗口稳定，每日重算结果几乎无变化 |

**建议**：spec 当前合理。但 `tactic_learner.run()` 内可以加一个 `last_calibrated` 检查——24h 内已跑过则跳过实际计算，仅返回上次结果。这是小优化，可放进 Sprint A。

#### 决策点 D3：战法权重是否启用

| 选项 | spec 当前 | 替代 | 权衡 |
|------|----------|-----|-----|
| 共振股 rise_prob 加权 | ✅ 当前 | 仅展示标签，不加权 | 加权可能让 rise_prob 偏离 calibrator 的真实命中率（与 P1 设计哲学冲突）|

**风险点**：P1 spec 明确"prob_cal 由 calibrator 决定"，rise_prob 加权会污染 prob_raw 的全市场排名。spec §3.3 已写"只对 rise_prob_raw 加权，不动 prob_cal"——但 prob_cal 来自 prob_raw，加权后 prob_raw 提升 → prob_cal 也跟着变。

**建议**：要么放弃此项（仅显示标签），要么改为"调整 rank_pct 而非 rise_prob"——共振股向前提名次，但模型概率不动。这是**需要用户拍板**的决策点。

#### 决策点 D4：bear 桶永远不足怎么办

| 选项 | spec 当前 | 替代 | 权衡 |
|------|----------|-----|-----|
| 维持 yaml.defaults | ✅ 当前 | global fallback 用全部 12 桶 | bear 没数据，永远学不到 bear 专用阈值 |

**现实**：截至今天，learning/market_state.json.history 中 bear 状态 **0 天**（205 天全是 bull/range）。P2 上线后，bear 桶可能数年都拿不到样本。

**建议**：spec §1.3 选择"维持 defaults"是合理的（否则 fallback 后变成 range 模型），保留即可。但 yaml.defaults 的 bear 应该**手工调整为更保守值**（参考行业经验，bear 市场提高 ROE 门槛、收紧 debt_ratio）。

### 3.3 spec 中可能有 bug 的细节（小修）

| 位置 | 问题 | 建议 |
|------|------|------|
| spec §3.4 history 日期格式 | `"2026-05-09"` 但 P1 实际写的是 `2026-05-08T19:10:39`（ISO） | 对齐为 ISO 时间戳 |
| spec §4.5 价格估算 | "Qwen3-8B 输入 ¥0.5/百万 + 输出 ¥1.5/百万" 数字未核实 | 实施时核实最新价格 |
| spec §3.2 收严方向 | 各战法"全部 +step"统一收严 | 个别阈值（如 `debt_ratio_max`）的"收严"方向应该是 **-step**（更低=更严），不是 +step | 需要逐阈值标注方向 |

---

## 4. P3/P4 spec 成熟度评估

P2 spec §9.2-9.4 给出 P3/P4/黑名单"预览"，但只有 4-7 行 bullet。早期 spec §6.3-6.4 + §7.2 写得更详细但有些已过时。

### 4.1 P3 特征层（成熟度：🟡 中）

| 维度 | 早期 spec §6.3 | P2 spec §9.2 | 缺口 |
|------|--------------|--------------|------|
| 算法 | Permutation Importance + 8 周淘汰 + IC 候选 | 同 | 算法成熟 |
| alt_data 数据源 | "主力净流入 / 龙虎榜 / 北向资金" | "主力净流入 / 龙虎榜 / 北向资金" | ✅ 一致 |
| 实施细节 | `feature_weights.json` schema 已给 | 简略 | ⚠️ 需要详细 spec |
| 验收硬指标 | "AUC 不下降" | "特征数 25 → 18-22；AUC 不下降" | ⚠️ 缺 walk-forward 验证方案 |
| Sprint 切分 | — | — | ⚠️ 需要 |

**结论**：P3 启动前需要写 ~400 行详细 spec。早期 spec §6.3 可作起点。

### 4.2 P4 价位层（成熟度：🟢 较高）

| 维度 | 早期 spec §6.4 | P2 spec §9.3 | 缺口 |
|------|--------------|--------------|------|
| 算法 | 网格搜索 ATR×{1.0~2.5}/振幅×{2.0~4.0} | 同 | 算法成熟 |
| 数据 | hit_5d / max_drawdown_5d | 同 | ✅ 数据已就绪（P0 回填） |
| 评估指标 | `(触止盈率 - 触止损率) × 平均持仓收益` | 同 | ✅ 公式明确 |
| 验收硬指标 | — | "短线 5 日触止盈率 ≥ +5pt" | ✅ 完整 |
| 实施细节 | `price_params.json` schema 已给 | 简略 | ⚠️ 需要 ~250 行 spec |

**结论**：P4 是 P2/P3 之外**风险最低、回报明确**的项目。可以在 P2 完成后立即推。

### 4.3 横向黑名单（成熟度：🟢 较高）

| 维度 | 早期 spec §7.2 | P2 spec §9.4 | 缺口 |
|------|--------------|--------------|------|
| 触发条件 | 同 (code, state) 连 3 次 buy+miss → 拉黑 30 天 | 同 | ✅ 一致 |
| 数据结构 | `blacklist.json` 已给 | 略 | ⚠️ 需要详细 spec |
| 推荐路径 | `cmd_scan_bot.assign_global_signals` 之前过滤 | 同 | ✅ 集成点明确 |
| 验收 | "黑名单股后续 30 日精准率 ≤ 全市场基线" | 同 | ✅ 验证逻辑合理 |

**结论**：黑名单是**最简单**的子模块（可能 ~200 LOC + 单测）。可以与 P2 并行甚至插队完成。

### 4.4 alt_data（成熟度：🟡 中低）

P2 spec 没有专门讨论 alt_data。早期 spec §7.3 有提，但实施细节不足：
- 数据源：akshare 接口名（需要核实是否仍可用）
- 存储格式：`data/cache/alt/*.parquet`
- 与 P3 feature_learner 的集成方式

**结论**：alt_data 是 P3 的前置依赖。如果直接做 P3，需要先确定 alt_data 的数据接入路径。

---

## 5. 跨 spec 矛盾盘点

### 5.1 `tactic_params.json` 路径

| 来源 | 路径 |
|------|-----|
| 早期总体 spec §6.2 产物 | `learning/tactic_params.json` |
| P2 spec §3.4 产物 | `learning/tactic_params.json` |
| 当前实施 | （未实施）|

✅ **一致**

### 5.2 ai_reason 落位

| 来源 | 路径 |
|------|-----|
| 早期总体 spec §7.4 | `ai_reason.py`（learning/ 还是 server/?未明） |
| P2 spec 决策 D2 | **`server/ai_reason.py`** |

✅ **P2 spec 已澄清**

### 5.3 触发节拍

| 模块 | 早期 spec | P2 spec | 实际部署 |
|------|---------|---------|---------|
| market_state | 每日 15:30 | — | ✅ Antenna-Daily-Review 15:32 + 编排器每次"学习"|
| model_learner | 校准日级 + 重训周级 | — | ✅ 编排器每次"学习" + Antenna-WeeklyTrain 周日 20:00 |
| tactic_learner | 每周三/周日 15:30 | 跟 model_learner 同步 | （未部署）|

⚠️ **轻微矛盾**：P2 spec 选择"跟 model_learner 同步"覆盖了早期 spec 的"周三/周日 15:30"。这是**有意调整**（spec §1.3 决策点 4），但建议在 P2 spec 显式注释"覆盖 4-27 spec 的频率约定"。

### 5.4 学习产物文件清单

早期总体 spec 列出 5 个 json，当前 + P2 后实际产生：

| 文件 | 早期 spec | P0/P1 实际 | P2 后 | 状态 |
|------|---------|----------|-------|------|
| `market_state.json` | ✅ | ✅ | ✅ | 一致 |
| `strategy.json` | ✅ | ✅ | ✅ | 一致（早期 spec 没列，但 P0 已有） |
| `model_learner.json` | ✅ | ✅ | ✅ | 一致 |
| `tactic_params.json` | ✅ | ❌ | ✅ | P2 后产生 |
| `feature_weights.json` | ✅ | ❌ | ❌ | P3 后产生 |
| `price_params.json` | ✅ | ❌ | ❌ | P4 后产生 |
| `blacklist.json` | ✅ | ❌ | ❌ | 横向后产生 |
| **`learning/alerts.jsonl`** | ❌（未列） | ✅ | ✅ | spec 漏写 |
| `learning/feedback/*.json` | ✅ | ✅ | ✅ | 一致 |
| `learning/history/*.json` | ✅ | ✅ | ✅ | 一致 |

---

## 6. 建议的 next-step

### 6.1 立即可做（不阻塞决策）

| 动作 | 工作量 | 价值 |
|------|------|-----|
| 归档 `2026-04-01-astock-analysis-design.md` 到 `archive/` | 1 分钟 | 减少认知负担，避免新人误用作参考 |
| P1 spec 加 "实施备忘"附录，记录 §2.1 的 4 处偏差 | 15 分钟 | 防止下次基于过时假设决策 |
| 早期总体 spec 加"⚠️ 部分章节由 P1/P2 spec 覆盖"提示 | 5 分钟 | 同上 |

### 6.2 需要用户拍板的决策

| ID | 决策点 | 默认 | 替代 |
|----|--------|-----|-----|
| **D3** | P2 共振权重是否启用 | 启用（boost rise_prob_raw） | 仅展示标签 / 改为调整 rank_pct |
| **D4** | bear 桶 yaml.defaults 是否手工调严 | 维持当前 | 提高 ROE / 收紧 debt_ratio |
| **D5** | P2 spec §3.2 阈值收严方向 | 全部 +step | 按阈值类型分（min 类 +，max 类 -）|

### 6.3 需要补充设计的工作（非紧急）

| 项目 | 优先级 | 预估 |
|------|------|-----|
| P3 详细 spec | 中 | ~400 行，~半天 |
| P4 详细 spec | 低 | ~250 行，~半天 |
| 黑名单详细 spec | 中（最简单） | ~150 行，~2 小时 |
| alt_data 接入设计 | P3 启动前必需 | ~200 行 |

### 6.4 推荐的实施路径

```
[ 当前状态 ]
P0 ✅  P1 ✅
P2 📐 spec 完成,待实施(~4.5 天)
P3/P4/黑名单 ⏳ 仅有大纲

[ 推荐顺序 ]
1. 解决 §6.2 三个决策点(D3/D4/D5)        ~10 分钟
2. 修订 P2 spec 反映决策                   ~30 分钟
3. 跑黑名单 spec(最简单,提供风控价值)    ~2 小时设计 + 1 天实施
4. 跑 P2 完整 5 个 Sprint                  ~4.5 天
5. 跑 P4 spec + 实施(数据已就绪,风险低)  ~半天 + 3 天
6. 跑 P3 spec + alt_data + 实施           ~1 天 + 5 天
合计: ~14 天分散执行
```

或者**更精益**的路径：

```
1. 决策 D3/D4/D5 + P2 spec 修订            10-30 min
2. 跑 P2 Sprint A(tactic_learner 核心)     1 天
3. 上线观察一周                            1 周
4. 视效果决定 P2 Sprint B/C 或转 P4        —
```

---

## 7. 一句话结论

**4 份 spec 总体一致，P0/P1 已实施落地，P2 spec 完整但有 3 个决策点（D3/D4/D5）需要拍板，P3/P4/黑名单只有大纲需要补 spec。** 最快速 unblock 的动作：用户对 D3/D4/D5 给出意见 → 我修订 P2 spec → 启动 Sprint A 或转向更轻量的黑名单。

---

**end of review**
