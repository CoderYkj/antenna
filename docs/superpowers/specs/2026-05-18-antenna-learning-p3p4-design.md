# Antenna 学习系统 P3 特征层 + P4 价位层 设计

**Date:** 2026-05-18  
**Status:** 设计已确认，待生成实现计划  
**Owner:** 意吟  
**Base branch:** master  
**前置阶段:** P0 地基 ✅ / P1 模型层 ✅ / P2 战法层 ✅ / 横向黑名单 ✅

---

## 1. 目标与非目标

### 目标

- **P3 特征层**：接入 5 个 alt_data 特征（资金面/情绪面），并通过 Permutation Importance 动态淘汰低贡献技术指标，使模型特征集随市场自适应
- **P4 价位层**：通过网格搜索学习最优 ATR/振幅倍数，让 `suggest_dual_period_trades()` 的止盈止损位跟市场状态走

### 非目标

- 不替换 LightGBM 核心模型
- 不改动飞书指令集和卡片格式
- 不引入数据库，继续用文件系统
- 不实盘下单

---

## 2. 背景

当前 25 个技术指标等权输入，从未验证哪些真正有用；`suggest_dual_period_trades()` 止盈止损常量写死（ATR×1.5、振幅×3），从未根据历史命中情况调整。P3/P4 补上这两个反馈回路。

---

## 3. 架构方案

采用**三模块独立注册**方案，延续 P1/P2 的单职责模式：

```
data/alt_fetcher.py          ← 新增：5 个 alt_data 特征拉取
learning/feature_learner.py  ← 新增：Permutation Importance + 动态淘汰
learning/price_learner.py    ← 新增：ATR/振幅系数网格搜索
```

三模块通过 `orchestrator.MODULES` 注册，互不依赖，任一失败不影响其他。

---

## 4. 数据层：alt_fetcher

### 4.1 接口

```python
fetch_alt_features(codes: list[str], date_str: str) -> dict[str, dict]
# 返回 {code: {main_net_in_1d, main_net_in_5d, dragon_top_cnt_10d,
#              sector_heat_rank, north_hold_chg_5d}}
```

### 4.2 五个特征

| 特征名 | akshare 接口 | 含义 |
|--------|-------------|------|
| `main_net_in_1d` | `stock_individual_fund_flow` | 主力净流入 1 日（亿元标准化） |
| `main_net_in_5d` | `stock_individual_fund_flow` | 主力净流入 5 日（亿元标准化） |
| `dragon_top_cnt_10d` | `stock_lhb_detail_em` | 近 10 日龙虎榜出现次数 |
| `sector_heat_rank` | `stock_board_industry_hist_em` | 所属行业 10 日涨幅排名（归一化 0~1，越高越热） |
| `north_hold_chg_5d` | `stock_hsgt_hold_stock_em` | 北向持股 5 日变动（不覆盖返回 None） |

### 4.3 实现细节

- 5 个接口**串行**拉取，避免 akshare 频率限制，总耗时约 10-20 秒
- 每个接口独立 `try/except`，失败返回 `None`，下游 `fillna(0)`
- 全量结果缓存到 `data/cache/alt/{date_str}.parquet`（同日重复调用直接读缓存）
- 所有数值标准化到 `[-1, 1]`，统一量纲

### 4.4 集成点

`scripts/task_scan.py` 扫描前调用：

```python
from data.alt_fetcher import fetch_alt_features
alt_cache = fetch_alt_features(codes, today)  # 失败返回 {}
```

`alt_cache` 传入 `cmd_scan_bot`，经 `builder.build_features(df, alt=alt_cache.get(code, {}))` 追加到 df 最后一行（predictor 推理只用最后一行特征）。

**降级链**：alt_fetcher 整体失败 → `alt={}` → builder 只构造原有 25 个特征 → 扫描正常进行，无感。

**v1 范围**：alt_data 仅注入 `cmd_scan_bot`（全市场扫描路径）。`cmd_predict`（单股预测）和 `cmd_tactic`（战法筛选）本期不注入 alt_data，`alt={}` 传空，dropped 特征同样补 0。

---

## 5. P3：feature_learner

### 5.1 触发时机

每周日，与 `model_learner` 同批次，在 `orchestrator` 内跟随 `cli.py learn` 触发。非周日调用直接返回 `{"status": "skipped", "reason": "weekly only"}`。

### 5.2 重要性计算

1. 加载最新 LightGBM 模型 + 最近 90 日 pred/outcome 样本
2. 用 `sklearn.inspection.permutation_importance` 计算各特征重要性分数
3. 对 alt_data 新特征**额外计算 IC**（信息系数）：`|IC| > 0.02` 才纳入 `active`，否则留 `candidates`

### 5.3 淘汰逻辑

| 状态 | 条件 | 推理层行为 |
|------|------|-----------|
| `active` | 重要性 ≥ 5%，或 alt_data IC ≥ 0.02 | 正常传入模型 |
| `candidate_drop` | 连续 4 周重要性 < 5% | 仍在 `active`，仅标记（尚未淘汰） |
| `dropped` | 连续 8 周重要性 < 5% | 推理时补 0；下次重训排除 |

**注**：`candidate_drop` 阶段特征仍参与推理，提供一个观察缓冲期，避免误淘汰。

### 5.4 推理层改造

新增工具函数 `features/technical.py::get_active_feature_cols() -> list[str]`：

```python
def get_active_feature_cols() -> list[str]:
    # 读 learning/feature_weights.json["active"]
    # 文件缺失 / 字段为空 → 回退完整 FEATURE_COLS
```

改造点：
- `models/predictor.predict()`：调 `get_active_feature_cols()`，`dropped` 特征在 df 中补 0 后传入
- `models/trainer.train()` / `train_weighted()`：调 `get_active_feature_cols()`，只训练 active 特征
- `features/builder.build_features(df, alt=None)`：计算全量特征（含 alt_data），不在 builder 层裁剪

### 5.5 产物：learning/feature_weights.json

```json
{
  "version": 1,
  "updated_at": "2026-05-18T20:00:00",
  "market_state": "range",
  "active": ["ma5", "ma10", "rsi6", "main_net_in_1d", "..."],
  "candidate_drop": [{"name": "cci", "weeks_below": 3}],
  "dropped": [],
  "candidates": [{"name": "north_hold_chg_5d", "ic": 0.018, "weeks_observed": 1}],
  "importance": {"ma5": 0.12, "rsi6": 0.09, "cci": 0.03}
}
```

### 5.6 配置：learning/feature_learner.yaml

```yaml
permutation:
  n_repeats: 10          # permutation_importance 重复次数
  lookback_days: 90      # 样本窗口
  importance_threshold: 0.05  # 低于此值计入"低重要"

drop:
  candidate_weeks: 4     # 连续多少周低重要 → candidate_drop
  drop_weeks: 8          # 连续多少周低重要 → dropped

alt_data:
  ic_threshold: 0.02     # IC 达标才加入 active
  ic_lookback_days: 90
```

### 5.7 冷启动

无模型 pkl / 样本 < 30 → 跳过，`feature_weights.json` 不更新，推理继续用全量 FEATURE_COLS。

---

## 6. P4：price_learner

### 6.1 触发时机

每周日，依赖 `market_state`，与 `feature_learner` 并行（互不依赖）。

### 6.2 数据来源

最近 90 日 `pred[signal=买入]` + 对应 `outcome[hit_5d, max_drawdown_5d]`。  
需 5 日滞后（outcome 第 6 个交易日才齐），因此最新 5 个交易日的记录不参与本次学习。

### 6.3 网格搜索

**模拟逻辑**：对每组参数，统计历史样本中止盈/止损触达率：
- 若 `hit_5d >= sell_pct(params)`：触及止盈
- 若 `max_drawdown_5d <= -stop_pct(params)`：触及止损

**目标函数**：`score = stop_win_rate - stop_loss_rate`（最大化）

**搜索空间**：

| 参数 | 范围 | 步长 | 说明 |
|------|------|------|------|
| `short_atr_mult` | 1.0 ~ 2.5 | 0.1 | 短线止损 ATR 倍数 |
| `short_gain_mult` | 1.5 ~ 3.0 | 0.1 | 短线止盈振幅倍数 |
| `long_amp_mult` | 2.0 ~ 4.0 | 0.2 | 长线止盈振幅倍数 |
| `long_ma60_buffer` | 0.95 ~ 0.99 | 0.01 | 长线止损 MA60 缓冲系数 |

**分桶**：bull / bear / range 各搜一套，每桶至少 **30 个样本**，不足则回退默认值。

### 6.4 推荐路径改造

`features/analyser.suggest_dual_period_trades()` 头部增加：

```python
from learning.price_learner import load_price_params
params = load_price_params(state)  # 文件缺失 → 返回硬编码默认值，永不抛
```

其余计算逻辑不变，仅将硬编码常量替换为 `params["short_atr_mult"]` 等字段。

### 6.5 产物：learning/price_params.json

```json
{
  "version": 1,
  "updated_at": "2026-05-18T20:00:00",
  "bull":  {"short_atr_mult": 1.8, "short_gain_mult": 2.5,
            "long_amp_mult": 3.2, "long_ma60_buffer": 0.98},
  "bear":  {"short_atr_mult": 1.2, "short_gain_mult": 1.8,
            "long_amp_mult": 2.5, "long_ma60_buffer": 0.96},
  "range": {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
            "long_amp_mult": 3.0, "long_ma60_buffer": 0.97},
  "history": [{"date": "2026-05-18", "state": "range",
               "score_before": null, "score_after": 0.12}]
}
```

### 6.6 配置：learning/price_learner.yaml

```yaml
lookback_days: 90
min_samples: 30          # 每桶最少样本数

grid:
  short_atr_mult:   {min: 1.0, max: 2.5, step: 0.1}
  short_gain_mult:  {min: 1.5, max: 3.0, step: 0.1}
  long_amp_mult:    {min: 2.0, max: 4.0, step: 0.2}
  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}

defaults:
  short_atr_mult:   1.5
  short_gain_mult:  2.2
  long_amp_mult:    3.0
  long_ma60_buffer: 0.97
```

---

## 7. 编排器集成

`learning/orchestrator.MODULES` 追加：

```python
{"name": "feature_learner", "deps": ["market_state"], "schedule": "weekly"},
{"name": "price_learner",   "deps": ["market_state"], "schedule": "weekly"},
```

`check()` 追加校验 `learning/feature_weights.json` 和 `learning/price_params.json`。

非周日调用：模块内部返回 `skipped`，orchestrator 记录但不报错。

---

## 8. 完整改动文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `data/alt_fetcher.py` | 新增 | 5 个 alt_data 接口，缓存到 parquet |
| `learning/feature_learner.py` | 新增 | Permutation Importance + 淘汰逻辑 |
| `learning/feature_learner.yaml` | 新增 | 重要性阈值、淘汰周数配置 |
| `learning/price_learner.py` | 新增 | 网格搜索 + load_price_params() |
| `learning/price_learner.yaml` | 新增 | 搜索范围、默认值配置 |
| `features/technical.py` | 修改 | 新增 `get_active_feature_cols()` |
| `features/builder.py` | 修改 | 接收 `alt=` 参数，追加 alt_data 列到最后一行 |
| `features/analyser.py` | 修改 | `suggest_dual_period_trades()` 读 price_params |
| `models/predictor.py` | 修改 | 用 `get_active_feature_cols()`，dropped 补 0 |
| `models/trainer.py` | 修改 | 用 `get_active_feature_cols()` |
| `learning/orchestrator.py` | 修改 | 注册 feature_learner、price_learner |
| `scripts/task_scan.py` | 修改 | 扫描前调 alt_fetcher |

---

## 9. 测试目标

| 模块 | 最低用例数 | 关键场景 |
|------|-----------|---------|
| `alt_fetcher` | 10 | 单接口失败降级、缓存命中、全失败返回 `{}` |
| `feature_learner` | 20 | 淘汰状态机、冷启动回退、IC 门槛过滤、get_active_feature_cols 回退 |
| `price_learner` | 15 | 样本不足回退默认、grid 最优解选取、load_price_params 文件缺失 |

全套现有 316 个用例继续保持绿。

---

## 10. 学习系统阶段进度（更新后）

- ✅ P0 地基层（2026-04-28）
- ✅ P1 模型层（2026-05-09）
- ✅ P2 战法层（2026-05-11）
- ✅ 横向黑名单（2026-05-11）
- 🚧 **P3 特征层 + P4 价位层（本次，2026-05-18 设计）**
