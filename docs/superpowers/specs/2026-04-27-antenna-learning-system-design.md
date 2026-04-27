# Antenna 推荐股深度分析与自学习系统设计

**Date:** 2026-04-27
**Status:** 设计已确认,待生成实现计划
**Owner:** 意吟
**Base branch:** master

---

## 1. 目标与非目标

### 目标

在现有 Antenna A 股量化系统基础上,为"推荐股"新增 **4 层自学习反馈**和 **4 项强化分析能力**,使系统能够根据历史推荐的真实表现自动调整参数、筛选特征、识别失效策略,并提供更有深度的分析内容。

### 非目标

- **不替换** LightGBM 核心模型(保留)
- **不改动**飞书机器人交互流程与指令集
- **不改动**现有战法体系(保留 4 战法:价值/成长/龙头/逆向)
- **不做**实盘下单、资金管理、组合优化
- **不做**股票回测(Walk-Forward 已有,本次只做"学习反馈的回放")
- **不引入数据库**,继续用 JSON/JSONL/parquet 文件系统存储

---

## 2. 背景与动机

**当前自学习状态(strategy.json @ 2026-04-27)**

- 累计 10 个月 `pred_*.jsonl` + `outcome_*.jsonl`(2025-07 至 2026-04)
- 近 7 日买入精准率 **31.7%**,30 日 **32.6%**,目标 55% — **长期严重不达标**
- 现有自学习只调一个标量 `buy_top_pct`,反复触底→自救重置→再触底,**反馈回路实际失效**
- LightGBM 模型固定参数、25 个特征等权、4 个战法阈值硬编码、双周期止盈止损常量写死 — **全无学习反馈**

**痛点锁定:学习反馈维度过于单一,数据在但没有被真正用起来。**

---

## 3. 决策摘要

用户确认的范围(brainstorming 决策):

| 维度 | 决策 |
|------|------|
| 核心痛点 | 学习反馈太单一 |
| 学习层级 | 模型层 + 战法层 + 特征层 + 价位层 **4 层全做** |
| 执行节奏 | 分 4 阶段,**P1 模型层首先** |
| 优化目标 | **维持买入精准率**(内部辅以涨幅分档加权) |
| 数据覆盖 | 扫描 + 战法 + 单股 **全覆盖** |
| 强化分析 | **全部 4 项纳入**(新数据源/AI 理由/市场状态/黑名单) |
| 架构风格 | 方案 B:**分层管线 + 瘦编排器** |
| 存储 | 文件系统(JSON/JSONL/parquet),不引入数据库 |
| 学习频率 | P1 校准日级 + 重训周级;P2 每周 2 次;P3/P4 每周 |
| 冷启动 | 样本 <30 全部回退默认 |
| 5 日滞后 | 接受(outcome 要等第 6 天才齐) |
| 市场状态分档 | 3 档:`bull / bear / range` |

---

## 4. 总体架构

```
             ┌──────────── 每日盘中 ─────────────┐
             │                                   │
┌─ cmd_scan_bot ─┬─ cmd_tactic ─┬─ cmd_predict ──┘
│                │              │
│  写入 pred_*.jsonl (带 scene 标签)
│                │              │
└────────────────┴──────────────┴─── 15:30 收盘后 ──┐
                                                    ▼
                                   ┌─ task_daily_review.py ──────────┐
                                   │   ① 回填 outcome_*.jsonl        │
                                   │   ② 调 cli.py learn (编排器)    │
                                   └──────────────┬──────────────────┘
                                                  ▼
       ┌──────────────────────────────────────────────────────────────┐
       │                      编排器 cli.py learn                      │
       │                                                               │
       │  P1 model_learner   ─→ models/saved/model_calibrated.pkl     │
       │  P2 tactic_learner  ─→ learning/tactic_params.json           │
       │  P3 feature_learner ─→ learning/feature_weights.json         │
       │  P4 price_learner   ─→ learning/price_params.json            │
       │                                                               │
       │  横向:                                                        │
       │    market_state ─→ learning/market_state.json (每日打标)     │
       │    blacklist    ─→ learning/blacklist.json                   │
       │    alt_data     ─→ data/cache/alt/*.parquet (资金/情绪)      │
       │    ai_reason    ─→ 推荐推送时实时调用,不参与学习              │
       │                                                               │
       │  每个子模块独立写 feedback/<module>_YYYY-MM-DD.json 归因日志  │
       └──────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                    次日推荐路径读取最新学习成果,生成更精准推荐
```

**核心约定**
- **读写分离**:推荐路径只读学习成果,学习路径只写,不互相阻塞
- **离线学习**:学习只在盘后做,盘中推荐零额外延迟
- **可失败**:任何子模块崩溃只跳过自己,不影响其他;文件缺失时推荐路径回退默认值
- **可回滚**:每份学习成果保留 7 份历史快照,可人工 revert

### 新文件结构

```
learning/
├── tracker.py              # 已有,扩展为支持 scan/tactic/predict 三场景
├── optimizer.py            # 已有,保留 buy_top_pct 调节
├── backtest_history.py     # 已有,不变
├── model_learner.py        # ★新增 P1 概率校准 + 错样本加权重训
├── tactic_learner.py       # ★新增 P2 战法阈值/权重自适应
├── feature_learner.py      # ★新增 P3 指标贡献度 + 动态筛选
├── price_learner.py        # ★新增 P4 价位 ATR/振幅系数反馈
├── market_state.py         # ★新增 大盘状态打标 + 多模型路由
├── blacklist.py            # ★新增 风控黑名单自学习
├── alt_data.py             # ★新增 资金面/情绪面数据接入
├── ai_reason.py            # ★新增 LLM 深度理由
├── feedback/               # ★新增 每模块每日归因日志
│   └── <module>_YYYY-MM-DD.json
├── history/                # ★新增 学习产物历史快照(保留 7 份)
│   └── <file>_YYYYMMDD.json
└── alerts.jsonl            # ★新增 告警流水
```

---

## 5. 接口协议与核心数据结构

### 5.1 预测快照 `pred_YYYY-MM-DD.jsonl`(扩展)

```python
{
  "ts":         "2026-04-27T10:15:32",
  "scene":      "scan" | "tactic:value" | "tactic:growth" | "tactic:leader"
              | "tactic:contra" | "predict",
  "code":       "600519",
  "name":       "贵州茅台",
  "signal":     "买入" | "观望" | "回避",
  "rise_prob":  0.72,
  "confidence": "高",
  "rank_pct":   0.035,
  "market_state": "bull" | "bear" | "range",
  "tactic_hits": ["value", "leader"],
  "features":    {...25 个现有指标},
  "dual_trade":  {...双周期价位快照},
  "alt":         {"main_net_in_1d": 0.8, "dragon_top_cnt_10d": 1, ...},
  "watchlist":   false,
}
```

**去重规则**:同一 code 同一天可能出现 3 条(3 scene 各 1 条),**不去重**;学习时按 scene 分桶统计。

### 5.2 实际结果 `outcome_YYYY-MM-DD.jsonl`(扩展)

```python
{
  "code":            "600519",
  "actual_pct":      2.8,                      # 次日涨幅
  "hit_tier":        "miss" | "weak" | "good" | "great",
                                               # <1% | 1-2% | 2-5% | >5%
  "hit_5d":          3.5,                      # 5 日累计涨幅(第 6 个交易日回填)
  "max_drawdown_5d": -1.2,                     # 5 日最大回撤(第 6 个交易日回填)
}
```

**5 日滞后**:`hit_5d` 和 `max_drawdown_5d` 需等到第 6 个交易日才能回填;价位学习因此滞后 5 天,已接受。

### 5.3 场景标签 `scene` 取值约定

| 值 | 含义 |
|----|------|
| `scan` | 全市场扫描推荐 |
| `tactic:value` | 价值投资战法筛选 |
| `tactic:growth` | 成长股战法筛选 |
| `tactic:leader` | 行业龙头战法筛选 |
| `tactic:contra` | 逆向战法筛选 |
| `predict` | 单股预测(含自选股,通过 `watchlist` 字段再区分) |

### 5.4 市场状态标签 `market_state`

| 状态 | 判定条件(基于沪深 300) |
|------|-------------------------|
| `bull` | `close > ma60` 且近 60 日累计涨幅 > +5%,连续 3 日触发 |
| `bear` | `close < ma60` 且近 60 日累计跌幅 < -5%,连续 3 日触发 |
| `range` | 其他;或 20 日 ATR/close > 2.5% 强制进入 |

每日 15:30 由 `market_state.py` 计算后写入 `learning/market_state.json`。

### 5.5 学习成果统一 schema

```python
{
  "version":      3,
  "updated_at":   "2026-04-28T15:35:00",
  "market_state": {...state-specific params},   # 每种市场状态一套参数
  "global":       {...fallback defaults},
  "history":      [{date, key_metric, change}, ...last 30],
  "audit": {
    "samples_used":     1240,
    "accuracy_before":  0.32,
    "accuracy_after":   0.41
  }
}
```

---

## 6. 核心学习子模块(P1 → P4)

### 6.1 P1 `model_learner.py` — 概率校准 + 错样本加权重训

**痛点**:`rise_prob=0.7` 实际只有 32% 命中,模型高估;且所有错样本被等权参与再训练。

**算法**
1. **isotonic 校准**:读最近 90 日 `(pred.rise_prob, outcome.hit)` 拟合单调映射 `prob_raw → prob_cal`,三个市场状态各一份校准器。
2. **错样本加权重训**(每周日晚重训):
   - `signal=买入 且 hit=miss` → 权重 **×2.0**(重点学这些错)
   - `signal=买入 且 hit=great` → 权重 **×1.5**(强化这些对)
   - `hit=good/weak` → 权重 ×1.0
   - `signal=观望/回避` → 权重 ×0.8
   - 权重系数写成可配置常量(`learning/model_learner.yaml`),不硬编码
3. **产物**
   - `models/saved/model_YYYYMMDD.pkl`
   - `models/saved/calibrator_{bull,bear,range}.pkl`

**推荐路径改造**:`predictor.predict()` 内,按当前 `market_state` 选对应 calibrator:
```python
rise_prob = calibrator[market_state].transform([rise_prob_raw])[0]
```

**触发**
- 日级:15:30 只重拟合校准器(<1 分钟)
- 周级:每周日 20:00 全量重训(约 10-15 分钟,CPU)

---

### 6.2 P2 `tactic_learner.py` — 四战法阈值/权重自适应

**痛点**:4 战法阈值硬编码(如 `roe>8, debt_ratio<50, total_score≥2`),从未根据效果调整。

**算法**
1. 按 `scene="tactic:xxx"` 分 4 桶,统计每战法最近 90 日买入精准率
2. **阈值微调**(每周 2 次,周三/周日):
   - 价值战法精准率 <30% → 收严 `roe_min` 步长 +1;精准率 >60% → 放宽 `roe_min` 步长 -1
   - 每次调整 **≤1 个单位**,防震荡
   - 阈值锁定范围(roe_min ∈ [5, 15],防止调飞)
3. **权重学习**(共振场景):多战法命中股的 `rise_prob` 加权提升,每战法权重 = 近 90 日精准率归一化
4. **产物**:`learning/tactic_params.json`
   ```json
   {
     "bull":  {"value":  {"roe_min": 8, "debt_ratio_max": 50, "weight": 0.35, "acc_90d": 0.42},
               "growth": {...}, "leader": {...}, "contra": {...}},
     "bear":  {...},
     "range": {...}
   }
   ```

**推荐路径改造**:`predict_cmd._enrich_tactic_scores()` 读 `tactic_params.json[current_state]`,替代硬编码阈值。

**触发**:每周三/周日 15:30。

---

### 6.3 P3 `feature_learner.py` — 指标贡献度 + 动态筛选

**痛点**:25 个技术指标等权输入,不知道哪些真有用。

**算法**
1. **Permutation Importance**:每周用最新模型对最近 90 日样本跑打乱特征测试,得每个特征重要性分数
2. **衰减淘汰**:连续 4 周重要性 <5% 标记"候选淘汰",继续 4 周标记"淘汰"→ 下次训练 drop
3. **候选新特征池**:`alt_data` 提供的 5 个新特征每周末扫一次 IC(information coefficient),满足 |IC| > 0.02 自动加入训练
4. **产物**:`learning/feature_weights.json`
   ```json
   {
     "version":    3,
     "active":     [...21 个激活特征],
     "dropped":    [{"name": "cci", "dropped_at": "2026-03-15", "reason": "8 周重要性 <5%"}],
     "candidates": [{"name": "main_net_in_1d", "ic_rank": 0.034, "weeks_observed": 3}],
     "importance": {...特征 → 分数}
   }
   ```

**推荐路径改造**:
- `features/builder.py` 和 `models/trainer.py` 读 `active` 列表,替代硬编码 `FEATURE_COLS`
- 提供 `get_active_feature_cols() → list[str]` 工具函数

**触发**:每周日,与 P1 全量重训绑定。

---

### 6.4 P4 `price_learner.py` — 双周期买卖价位反馈

**痛点**:`suggest_dual_period_trades()` 用固定常量(ATR×1.5、振幅×3)算止盈止损,无人验证合理性。

**算法**
1. 对过去 90 日每条买入信号,拿 outcome 的 `hit_5d` / `max_drawdown_5d` 回测:
   - **短线**:买入后 5 日内最高价是否触达 `sell_price`?最低价是否触达 `stop_price`?
   - 统计 `{触及止盈率, 触及止损率, 平均持有收益}`
2. **网格搜索**:对 ATR 倍数从 1.0 到 2.5 步 0.1、振幅倍数从 2.0 到 4.0 步 0.2,选最大化 `触止盈率 - 触止损率` 的组合
3. 分市场状态各学一套
4. **产物**:`learning/price_params.json`
   ```json
   {
     "bull":  {"short_atr_mult": 1.8, "short_gain_mult": 2.5,
               "long_amp_mult":   3.2, "long_ma60_buffer": 0.98, ...},
     "bear":  {"short_atr_mult": 1.2, "short_gain_mult": 1.8, ...},
     "range": {...}
   }
   ```

**推荐路径改造**:`features/analyser.suggest_dual_period_trades()` 读 `price_params.json[state]`,替代硬编码常量。

**触发**:每周日 15:30(数据等齐后 5 天)。

---

## 7. 横向强化分析模块

### 7.1 `market_state.py` — 大盘状态打标

所有子模块的状态依赖,task_daily_review.py 链路首个任务。

**数据源**:`akshare.stock_zh_index_daily`(沪深 300 / 创业板指 / 上证 50)

**算法**:见 §5.4

**产物**:`learning/market_state.json`
```json
{
  "current":  "range",
  "since":    "2026-04-18",
  "hs300":    {"close": 3825.4, "ma60": 3810.2, "ret_60d": 0.023, "atr_pct": 0.019},
  "pending":  null,                      // 若处于切换中(未满 3 日)
  "history":  [{"date":"2026-04-27","state":"range"}, ...last 90]
}
```

### 7.2 `blacklist.py` — 风控黑名单自学习

**算法**
1. 扫过去 60 日 (pred, outcome),按 code 聚合:
   - 被推≥3 次,平均 `actual_pct` < -2% → **临时黑名单**(30 天)
   - 连续 5 日跌幅 >10% → 临时黑名单(10 天)
   - ST / 退市风险标签(akshare) → **条件黑名单**(每周扫新)
2. **追涨黑名单**:已连续涨 >30% 的票,短线推荐排除、长线可留(按 scene 分级应用)
3. **自动撤销**:黑名单期满自动移出;若回归上升趋势(站上 ma60 且近 10 日涨 >3%)提前撤销
4. **无永久类**:所有类别定期复核,避免误伤

**产物**:`learning/blacklist.json`
```json
{
  "temp": [{"code":"300xxx","reason":"连续错推3次均跌","until":"2026-05-27"}],
  "st_risk": [{"code":"600xxx","label":"ST","refreshed":"2026-04-27"}],
  "chase_top": [{"code":"688xxx","reason":"30日+42%","until":"2026-05-10"}]
}
```

**推荐路径改造**:`cmd_scan_bot` 在 `assign_global_signals` 之前过滤黑名单。
`cmd_tactic` 按 scope 过滤(短线场景严过滤,长线场景松过滤)。

**触发**:每日 15:30。

### 7.3 `alt_data.py` — 资金面 + 情绪面数据接入

| 特征 | 数据源 | 更新 |
|------|--------|------|
| `main_net_in_1d` 主力净流入 1 日 | `stock_individual_fund_flow` | 每日 |
| `main_net_in_5d` 主力净流入 5 日 | 同上聚合 | 每日 |
| `dragon_top_cnt_10d` 10 日龙虎榜次数 | `stock_lhb_detail_em` | 每日 |
| `sector_heat_rank` 所属行业 10 日涨幅排名 | `stock_board_industry_hist_em` | 每日 |
| `north_hold_chg_5d` 北向持股 5 日变动 | `stock_hsgt_hold_stock_em` | 每日 |

**处理**
- 每日 09:00 前(task_alt_data.py)批量拉取,缓存到 `data/cache/alt/{feature}_{date}.parquet`
- 标准化到 `[-1, 1]`,失败返 `None`,下游 `fillna(0)`
- 只作特征输入,不直接决策(由 P3 决定权重)

**产物**:`data/cache/alt/*.parquet`;预测时 `builder.build_features()` 自动合并。

**触发**:每日盘前 09:00。

### 7.4 `ai_reason.py` — LLM 深度推荐理由

**触发场景**:仅 scan_bot / tactic 推送的 **Top-N** 每股调一次(N=10,每日约 30 次调用)。

**输入 prompt 材料**
- 模型预测(rise_prob、signal、全市场排名)
- 技术指标快照
- 财务评分 `fin_res.summary`
- 命中战法 + 战法理由
- 资金/情绪快照(alt_data)
- 最近 3 条财联社新闻
- 当前 market_state + 近 30 日精准率

**输出 schema(强制 JSON)**
```json
{
  "bull_reasons": ["...", "...", "..."],
  "risks":        ["...", "..."],
  "verdict":      "稳健加仓" | "少量试仓" | "观望" | "规避",
  "confidence":   "高" | "中" | "低"
}
```

**路由**:Qwen3(默认) → 失败降级 Claude Haiku → 全失败回退现有 `build_commentary`。

**成本控制**
- 并发上限 3,每股 prompt < 1K tokens
- 缓存:同一股 + 同一 market_state + 同一 rise_prob 档位(四舍五入 0.05) 24 小时内复用
- 预估:30 股 × 1K token × Qwen3 ≈ ¥0.15/天

**产物**:不落盘,直接进飞书卡片 `elements`。不参与学习循环。

---

## 8. 错误处理与降级策略

**总则:任一学习子模块失败,推荐流程仍能跑,只是回退到默认行为。**

### 8.1 编排器层

```python
for module in [market_state, alt_data, model, tactic, feature, price, blacklist]:
    try:
        result = module.run()
        _write_feedback_log(module.name, result)
    except Exception as e:
        _log_error(module.name, e)
        _send_feishu_alert(f"⚠️ {module.name} 学习失败: {e}")
        _append_alert_jsonl(module.name, e)
        continue
```

### 8.2 推荐路径读取层

每个学习产物都有 `_load_xxx_params()` 的 **safe 读取 + 默认值** 模式:

```python
def load_tactic_params(state: str) -> dict:
    try:
        with open("learning/tactic_params.json") as f:
            data = json.load(f)
        return data.get(state) or data["global"] or _DEFAULT_TACTIC_PARAMS
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return _DEFAULT_TACTIC_PARAMS
```

### 8.3 降级矩阵

| 失败模块 | 影响 | 降级行为 |
|---------|------|---------|
| `market_state` | 无法按状态分桶 | 全部当 `range`,用 `global` 参数 |
| `alt_data` | 5 个新特征缺失 | `fillna(0)`,模型略偏但不崩 |
| `model_learner` 校准失败 | 概率不校准 | 用原始 `rise_prob` |
| `tactic_learner` | 阈值不更新 | 用现有硬编码 |
| `feature_learner` | 特征不筛选 | 保留所有 25+N 特征 |
| `price_learner` | 价位系数不更新 | 用现有 ATR×1.5/振幅×3 |
| `blacklist` | 不过滤 | 推荐照常(风险提示依靠 AI 理由) |
| `ai_reason` | 无 AI 理由 | 回退现有 `build_commentary` |

### 8.4 文件完整性

- 写入用"临时文件 + `os.replace`"原子写入(复用 `optimizer.save_strategy()` 模式)
- 每份学习产物保留最近 7 份:`learning/history/<file>_YYYYMMDD.json`
- 启动自检:`cli.py learn --check` 验证所有 JSON 可解析、版本号匹配

### 8.5 告警

- 任一子模块异常 → 飞书主动推送 + 写入 `learning/alerts.jsonl` + 打印到 `logs/learning_YYYYMMDD.log`
- 连续 3 日精准率回落 >10% → 推送"可能过拟合"告警,建议手动 revert
- 告警文件永不自动清理(人工复盘用)

---

## 9. 测试策略

### 9.1 单元测试(pytest)

- 每个学习模块的纯函数:`_calibrate_prob()`, `_compute_tactic_params()`, `_rank_feature_importance()`, `_backtest_price_grid()`
- Fixtures:`tests/fixtures/sample_preds.jsonl`、`sample_outcomes.jsonl` 造 30-50 条假数据
- 覆盖率目标:4 个 learner + market_state + blacklist ≥ **80%**
- `alt_data` 和 `ai_reason` 外部 API mock(`responses` / `unittest.mock`)

### 9.2 历史回放测试(最关键)

用项目已有 2025-07~2026-04 **10 个月真实数据**模拟"滚动学习":

```
D=2025-07-01 起,每日模拟:
  1. 读 pred_D.jsonl (当时推荐)
  2. 读 outcome_D.jsonl (当时结果)
  3. 调 learn(D) 更新所有学习成果
  4. 下一日 D+1 用更新后参数重新计算 signal
  5. 记录"校准后精准率" vs "原始精准率"
```

**验收门槛**:
- P1 模型层后:滚动 30 日精准率 ≥ 基线 +3pt
- P2 战法层后:战法子场景精准率 ≥ 基线 +3pt(累计 ≥ +5pt)
- P3 特征层后:累计精准率 ≥ 基线 +8pt;新特征 Importance > 5%
- P4 价位层后:短线止盈触发率 ≥ 40%,止损触发率 ≤ 30%

**未达验收不上线该阶段,回画板重做。**

脚本:`scripts/replay_learn.py`,输出对比曲线 CSV + 飞书卡片。

### 9.3 集成测试

- `tests/test_learn_pipeline.py`:完整跑一次 `cli.py learn`,确认所有输出文件存在且 schema 正确
- `tests/test_recommend_path.py`:mock 学习文件,跑 `cmd_scan_bot` 和 `cmd_tactic`,确认读取正确

### 9.4 冒烟测试

`scripts/smoke_test.py` 扩展:调 `/预测`、`/扫描`、`/战法 价值`,确认学习文件被读取、无报错。

---

## 10. 分阶段落地路线图

| Phase | 工期 | 内容 | 验收 |
|-------|------|------|------|
| **P0 地基** | 1 周 | `scene` 落盘 · outcome 扩展 · `cli.py learn` 骨架 · `market_state.py` · 分桶工具 · 回放脚本 | 回放跑通 10 个月历史,输出基线精准率曲线 |
| **P1 模型层** | 2 周 | `model_learner.py`(校准 + 错样本加权) · `blacklist.py` | 回放:精准率 ≥ 基线 +3pt;飞书告警联通 |
| **P1 观察期** | 2 周 | 线上运行,每日盘后自动学习 | 30 日精准率环比上升 |
| **P2 战法层** | 1.5 周 | `tactic_learner.py` · `ai_reason.py` | 回放:战法场景精准率 ≥ 基线 +3pt |
| **P2 观察期** | 2 周 | — | 战法子场景精准率改善 |
| **P3 特征层** | 2 周 | `alt_data.py` · `feature_learner.py` | 回放:累计精准率 ≥ 基线 +8pt;新特征 Importance > 5% |
| **P3 观察期** | 2 周 | — | 特征池稳定,无频繁淘汰反复 |
| **P4 价位层** | 1 周 | `price_learner.py` | 回放:短线止盈触发率 ≥ 40%,止损触发率 ≤ 30% |

**总工期**:纯开发 ~8.5 周,含观察期 ~14 周(约 3.5 个月)
**Kill switch**:任一阶段回放未达验收,不上线该阶段。

---

## 11. 风险与开放问题

| 风险 | 应对 |
|------|------|
| **样本不足导致学习过拟合** | 冷启动门槛 `samples < 30 回退默认`;分市场状态学习可能让 bear 状态样本很少 → 加保底"若某状态样本 <100 用 global 参数" |
| **5 日滞后让价位学习迟钝** | 先只学"触止盈/触止损"的 binary 信号,不学精细盈利;P4 观察期延长到 3 周 |
| **akshare 接口不稳定** | `alt_data` 每个特征独立 try/except,缓存失效也能降级 |
| **精准率定义"涨幅≥1% 算命中"的缺陷** | 内部增加 `hit_tier` 分档权重,虽然对外仍叫精准率,但学习信号更精细 |
| **市场状态误切导致策略抖动** | "连续 3 日触发才切"已防抖;额外保留 `pending` 状态供飞书告警 |
| **LLM API 成本失控** | 并发上限 + 缓存 + 仅 Top-N 调用;monthly 预算告警 ¥100/月硬上限 |
| **战法阈值调得太激进** | 单次 ≤ 1 单位;阈值锁定范围(如 roe_min ∈ [5,15]) |
| **blacklist 误伤牛股** | 无永久类,最长 30 天;设"撤销阈值"(站上 ma60 且近 10 日涨 >3%) |

---

## 12. 后续工作

- 本文档定稿 → 调 `writing-plans` 技能生成**实现计划**(plan.md),P0 将第一个交付
- P0 必须在任何学习模块之前完成,它是其他所有阶段的基础设施
- 每阶段完成后,**回放验收 > 线上观察期 > 下一阶段启动**,严格按门槛走
