# Antenna 学习系统 P2 战法层设计

> **阶段:** P2 `tactic_learner` — 4 战法阈值自适应 + 战法权重学习 + LLM 深度推荐理由
> **作者:** 意吟 + Claude
> **日期:** 2026-05-09
> **前置:** P0 地基(scene/hit_tier/market_state/orchestrator/feedback_io) + P1 模型层(isotonic 校准 + abs_threshold + 加权重训) 均已完成
> **目标:** 战法精准率从 baseline +3pt;共振股(≥2 战法命中)的相对提升 +5pt;Top-10 推荐每股有 LLM 深度理由 ≤ 0.15 元/天

---

## 1. 背景与痛点

### 1.1 现状痛点

- **4 战法阈值硬编码在代码里**(`server/predict_cmd.py:780-870`):
  - 价值:`roe>8 AND debt_ratio<50 AND total_score>=2`
  - 成长:`rev_growth>15 AND profit_growth>15 AND roe>12`(任 2 项)
  - 龙头:`roe>15 AND gross_margin>30 AND above_ma60`(任 2 项)
  - 逆向:`drawdown<-0.15 AND roe>3 AND debt_ratio<65`
- **从未根据效果调整**:bull 市场理应放宽门槛(把握机会),bear 市场理应收紧(避雷),都没做
- **共振只是简单标签**:`★价值+龙头+逆向` 仅显示文案,不影响 rise_prob 排序;实际多战法命中 ≠ 更精准
- **推荐理由只是模板字符串**(`build_commentary`):无法解释"为什么这只股在当前 market_state 下值得买入"
- **战法精准率从未量化**:用户看到"📊 价值"标签,但不知道价值战法近期实际命中率多少

### 1.2 现有条件

| 维度 | 状态 | 对 P2 的意义 |
|------|------|------------|
| `tracker.scene` | 已含 `tactic:value/growth/leader/contra` | 可按战法分桶统计精准率 |
| `outcome.hit_tier` | 已回填 1970 条 | 命中率计算样本充足 |
| `market_state.history` | 已回填 205 天(bull 95 / range 110 / bear 0) | 状态分桶可用,但 bear 桶仍稀疏 |
| `orchestrator.MODULES` | 已挂 market_state + model_learner | P2 只需追加一行 |
| `feishu` 双向交互 + AI 闲聊路由 | Qwen3 + Claude Haiku 双备份 | ai_reason 可直接复用 |
| `cmd_scan_bot._enrich_tactic_scores()` | 战法评分函数已存在 | P2 改造此函数读 yaml 阈值 |

### 1.3 决策摘要

| 决策点 | 选择 | 理由 |
|-------|------|------|
| 交付范围 | **完整 P2** = tactic_learner + ai_reason | 两者强耦合(共振权重影响 prompt 顺序);分两次做 spec 信息冗余 |
| 阈值调整算法 | **固定步长 ±1**(±2 触底回弹) | 对齐 P1 abs_threshold 风格,保守稳定;贝叶斯优化数据量不足 |
| 战法权重 | **共振 rise_prob 加权** | spec §6.2 既定;权重 = 近 90 日精准率归一化 |
| 触发频率 | **跟 model_learner 同步**(每次"学习"指令 + 每周日 15:30 兜底) | 与 P1 节拍一致;避免单独定时任务 |
| 代码落位 | **`learning/tactic_learner.py` + `tactic_learner.yaml`** | 与 P1 风格统一;参数全部走配置 |
| 冷启动 | 单战法样本 < 30 → **维持上版阈值**(不调整,不报错) | 比硬编码默认更保守(P1 经验:盲目重置反而弱化) |
| ai_reason 落位 | **`server/ai_reason.py`** | server/ 下与 cmd_chat 共用 LLM 路由;不进 learning/(不参与学习闭环) |
| ai_reason 缓存 | **(code, market_state, prob_cal_bucket=0.05) 24h 复用** | spec §7.4 既定;30 股 × 1K token ≈ 0.15 元/天 |

---

## 2. 总体架构

```
                ┌─ 每次"学习"指令 / 每周日 15:30 ─────────────┐
                │                                              │
                │  orchestrator.run_all 增加 tactic_learner   │
                │                                              │
                │  ┌─ tactic_learner.run() ───────────────┐   │
                │  │  1. 读 pred[scene=tactic:*] + outcome │   │
                │  │  2. 按 (state, tactic) 分 12 桶       │   │
                │  │     (3 state × 4 tactic)              │   │
                │  │  3. 每桶算近 90 日精准率              │   │
                │  │  4. 阈值微调(spec §3.2):              │   │
                │  │     <30% → 收严 +1 单位               │   │
                │  │     30~60% → 维持                     │   │
                │  │     >60% → 放宽 -1 单位               │   │
                │  │  5. 战法权重 = acc_90d 归一化         │   │
                │  │  6. 写 tactic_params.json             │   │
                │  │     + history(7 份快照)               │   │
                │  └───────────────────────────────────────┘   │
                └──────────────────────────────────────────────┘

                ┌─ 推荐路径(scan / tactic) ──────────────────┐
                │                                              │
                │  cmd_scan_bot._enrich_tactic_scores():       │
                │    state = market_state.load_current()       │
                │    params = tactic_params.json[state]        │
                │                                              │
                │    for tactic in (value/growth/leader/contra):│
                │       thresh = params[tactic][阈值列]         │
                │       hit = check(stock, thresh)              │
                │       if hit:                                 │
                │          rise_prob += params[tactic].weight   │
                │                                               │
                │    重新排序后 → assign_global_signals()       │
                └──────────────────────────────────────────────┘

                ┌─ 推荐推送(scan_bot Top-N / tactic) ─────────┐
                │                                              │
                │  ai_reason.generate(stock, context):         │
                │    cache_key = (code, state, prob_bucket)    │
                │    if cached(24h): return cached             │
                │                                              │
                │    prompt = 拼装(rise_prob/技术/财务/        │
                │                  战法/新闻/state)             │
                │    Qwen3(1K tokens) → Claude Haiku 兜底       │
                │    → 全失败回退 build_commentary             │
                │                                              │
                │    输出 JSON:                                │
                │      {bull_reasons[3], risks[2],             │
                │       verdict, confidence}                   │
                │                                              │
                │    嵌入飞书卡片 elements                     │
                └──────────────────────────────────────────────┘
```

**核心约定:**
- tactic_params 与 model_learner 解耦:任一失败,另一仍照跑
- 所有阈值有锁定上下界(见 §3.2),防止"学飞"
- ai_reason 缺省时回退现有 build_commentary,推荐路径永不阻断
- 每个 (state × tactic) 桶独立学习,bear 桶样本不足时单独维持

---

## 3. tactic_learner 算法详情

### 3.1 阈值参数化

每战法的硬编码门槛全部抽到 `tactic_learner.yaml`:

```yaml
defaults:           # 冷启动 / 单桶不足时的默认值
  value:
    roe_min:        8.0
    debt_ratio_max: 50.0
    total_score_min: 2.0
  growth:
    rev_growth_min:    15.0
    profit_growth_min: 15.0
    roe_min:           12.0
  leader:
    roe_min:        15.0
    gross_margin_min: 30.0
    above_ma60_required: true
  contra:
    drawdown_max: -0.15
    roe_min:       3.0
    debt_ratio_max: 65.0

bounds:             # 调参锁定范围(防"学飞")
  value:
    roe_min:        [5.0, 15.0]
    debt_ratio_max: [40.0, 60.0]
  growth:
    rev_growth_min: [10.0, 25.0]
    profit_growth_min: [10.0, 25.0]
    roe_min: [8.0, 18.0]
  leader:
    roe_min: [12.0, 20.0]
    gross_margin_min: [25.0, 40.0]
  contra:
    drawdown_max: [-0.25, -0.10]
    roe_min: [2.0, 8.0]
    debt_ratio_max: [55.0, 75.0]

step:               # 单次调整步长
  default: 1.0      # 大部分阈值用 ±1
  drawdown: -0.01   # drawdown 用 ±0.01(数值更小)

evaluation:
  lookback_days:  90
  min_samples:    30        # 单 (state, tactic) 桶 <30 维持上版
  acc_low:        0.30      # < 此值收严
  acc_high:       0.60      # > 此值放宽

weights:                    # 战法权重相关
  enable_resonance_boost: true
  boost_per_tactic:       0.02   # 每命中一个战法 rise_prob +0.02 上限
  weight_min:             0.05
  weight_max:             0.40
```

### 3.2 阈值微调规则

按近 90 日精准率(信号 = `signal=买入` 在该战法子场景下),**逐阈值独立调整**:

| 90 日精准率 | 动作 | 边界处理 |
|------------|------|---------|
| < 30%      | 阈值收严 +step | 触上界回退 -2 step(超买信号危险) |
| 30% ~ 60%  | 维持 | — |
| > 60%      | 阈值放宽 -step | 触下界回退 +2 step |
| 样本 <30   | 维持(reason='insufficient_samples') | — |

**为什么"逐阈值独立"**:`value` 战法有 3 个阈值(roe_min/debt_ratio_max/total_score_min),不是统一调一个倍数,而是按战法当前精准率统一改方向(收严 → 全部 +step,放宽 → 全部 -step)。每周一次,变化有限。

### 3.3 战法权重学习

```python
# 伪代码
weights = {}
for state in ("bull", "bear", "range"):
    for tactic in ("value", "growth", "leader", "contra"):
        bucket = bucket_samples[(state, tactic)]
        if len(bucket) < min_samples:
            weights[state][tactic] = cfg.weights.weight_min  # 兜底
            continue
        weights[state][tactic] = compute_acc(bucket)  # [0, 1]

    # 状态内归一化:权重总和 = 1.0,但单战法夹到 [weight_min, weight_max]
    state_weights = weights[state]
    total = sum(state_weights.values())
    for t in state_weights:
        state_weights[t] = clip(state_weights[t] / total, weight_min, weight_max)
```

**rise_prob 加权**(只在 scan_bot 共振场景):
```python
# cmd_scan_bot._enrich_tactic_scores 内
hits_count = sum(1 for t in TACTICS if r["tactic_tags"].get(t))
weighted_boost = sum(weights[state][t] for t in TACTICS if r["tactic_tags"].get(t))
r["rise_prob"] *= (1.0 + weighted_boost * cfg.weights.boost_per_tactic)
# 重要:只对 rise_prob_raw 加权;rise_prob_cal 由 calibrator 决定,不再加权
```

### 3.4 产物 `learning/tactic_params.json`

```json
{
  "version":           1,
  "last_calibrated":   "2026-05-09T15:32:01",
  "current_state":     "range",
  "params": {
    "bull": {
      "value":  {"roe_min": 9.0, "debt_ratio_max": 49.0, "total_score_min": 2.0,
                 "weight": 0.32, "acc_90d": 0.41, "samples": 87, "status": "ok"},
      "growth": {"rev_growth_min": 14.0, ..., "status": "ok"},
      "leader": {...},
      "contra": {...}
    },
    "bear":  {... 多 status="insufficient_samples" 维持默认},
    "range": {...}
  },
  "history": [
    {"date": "2026-05-09", "state": "range", "tactic": "value",
     "old": {"roe_min": 8.0}, "new": {"roe_min": 9.0},
     "acc_90d": 0.27, "samples": 102,
     "change": "value 90 日精准率 27% < 30%, roe_min 8.0 → 9.0(+step)"}
  ]
}
```

原子写 + 7 份历史快照(复用 `feedback_io.atomic_write_json` + `snapshot_file`)。

### 3.5 推荐路径集成点

| 位置 | 改动 |
|-----|-----|
| `server/predict_cmd._enrich_tactic_scores()` | 改为读 `tactic_params.json[current_state]`,替代硬编码;失败时回退 yaml.defaults |
| `server/predict_cmd.cmd_scan_bot()` | 收尾处加 `_apply_resonance_boost(results)`,按权重微调 rise_prob_raw |
| `learning/tactic_learner.load_params(state)` | 提供给 predict_cmd 的入口函数;失败回退 default;永不抛异常 |

---

## 4. ai_reason 算法详情

### 4.1 触发场景

| 调用方 | 数量 | 频率 |
|-------|------|------|
| `cmd_scan_bot` Top-N(N≤10) | ≤10 股 | 每次扫描 |
| `cmd_tactic` Top-N(N≤10) | ≤10 股 | 每次战法查询 |
| `cmd_predict` 单股 | 1 股 | 每次单股查询 |

预估:**每日 ≤ 30 次调用**(扫描 1 次/日 × 10 + 战法 ~3 次/日 × 5 + predict ~5 次/日 × 1)。

### 4.2 prompt 拼装

```python
def build_prompt(stock, context):
    return f"""你是一个股票分析师。请基于以下信息给出买入决策建议,严格输出 JSON。

【股票】{stock.name}({stock.code}) {stock.sector}
【市场状态】{context.market_state}(近 30 日精准率 {context.acc_30d:.1%})
【AI 评分】rise_prob_raw={stock.rise_prob_raw:.3f} → cal={stock.rise_prob_cal:.3f}
       全市场排名 {stock.global_rank}/{context.scan_total}
【技术】RSI6={tech.rsi6:.1f} MACD={tech.macd_hist:+.3f}
       距 52 周高 {stock.drawdown:+.1%}
【财务】ROE={fin.roe:.1f}% 毛利={fin.gross_margin:.1f}%
       负债 {fin.debt_ratio:.1f}% 营收增 {fin.rev_growth:+.1f}%
【命中战法】{', '.join(stock.tactic_hits) or '纯技术驱动'}
【最近资讯】{news_summary[:300]}

输出 JSON,字段:
  bull_reasons: 3 条买入理由(每条 ≤30 字)
  risks: 2 条风险(每条 ≤30 字)
  verdict: 稳健加仓 / 少量试仓 / 观望 / 规避
  confidence: 高 / 中 / 低
"""
```

### 4.3 路由与降级

```python
def generate(stock, context) -> dict:
    cache_key = (stock.code, context.market_state, round(stock.rise_prob_cal / 0.05) * 0.05)
    if cached := _cache.get(cache_key, max_age_h=24):
        return cached

    try:
        result = _call_qwen(prompt, max_tokens=300)  # 主路由
    except Exception:
        try:
            result = _call_claude_haiku(prompt)      # 兜底
        except Exception:
            result = _fallback_build_commentary(stock, context)  # 终极兜底

    _cache.set(cache_key, result, ttl=24*3600)
    return result
```

### 4.4 飞书卡片集成

`predict_cmd.cmd_scan_bot` / `cmd_tactic` / `cmd_predict` 在生成卡片 elements 时:
1. 调 `ai_reason.generate(stock, context)` 拿 JSON
2. 渲染为飞书 markdown:
   ```
   🎯 verdict (confidence)
   ✅ 看多
     · {bull_reasons[0]}
     · {bull_reasons[1]}
     · {bull_reasons[2]}
   ⚠️ 风险
     · {risks[0]}
     · {risks[1]}
   ```
3. 失败时直接跳过这一段,不影响其他卡片元素

### 4.5 成本控制

| 项 | 数值 |
|---|---|
| 每股 prompt | ≤ 1000 tokens |
| 每股 completion | ≤ 300 tokens |
| Qwen3-8B 价格 | 输入 ¥0.5/百万 + 输出 ¥1.5/百万 = 每股约 ¥0.001 |
| 日最大调用 | 30 次 → ¥0.03/天 → ¥0.9/月 |
| 缓存命中预期 | ≥ 50%(同股 24h 内多次推送) → 实际 ≤ ¥0.5/月 |

---

## 5. 失败隔离与告警

| 故障 | 行为 | 告警级别 |
|-----|-----|-----|
| pred[scene=tactic:*] / outcome 缺失 | tactic_learner 跳过该战法 | info |
| 单 (state, tactic) 桶 <30 样本 | 维持上版参数,reason='insufficient_samples' | info |
| `tactic_params.json` 文件损坏 | 推荐路径回退 yaml.defaults | **error + 飞书** |
| 阈值更新触界 | 自动夹紧到 bound,reason='clamped' | warn |
| ai_reason Qwen 失败 | 自动降级 Claude Haiku | warn |
| ai_reason 全失败 | 回退 build_commentary | warn(每会话首次) |
| ai_reason 缓存写盘失败 | 不缓存,下次重算 | info |
| ai_reason JSON 解析失败 | 视作全失败 | warn |

告警链路沿用 `learning/alerts.py`(P0 已有)。

---

## 6. 测试策略

### 6.1 分层测试

| 层 | 位置 | 覆盖 |
|---|---|---|
| 纯函数 | `tests/test_tactic_learner_units.py` | 阈值调整 / 权重归一化 / clamp 边界 |
| 冷启动 | `tests/test_tactic_learner_cold_start.py` | 单桶 <30 维持 / 全部桶不足 / 文件损坏回退 |
| 推荐路径 | `tests/test_tactic_resonance.py` | _enrich_tactic_scores 改造后保持原标签 + 加权;param 缺失回退 |
| 端到端 | `tests/test_p2_e2e.py` | orchestrator.run_all 含 tactic_learner;tactic_params.json 落盘 |
| ai_reason | `tests/test_ai_reason.py` | prompt 拼装 / 路由降级链 / 缓存命中 / JSON 解析失败 |

### 6.2 覆盖率目标

- `learning/tactic_learner.py` ≥ 90%
- `server/ai_reason.py` ≥ 85%(LLM mock 较多,边界覆盖优先)
- 不强求 e2e 跑真实 LLM;关键路径 mock 即可

### 6.3 历史回放

`scripts/replay_learn.py --with-p2` 加 P2 模式:
- 每日:用前 90 日数据重新拟合 tactic_params,用当日 pred[scene=tactic:*] 评估
- 输出 `learning/replay/p2_compare_*.csv`:`date / state / tactic / baseline_acc / p2_acc / delta_acc / weight`

---

## 7. 验收硬指标

| 指标 | 阈值 | 测量 |
|------|-----|------|
| 战法子场景精准率(均值) | ≥ baseline + 3pt | replay 198 天 `signal=买入 在 scene=tactic:*` |
| 共振股(≥2 战法)精准率 | ≥ baseline + 5pt | replay 同上,filter `len(tactic_hits) >= 2` |
| `tactic_params.json` schema 校验 | 通过 | `cli.py learn --check` 加入 |
| 阈值不触底/触顶 | 100% 在 bound 内 | replay 全程审计 history |
| ai_reason 平均日成本 | ≤ ¥0.5/月 | 实际部署观测一周 |
| ai_reason 失败时推荐仍出 | 100% | 集成测试断言 build_commentary 回退 |
| 推荐路径延迟增加 | < 50ms 单次 scan | benchmark `cmd_scan_bot` 前后对比 |

---

## 8. 新文件与修改清单

### 8.1 新文件

| 路径 | 职责 | 预估 LOC |
|-----|-----|---|
| `learning/tactic_learner.py` | 阈值微调 + 权重学习 + 产物落盘 | 300 |
| `learning/tactic_learner.yaml` | 默认阈值 / 边界 / 步长 / 评估窗口 | 80 |
| `server/ai_reason.py` | prompt 拼装 + LLM 路由 + 缓存 | 250 |
| `tests/test_tactic_learner_units.py` | 纯函数 | 150 |
| `tests/test_tactic_learner_cold_start.py` | 冷启动 + 文件损坏 | 100 |
| `tests/test_tactic_resonance.py` | 推荐路径加权 | 130 |
| `tests/test_p2_e2e.py` | 端到端 | 90 |
| `tests/test_ai_reason.py` | prompt + 路由 + 缓存 | 180 |
|  **合计** | | **~1280** |

### 8.2 修改文件

| 路径 | 改动点 |
|-----|-----|
| `learning/orchestrator.py` | `MODULES` 追加 `tactic_learner`(deps=["market_state"]) |
| `server/predict_cmd.py:_enrich_tactic_scores` | 改读 `tactic_params.json`,缺失回退 defaults |
| `server/predict_cmd.py:cmd_scan_bot` | 收尾加 `_apply_resonance_boost(results)` |
| `server/predict_cmd.py:cmd_scan_bot` | 卡片渲染处加 ai_reason 调用(可独立 flag 关闭) |
| `server/predict_cmd.py:cmd_tactic` | 同上 |
| `server/predict_cmd.py:cmd_predict` | 同上 |
| `scripts/replay_learn.py` | 加 `--with-p2` flag |
| `cli.py` | `learn --check` 加 `tactic_params.json` 校验 |
| `README.md` | P2 章节状态 ⏳ → ✅ |
| `config.example.yaml` | 加 `ai_reason: { enable: true, cache_ttl_h: 24 }` |
| `learning/orchestrator.check()` | 加 `learning/tactic_params.json` 校验 |

---

## 9. 路线图与后续

### 9.1 P2 内部 Sprint 切分

| Sprint | 范围 | 预估 |
|--------|------|------|
| **A** | tactic_learner 核心(阈值微调) + 单测 | 1 天 |
| **B** | 权重学习 + 推荐路径加权 + 集成测试 | 1 天 |
| **C** | ai_reason.py + 缓存 + 飞书卡片集成 | 1.5 天 |
| **D** | 编排器挂载 + replay --with-p2 验收 | 0.5 天 |
| **E** | 文档更新 + memory 同步 + commit | 0.5 天 |
| 合计 | | **~4.5 天** |

### 9.2 P3 — 特征层(预览)

| 维度 | 设计要点 |
|------|---------|
| `learning/feature_learner.py` | Permutation Importance 周级跑;连续 8 周 <5% 自动 drop |
| `learning/feature_weights.json` | `active` / `dropped` / `candidates` 三池 |
| `data/cache/alt/` 数据源 | 主力净流入(akshare `stock_individual_fund_flow`)/龙虎榜/北向资金 |
| `features/builder.py` 改造 | 读 `active` 列表,替代硬编码 `FEATURE_COLS` |
| 验收 | 特征数 25 → 18~22(精简);AUC 不下降 |

依赖 P2 完成的 alt_data 字段集成。

### 9.3 P4 — 价位层(预览)

| 维度 | 设计要点 |
|------|---------|
| `learning/price_learner.py` | 网格搜索 ATR×{1.0~2.5}/振幅×{2.0~4.0} |
| `learning/price_params.json` | 按 market_state 分 3 组系数 |
| 评估指标 | `(触止盈率 - 触止损率) × 平均持仓收益` |
| `features/analyser.suggest_dual_period_trades()` | 读 price_params,替代硬编码 |
| 验收 | 短线 5 日触止盈率 ≥ 基线 +5pt |

依赖 P0 已有的 `hit_5d` / `max_drawdown_5d` 字段(已回填)。

### 9.4 横向:黑名单(可在 P2 后并行)

| 维度 | 设计要点 |
|------|---------|
| `learning/blacklist.py` | 同 (code, market_state) 连续 3 次买入 miss → 拉黑 30 天 |
| `learning/blacklist.json` | `{code: {state, expires, reason}}` |
| 推荐路径过滤 | `cmd_scan_bot` 在 `assign_global_signals` 之前过滤 |
| 飞书"列表"指令 | 可选展示当前黑名单 |
| 验收 | 黑名单股票后续 30 日精准率 ≤ 全市场基线(证明拉黑合理) |

### 9.5 完整路线图状态

| 阶段 | 状态 | 工作量 |
|------|------|-------|
| P0 地基 | ✅ 已上线 | — |
| P1 模型层 | ✅ 已上线 | — |
| **P2 战法层** | **📐 设计完成,待实施** | ~4.5 天 |
| P3 特征层 | ⏳ 设计待写 | ~5 天 |
| P4 价位层 | ⏳ 设计待写 | ~3 天 |
| 横向 黑名单 | ⏳ 设计待写 | ~1.5 天 |

---

## 附录 A:关键函数签名

```python
# learning/tactic_learner.py

from dataclasses import dataclass
from typing import Literal

MarketState = Literal["bull", "bear", "range"]
Tactic = Literal["value", "growth", "leader", "contra"]


@dataclass(frozen=True)
class TacticConfig:
    defaults:    dict[Tactic, dict[str, float]]
    bounds:      dict[Tactic, dict[str, tuple[float, float]]]
    step:        dict[str, float]
    evaluation:  dict
    weights:     dict


def load_config(path: str = "learning/tactic_learner.yaml") -> TacticConfig: ...

def evaluate_tactic(
    state: MarketState,
    tactic: Tactic,
    lookback_days: int = 90,
) -> dict:
    """返回 {acc_90d, samples, hits}。样本不足时 acc=None。"""

def adjust_thresholds(
    current: dict[str, float],
    acc_90d: float | None,
    samples: int,
    cfg: TacticConfig,
    tactic: Tactic,
) -> tuple[dict[str, float], str]:
    """按 §3.2 表返回 (新阈值, 调整原因)。"""

def compute_resonance_weights(
    bucket_accs: dict[Tactic, float | None],
    cfg: TacticConfig,
) -> dict[Tactic, float]:
    """归一化权重,夹到 [weight_min, weight_max]。"""

def load_params(state: MarketState) -> dict:
    """供 predict_cmd 使用。文件缺失/损坏 → cfg.defaults。永不抛。"""

def run(date_str: str | None = None) -> dict:
    """编排器入口。返回 summary{(state, tactic): {old, new, status, ...}}。"""
```

```python
# server/ai_reason.py

@dataclass(frozen=True)
class StockContext:
    rise_prob_raw: float
    rise_prob_cal: float
    market_state:  str
    acc_30d:       float | None
    tactic_hits:   list[str]
    drawdown:      float
    fin:           dict
    tech:          dict
    news_summary:  str

def generate(stock_ctx: StockContext, code: str, name: str) -> dict | None:
    """返回 {bull_reasons, risks, verdict, confidence} 或 None(全降级)。"""

def render_card_section(reason: dict) -> list[dict]:
    """渲染为飞书 markdown elements 列表。"""
```

---

## 附录 B:风险与缓解

| 风险 | 后果 | 缓解 |
|------|-----|-----|
| bear 市场样本永远不足 | bear 策略不学习,永远用 defaults | 接受;P3 特征层稳定后样本积累自然增加 |
| 阈值调整振荡(本周 +1 / 下周 -1) | 战法稳定性差 | 单步 ±1 限制;每周一次,不更频繁 |
| 共振权重 push 过头(rise_prob > 1.0) | 信号失真 | `boost_per_tactic` 上限 0.02 + 全市场 rank 排序后切分位,绝对值压缩自然消失 |
| ai_reason LLM 输出不稳定 | bull_reasons/risks 文案漂移 | strict JSON schema + 失败重试 1 次 |
| ai_reason 成本超预期 | 月度账单失控 | 缓存 24h + 调用上限(每日 50 次硬封顶,触发后回退 build_commentary) |
| tactic_params.json 损坏 | 推荐路径行为不可预测 | 加载时 schema 校验 + atomic write + 损坏自动回退 defaults |
| 战法权重稀释模型预测 | rise_prob 偏离原始排序 | 仅作用于共振股,且 `boost_per_tactic` 小;不动 prob_cal(由 calibrator 决定) |

---

**end of spec**
