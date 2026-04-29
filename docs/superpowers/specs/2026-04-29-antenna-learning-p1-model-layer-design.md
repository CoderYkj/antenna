# Antenna 学习系统 P1 模型层设计

> **阶段:** P1 `model_learner` — 概率校准 + 错样本加权重训 + 绝对阈值 gate
> **作者:** 意吟 + Claude
> **日期:** 2026-04-29
> **前置:** P0 地基(scene 场景标签 / hit_tier 分档 / market_state 大盘状态 / orchestrator 编排器 / feedback_io 归因日志)已完成
> **目标精准率提升:** 买入信号精准率从当前 33% → ≥ 35%(+2pt 硬指标);Brier score ≤ raw × 0.95

---

## 1. 背景与痛点

### 1.1 现状痛点

- **买入信号 miss 率 65%**:最近 30 日 P0 打过 hit_tier 的 29 条买入级预测,19 条 miss(涨幅 < 1%)
- **rise_prob 压缩**:模型输出分布在 40~55%,靠**全市场前 10% 分位**切信号,而非绝对概率
- **仅靠调 `buy_top_pct` 已触天花板**:从 14% → 8% 反复探索,30 日精准率在 32~34% 徘徊,纯分位门槛无法再推高
- **错样本等权参与重训**:模型从没针对"买入且 miss"这类重点错样本做强化学习
- **无概率校准**:预测概率和实际命中率脱节,用户看到的 rise_prob 不是真实命中率

### 1.2 现有条件

| 维度 | 状态 | 对 P1 的意义 |
|------|------|------------|
| outcome 历史 | **198 天 2023 条**(2025-07-04 起) | 校准器样本充足,分 3 桶每桶仍数百条 |
| `actual_pct` 字段 | 全部有 | 可离线批量回填 `hit_tier` |
| `pred_*.jsonl` | 含 `rise_prob`、`signal`、`scene` | 可与 outcome 按 `(date, code, scene)` 联合查询 |
| market_state | 今日首条,198 天历史可回放打标 | 需要一次性回填脚本 |
| 模型训练 | 全市场拼盘 LightGBM,每天 1 个 pkl | 加 `sample_weight` 即可接入 |
| P0 编排器 | `orchestrator.run_all` + 失败隔离 + dry-run + `--check` 已上线 | P1 只需追加 MODULES 一行 |

### 1.3 决策摘要(来自 brainstorming)

| 决策点 | 选择 | 理由 |
|-------|------|------|
| 交付范围 | **完整 P1**(校准 + 错样本加权重训) | 数据已足,一次到位;黑名单留给后续 |
| signal 切法 | **绝对阈值 + 分位双门槛** | 单纯校准无法改排名,不动 signal 规则则精准率不会提升 |
| 代码落位 | 单模块 `learning/model_learner.py` | 对齐 spec,编排器接入最简;预估 350 LOC 在可控范围 |
| 权重参数 | 全部配置化到 `learning/model_learner.yaml` | 不硬编码;便于后续调参不改代码 |
| 周级重训 | 独立任务 `Antenna-WeeklyTrain`(周日 20:00) | 与日级校准解耦,失败不影响日级链路 |

---

## 2. 总体架构

```
                ┌─ 每日 15:30 (日级校准) ────────────────────┐
                │                                             │
                │  task_daily_review.py 末尾调 orchestrator   │
                │                                             │
                │  ┌─ model_learner.fit_calibrators() ────┐   │
                │  │  1. 读 pred + outcome 近 90 日        │   │
                │  │  2. 按 market_state 分 3 桶           │   │
                │  │  3. 用"最新模型"重打分历史样本得       │   │
                │  │     rise_prob_raw(贴合当前模型分布) │    │
                │  │  4. 每桶拟合 IsotonicRegression       │   │
                │  │  5. 保存 calibrator_{bull,bear,range} │   │
                │  │  6. 重校 abs_threshold(月度滚动)    │    │
                │  │  7. 写 model_learner.json + history   │   │
                │  └───────────────────────────────────────┘   │
                └─────────────────────────────────────────────┘

                ┌─ 每周日 20:00 (周级重训) ──────────────────┐
                │                                             │
                │  Antenna-WeeklyTrain 触发                    │
                │  python cli.py train --weighted             │
                │                                             │
                │  ┌─ model_learner.retrain_with_weights() ┐  │
                │  │  1. 读近 3 年行情 + build_features    │   │
                │  │  2. 对每行算 sample_weight(查表)    │   │
                │  │  3. lgb.Dataset(weight=w) 训练        │   │
                │  │  4. 保存 model_YYYYMMDD.pkl           │   │
                │  │  5. 立即触发 fit_calibrators 保持     │   │
                │  │     model ↔ calibrator 原子匹配       │   │
                │  └───────────────────────────────────────┘   │
                └─────────────────────────────────────────────┘

                ┌─ 盘中推荐路径(零额外延迟) ─────────────────┐
                │                                             │
                │  predict(code) / scan() / tactic():         │
                │    prob_raw  = model.predict(features)      │
                │    state     = load_current_state()         │
                │    cal       = load_calibrator(state)       │
                │    prob_cal  = cal.transform([prob_raw])[0] │
                │    abs_thres = load_abs_threshold()         │
                │                                             │
                │    assign_global_signals(..., abs_thres):   │
                │      门槛 1:rank_pct < buy_top_pct         │
                │      门槛 2:prob_cal >= abs_thres          │
                │      两门槛都过 → signal="买入"             │
                │      只过门槛 1 → signal="观望"             │
                │      都不过      → signal="回避"            │
                └─────────────────────────────────────────────┘
```

**核心约定:**
- 校准器**贴合当前模型**:每次拟合前用最新 pkl 重打分历史样本得 `rise_prob_raw`,避免"旧模型 prob + 新模型套"的 mismatch
- 市场状态历史**离线回填** 198 天后才能按状态分桶(见 §6)
- 周级重训每次**训完立刻重拟合校准器**,保持原子匹配
- 编排器只负责**日级校准**;周级重训独立任务,失败不影响日级
- calibrator 加载失败 → `predictor.predict()` 回退恒等映射(`prob_cal = prob_raw`),推荐路径永不阻断

---

## 3. signal 分配规则升级

### 3.1 双门槛判决逻辑

**从上到下择先匹配**(case-when 语义):

| # | 条件 | signal | 说明 |
|---|------|--------|------|
| 1 | `rank_pct < buy_top_pct` **且** `prob_cal >= abs_threshold` | 买入 | 双门槛都过 |
| 2 | `rank_pct < buy_top_pct` **且** `prob_cal < abs_threshold` | 观望(降级) | 分位过线但概率不够 |
| 3 | `buy_top_pct ≤ rank_pct < watch_top_pct` | 观望 | 分位刚够 watch 档(不受 abs_threshold 约束) |
| 4 | `rank_pct ≥ watch_top_pct` | 回避 | 连 watch 都没过 |

`watch_top_pct = min(buy_top_pct * 3, 0.40)` 沿用现状,由 `optimizer.py` 间接控制。

- `buy_top_pct` 继续由 `optimizer.py` 管(P0 已有)
- `abs_threshold` 由 `model_learner.py` 管(P1 新增)
- 两者**互补**:全市场都烂的日子(所有 prob_cal < abs_threshold),买入信号自动为 0,即使分位前 10%

### 3.2 abs_threshold 月度自校

在 `model_learner.fit_calibrators()` 末尾顺带做:

| 近 30 日 `signal=买入` 精准率 | 动作 | 边界夹逼 |
|-----|-----|-----|
| < 35% | 阈值 +0.02(收严) | 上限 0.60 |
| 35% ~ 55% | 维持 | — |
| > 55% | 阈值 −0.02(放宽) | 下限 0.30 |
| 近 30 日"买入"< 10 条 | 维持,写 `reason=insufficient_samples` | — |

每次调整写 `model_learner.json.threshold_history`。

### 3.3 推荐路径集成点

| 位置 | 改动 |
|-----|-----|
| `models/predictor.predict()` | 加载 calibrator + abs_threshold;输出新字段 `rise_prob_cal` |
| `models/predictor.assign_global_signals()` | 增加 `prob_cal >= abs_threshold` 第二门槛 |
| `learning/tracker.log_predictions()` | pred 快照同时写 `rise_prob_raw` + `rise_prob_cal` + `abs_threshold_snapshot`,便于后续归因 |
| `server/predict_cmd.py` | 飞书消息"上涨概率"字段改用 `rise_prob_cal`(文案更贴语义);保留 `rise_prob_raw` 作为技术字段 |

---

## 4. 错样本加权重训

### 4.1 sample_weight 公式

训练样本 `(日期, code, features, label)` 按 `(date, code)` 查历史 pred/outcome 的 `(signal, hit_tier)`:

| signal | hit_tier | 权重 | 含义 |
|--------|---------|------|-----|
| 买入 | miss | **2.0** | 重点惩罚"高估"错样本 |
| 买入 | weak | 1.2 | 略偏低,轻微惩罚 |
| 买入 | good | 1.0 | 基线 |
| 买入 | great | **1.5** | 强化"低估正确"样本 |
| 观望 / 回避 | 任意 | 0.8 | 降权(模型不追求精准分类"不推荐") |
| 无对应 pred 记录 | — | **1.0** | 历史默认(最近 3 年多数训练样本都是这种) |

### 4.2 参数配置 `learning/model_learner.yaml`

```yaml
sample_weights:
  buy_miss:   2.0
  buy_weak:   1.2
  buy_good:   1.0
  buy_great:  1.5
  non_buy:    0.8
  default:    1.0

calibration:
  lookback_days:          90
  min_samples_per_bucket: 50
  cold_start_fallback:    "global"   # 单桶 <50 时退化全局校准

absolute_threshold:
  initial:         0.45
  min:             0.30
  max:             0.60
  step:            0.02
  lookback_days:   30
  min_buy_signals: 10
```

全部参数**不进代码常量**。`model_learner` 启动时一次性读 yaml 到 frozen dataclass,跑完一趟释放。

### 4.3 Walk-forward 时间隔离(防泄漏)

sample_weight 来自历史 pred/outcome,但 hit_tier 依赖未来 5 日涨幅。若训练第 T 日样本时引用 T 当天的 outcome,等于**用 T+5 的信息训练 T 的样本**,是泄漏。

实现约束:`resolve_sample_weight(date_T, code)` 只允许读 `outcome_*.jsonl` 中**日期 ≤ T−6** 的记录。超出窗口或无记录一律 weight=1.0。

单元测试 `test_sample_weight_leakage.py` 针对这个约束做红绿测试。

### 4.4 训练流程

```python
# 伪代码,真实实现在 model_learner.retrain_with_weights()
def retrain_with_weights(df: pd.DataFrame, feature_cols: list) -> lgb.Booster:
    df = df.dropna(subset=feature_cols + ["label"])
    df["weight"] = df.apply(
        lambda row: resolve_sample_weight(row["date"], row["code"]),
        axis=1,
    )
    split_date = df["date"].max() - pd.DateOffset(months=3)
    train_df = df[df["date"] <= split_date]
    test_df  = df[df["date"] >  split_date]

    train_data = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["label"],
        weight=train_df["weight"],    # ← 唯一与原 train() 的差异
    )
    valid_data = lgb.Dataset(
        test_df[feature_cols],
        label=test_df["label"],
        weight=test_df["weight"],
        reference=train_data,
    )
    # ... 其余训练参数与 models/trainer.train 完全一致
```

---

## 5. 校准器 `fit_calibrators()`

### 5.1 算法

**walk-forward 约束澄清**:校准拟合**不**受 §4.3 的泄漏约束,因为它不是"训练下一版模型",而是学习"当前模型原始概率 → 真实命中率"的后验映射。用最新模型对历史 pred 打分、配对同日 outcome,没有把未来信息灌回"当前预测"的路径。


```python
# 伪代码
def fit_calibrators(date_str: str) -> dict:
    model = load_latest_model()
    state_history = load_state_history()  # P0 market_state.json.history

    # 收集 (rise_prob_raw, hit_binary, state) 三元组
    samples_by_bucket = {"bull": [], "bear": [], "range": []}
    for day in last_90_days(date_str):
        preds    = load_predictions(day)
        outcomes = load_outcomes(day)
        state    = load_state_on_date(day)  # P0 api
        for p in preds:
            if p["code"] not in outcomes:
                continue
            features  = rebuild_features(p["code"], day)   # 从缓存复原
            prob_raw  = float(model.predict(features[[feature_cols]])[0])
            hit       = 1 if outcomes[p["code"]]["hit_tier"] in ("good","great") else 0
            samples_by_bucket[state].append((prob_raw, hit))

    calibrators = {}
    for state, samples in samples_by_bucket.items():
        if len(samples) < cfg.min_samples_per_bucket:
            if cfg.cold_start_fallback == "global":
                samples = sum(samples_by_bucket.values(), [])
            else:
                calibrators[state] = IdentityCalibrator(reason="insufficient_data")
                continue
        if len(samples) < cfg.min_samples_per_bucket:
            calibrators[state] = IdentityCalibrator(reason="insufficient_data")
            continue
        X = [s[0] for s in samples]
        y = [s[1] for s in samples]
        cal = IsotonicRegression(out_of_bounds="clip").fit(X, y)
        calibrators[state] = cal
        save_pkl(f"models/saved/calibrator_{state}.pkl", cal)

    return calibrators
```

### 5.2 产物 `learning/model_learner.json`

```json
{
  "abs_threshold":   0.45,
  "last_calibrated": "2026-04-29T15:30:12",
  "last_retrained":  "2026-04-26T20:15:03",
  "model_file":      "models/saved/model_20260429.pkl",
  "calibrators": {
    "bull":  {"samples": 612, "brier_before": 0.213, "brier_after": 0.187, "status": "ok"},
    "bear":  {"samples": 445, "brier_before": 0.231, "brier_after": 0.194, "status": "ok"},
    "range": {"samples": 966, "brier_before": 0.218, "brier_after": 0.201, "status": "ok"}
  },
  "threshold_history": [
    {
      "date":          "2026-04-29",
      "acc_30d":       0.338,
      "buy_signals":   45,
      "abs_threshold": 0.45,
      "change":        "精准率 33.8% 低于下限 35%,阈值 +0.02 → 0.47"
    }
  ]
}
```

原子写(P0 `feedback_io.atomic_write_json` 已有),历史快照保留 7 份(P0 `snapshot_file` 已有)。

### 5.3 model ↔ calibrator 配对校验

pkl 存入时附加 `model_sha`(对应 pkl 的 sha256 前 12 位);加载时对比当前最新 `model_*.pkl` 的 sha,**不匹配 → 回退恒等映射 + alerts.jsonl error 级**。这样:
- 手动换模型(pkl 被替换但没重拟合校准器)→ 不会套用错位的校准器
- 周级重训忘了调 `fit_calibrators` → 自动降级,不静默出错

---

## 6. 历史回填(一次性,不入正式编排器)

P1 上线**前必须跑完**的预备作业。三个子任务 + 一个验收回放:

### 6.1 `scripts/backfill_hit_tier.py`

- 遍历 `learning/data/outcome_*.jsonl` 所有 198 天
- 对每条 `actual_pct` 跑 `compute_hit_tier`,**原子覆盖**写回(保留其他字段)
- 输出进度:`{processed}/{total} · {updated} 条补齐 · {already_tagged} 跳过`
- 支持 `--from YYYY-MM-DD --to YYYY-MM-DD` 限定范围
- 幂等:再跑一次对已有 hit_tier 的条目不重算

### 6.2 `scripts/backfill_market_state.py`

- 拉 `ak.stock_zh_index_daily(symbol="sh000300")` 全量(5897 条日线)
- 按 2025-07-04 ~ 2026-04-29 逐日调用 `update_state(df_until_T, T)` 模拟真实序列
- 最终 `learning/market_state.json.history` 应有 198 条(对齐 outcome 天数)
- 支持 `--from YYYY-MM-DD` 续跑(从 history 最后一条之后接力)
- checkpoint 机制:每处理 30 天原子写一次,中断后可续

### 6.3 `cli.py learn --date 2026-04-29`(初次触发)

回填完跑一次编排器,触发首版 `model_learner.fit_calibrators()`:
- 生成 3 份 `calibrator_{bull,bear,range}.pkl`
- 生成首版 `model_learner.json`
- 写一条 `feedback/model_learner_<当日>.json` 归因

### 6.4 `scripts/replay_learn.py --with-p1`

替换 P0 `--from / --scene` 的基线回放,新增 P1 对比:
- 遍历 198 天,每天前 90 天作为 "P1 校准训练窗口",当天的 pred 作为评估
- 输出 `learning/replay/p1_compare_YYYYMMDD_HHMMSS.csv`:
  - `date, scene, baseline_acc, p1_calibrated_acc, delta_acc, delta_brier`
- 验收硬指标见 §8

---

## 7. 失败隔离与告警

延续 P0 `orchestrator.run_all` 的"模块崩溃只影响自己"原则。

| 故障 | 行为 | 告警级别 |
|-----|-----|-----|
| pred/outcome 缺失 | `fit_calibrators` 跳过该日样本,不中断 | info |
| 单桶样本 < 50 | 回退全局校准(`cold_start_fallback`) | warn |
| 全局仍 < 50 | 输出 `IdentityCalibrator`,`status="insufficient_data"` | warn |
| 校准器拟合异常 | 保留上次 pkl,`calibrators[state].status="failed"` | **error + 飞书** |
| calibrator pkl 加载失败 | `predictor.predict()` 回退 `prob_cal = prob_raw` | error(但不阻断推荐) |
| 周级重训失败 | 保留上周 model pkl,下次日级校准照跑 | **error + 飞书** |
| abs_threshold 计算发散 | 夹到 [min, max],`reason="clamped"` | warn |
| 回填脚本中途失败 | checkpoint 保护,`--from` 续跑 | stdout 打印,无告警 |

告警链路沿用 P0 `learning/alerts.send_alert`(三路分发:飞书 + `alerts.jsonl` + stdout)。

---

## 8. 测试策略

### 8.1 分层测试

| 层 | 位置 | 覆盖 |
|---|---|---|
| 纯函数 | `tests/test_model_learner_units.py` | `resolve_sample_weight` · `compute_abs_threshold_delta` · `fit_calibrator_for_bucket` |
| walk-forward | `tests/test_sample_weight_leakage.py` | 构造 T+5 outcome,验证 `resolve_sample_weight(T)` 不引用 |
| 冷启动 | `tests/test_calibrator_cold_start.py` | 单桶 30 条 → 回退全局;全局仍不足 → IdentityCalibrator |
| 推荐路径 | `tests/test_predict_with_calibrator.py` | mock 3 calibrator + abs=0.45,验证 predict 输出正确 |
| 端到端冒烟 | `tests/test_p1_e2e.py` | 跑 `orchestrator.run_all()` 验证 calibrator pkl + json 生成;`--check` 通过 |
| 加权重训 | `tests/test_retrain_weighted.py` | 高权重样本上预测误差低于基线 |

### 8.2 覆盖率目标

- `learning/model_learner.py` 单元测试覆盖率 ≥ 90%
- 所有错误分支(§7 故障行为)都必须有至少一条测试断言
- walk-forward 泄漏测试是**硬性红绿**,不允许绕过

---

## 9. 验收硬指标

P1 上线且回填完成后必须通过:

| 指标 | 阈值 | 测量方式 |
|------|-----|------|
| 买入信号精准率 | ≥ 35%(基线 33% + 2pt) | `replay_learn.py --with-p1` 198 天回放,`signal=买入 且 hit_tier ∈ {good, great}` 比例 |
| Brier score | ≤ raw × 0.95 | `replay_learn.py --with-p1` 输出 CSV 的 `delta_brier` 列均值 |
| `cli.py learn --check` | 通过 | 新增 calibrator pkl 合法性 + `model_learner.json` schema 校验 |
| 单元测试覆盖率 | ≥ 90%(model_learner) | `pytest --cov=learning.model_learner` |
| 推荐路径延迟 | 单次 predict 额外 < 10ms | 校准器加载缓存 + 单个 transform 开销 |
| calibrator 失败降级 | 删掉 pkl 后 predict 仍出信号 | 集成测试断言回退到 IdentityCalibrator 路径 |

---

## 10. 新文件与修改清单

### 10.1 新文件

| 路径 | 职责 | 预估 LOC |
|-----|-----|---|
| `learning/model_learner.py` | 校准 + 加权重训 + 阈值自校 | 350 |
| `learning/model_learner.yaml` | 可配置参数 | 40 |
| `scripts/backfill_hit_tier.py` | 2023 条 outcome 批量回填 | 60 |
| `scripts/backfill_market_state.py` | 198 天 state 回放 | 100 |
| `tests/test_model_learner_units.py` | 纯函数单元 | 150 |
| `tests/test_sample_weight_leakage.py` | walk-forward 隔离 | 80 |
| `tests/test_calibrator_cold_start.py` | 冷启动兜底 | 100 |
| `tests/test_predict_with_calibrator.py` | 推荐路径集成 | 120 |
| `tests/test_p1_e2e.py` | 端到端冒烟 | 100 |
| `tests/test_retrain_weighted.py` | 加权重训冒烟 | 120 |
|  **合计** | | **~1220** |

### 10.2 修改文件

| 路径 | 改动点 |
|-----|-----|
| `models/predictor.py` | `predict()` 加载 calibrator + abs_threshold;输出 `rise_prob_cal`;`assign_global_signals()` 增加绝对阈值门槛 |
| `models/trainer.py` | 新增 `train_weighted(df, feature_cols, weights)` 函数(不改原 `train`) |
| `cli.py` | `cmd_train` 加 `--weighted` flag;新增 `cmd_backfill` 子命令 |
| `learning/orchestrator.py` | `MODULES` 追加 `model_learner` 项(deps=["market_state"]) |
| `learning/tracker.py:log_predictions()` | pred 快照加 `rise_prob_raw` / `rise_prob_cal` / `abs_threshold_snapshot` |
| `server/predict_cmd.py` | 飞书消息"上涨概率"用 `rise_prob_cal` |
| `scripts/replay_learn.py` | 加 `--with-p1` flag,输出 P0 vs P1 对比 CSV |
| `scripts/setup_tasks.ps1` | 注册 `Antenna-WeeklyTrain`(周日 20:00 → `cli.py train --weighted`) |
| `README.md` | 更新 P1 章节与路线图;P1 状态从 ⏳ 改 ✅ |
| `.gitignore` | 加 `models/saved/calibrator_*.pkl` |
| `config.yaml` | 可选:`learning.model_learner.yaml_path`(默认 `learning/model_learner.yaml`) |

---

## 11. 路线图与后续

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 地基 | scene 标签 / hit_tier / market_state / orchestrator / replay_learn | ✅ 完成 |
| **P1 模型层** | **本 spec:isotonic 校准 + 错样本加权重训 + 绝对阈值 gate** | **📐 设计中** |
| P2 战法层 | tactic_learner(阈值/权重自适应)+ AI 深度理由 | ⏳ 未启动 |
| P3 特征层 | alt_data 资金面/情绪面 + feature_learner(指标贡献度) | ⏳ 未启动 |
| P4 价位层 | price_learner(ATR/振幅系数网格搜索) | ⏳ 未启动 |
| 横向:黑名单 | 连错股票自动拉黑 30 天,推荐路径过滤 | ⏳ P1 稳定后启动 |

### 后续可接工作(不在本 spec 范围)

- **黑名单(blacklist)**:同股 × 市场状态下连续 3 次买入信号 miss → 拉黑 30 天
- **AI 深度理由(ai_reason)**:推荐时实时调用 LLM 解释为什么该股评分高
- **资金面数据(alt_data)**:主力净流入、龙虎榜等数据源接入,为 P3 做准备

---

## 附录 A:关键函数签名

```python
# learning/model_learner.py

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MarketState = Literal["bull", "bear", "range"]

@dataclass(frozen=True)
class LearnerConfig:
    sample_weights:       dict[str, float]
    calibration:          dict
    absolute_threshold:   dict

def load_config(path: str = "learning/model_learner.yaml") -> LearnerConfig: ...

def resolve_sample_weight(date: str, code: str, cfg: LearnerConfig) -> float:
    """给定训练样本 (date, code) 按 pred/outcome 查表返回 weight。
    walk-forward 约束:只读 outcome 日期 ≤ date−6。"""

def compute_abs_threshold_delta(acc_30d: float, buy_signals: int, cfg: LearnerConfig) -> float:
    """按 §3.2 表返回 delta(-0.02 / 0 / +0.02)。"""

def fit_calibrator_for_bucket(
    samples: list[tuple[float, int]],
    cfg: LearnerConfig,
) -> IsotonicRegression | "IdentityCalibrator":
    """拟合单桶 IsotonicRegression;样本不足返回 IdentityCalibrator(reason='insufficient_data')。"""

def fit_calibrators(date_str: str, cfg: LearnerConfig) -> dict[MarketState, object]:
    """日级入口:按市场状态拟合 3 个校准器,保存 pkl 并更新 model_learner.json。"""

def retrain_with_weights(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: LearnerConfig,
) -> lgb.Booster:
    """周级入口:带 sample_weight 的全量重训,返回 LightGBM Booster。"""

def run(date_str: str | None = None) -> dict:
    """编排器入口:调 fit_calibrators + 更新 abs_threshold。"""
```

---

## 附录 B:计算 `rebuild_features(code, day)`

校准器拟合需要用**当前最新模型**对历史样本重打分。简化实现:
- 从 `data/cache/{code}.parquet` 读该股全量行情
- 裁剪到 `day` 及之前
- 调 `features.builder.build_features` 算特征
- 返回最后一行(对应 `day` 收盘后的特征快照)

若缓存缺失(极少见)→ 跳过该样本,`alerts.jsonl` info 级。

---

## 附录 C:风险与缓解

| 风险 | 可能后果 | 缓解 |
|------|--------|-----|
| `rebuild_features` 与盘中预测特征不一致 | 校准器拟合点偏离真实 | 复用 `build_features` 同一函数,单元测试断言一致性 |
| isotonic 过拟合小样本 | 校准器不稳定 | 分桶最小 50 条硬门槛 + 3 档 fallback |
| abs_threshold 震荡 | 信号数来回翻飞 | 月度调整 + ±0.02 小步长 + [0.30, 0.60] 边界夹逼 |
| 周级重训卡 CPU | 拖慢机器 | 独立任务 20:00 执行,盘后且用户无感;失败告警 |
| pred 日期 × code 匹配失败 | sample_weight 全退化 1.0 | 单元测试覆盖查表 miss 路径;alerts.jsonl 日志 |
| calibrator.pkl 版本错配 | predict 返回乱 | pkl 存入时附加 `model_sha`,加载时校验失配 → 回退恒等 |

---

**end of spec**
