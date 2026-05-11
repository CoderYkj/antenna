# Antenna 学习系统 · 横向黑名单设计

> **阶段:** 横向能力(独立于 P0-P4 主线)
> **作者:** 意吟 + Claude
> **日期:** 2026-05-11
> **前置:** P0 地基(scene/hit_tier/market_state/orchestrator) + P1 模型层 + P2 战法层均已完成
> **目标:** 过滤"连错股票",提升推荐质量,降低用户对"假信号"的疲劳感

---

## 1. 背景与痛点

### 1.1 现状痛点

- **连错股票继续出现在推荐**:即使某只股票在 bull 市场下连续 3 次发出"买入"信号都 miss,下次 scan 时它仍有机会再上榜
- **用户感知**: 推荐列表反复出现"老熟人",信任度下降
- **模型层无法解决**: calibrator 是全市场校准,单股维度的"连错"信息无法在模型权重里体现
- **战法层无法解决**: tactic_learner 调整战法阈值,不过滤个股

### 1.2 现有条件

- ✅ `pred[scene=scan]` / `outcome.hit_tier` 数据完整,按 (code, state) 聚合连续 miss 有依据
- ✅ `market_state.load_state_on_date(d)` API 可查历史日状态
- ✅ `orchestrator.MODULES` 已成熟,新 learner 追加一行即可
- ✅ `feedback_io` 提供原子写 + 历史快照

### 1.3 决策摘要

| 决策点 | 选择 | 理由 |
|-------|------|------|
| **B1 streak 阈值** | 连续 **3 次** buy+miss | 3 次避免偶发噪声(A 股单股日内变动 10% 常见);>3 次则太晚 |
| **B1 拉黑天数** | **30 天** | 给市场"重置"时间;A 股中短线 ≤ 1 个月周期 |
| **B2 (code, state) 维度** | **按 (code, state) 分别拉黑** | bull 下连错不代表 bear 下也差(风格轮动) |
| **B3 覆盖范围** | **仅 scan 场景** | tactic 有严格财务筛选,命中率高,不需额外过滤;predict 是用户主动查询,不屏蔽 |
| **B4 自动过期** | **到期自动解封** | 每日编排器 run 时清理 `expires < today`; 无需人工 |
| **B5 白名单 override** | **watchlist 自动白名单** | 自选股不参与拉黑,用户主动跟踪的股票不屏蔽 |
| **B6 代码落位** | `learning/blacklist.py` + `blacklist.yaml` | 与 P1/P2 风格一致 |
| **B7 触发时机** | **跟 model_learner / tactic_learner 同步** | 每次"学习"指令 + 周日 20:00 兜底 |

---

## 2. 总体架构

```
                ┌─ 每次"学习"指令 / 每周日 15:30 ────────────┐
                │                                             │
                │  orchestrator.run_all 追加 blacklist       │
                │                                             │
                │  ┌─ blacklist.run() ────────────────────┐   │
                │  │  1. 扫近 90 日 pred[scene=scan]       │   │
                │  │  2. 按 (code, state) 计算 miss streak │   │
                │  │  3. streak ≥ 3 且最后一次 < 7 日内    │   │
                │  │     → 加入黑名单, expires=today+30d   │   │
                │  │  4. 清理已 expires < today 的条目     │   │
                │  │  5. 写 blacklist.json(原子 + 快照)    │   │
                │  └───────────────────────────────────────┘   │
                └─────────────────────────────────────────────┘

                ┌─ 推荐路径(cmd_scan_bot) ───────────────────┐
                │                                             │
                │  assign_global_signals 之前:                │
                │    bl = load_blacklist()                    │
                │    state = market_state.current             │
                │    watchlist = cfg.universe.watchlist       │
                │                                             │
                │    results = [                              │
                │      r for r in results                     │
                │      if not bl.is_blocked(r.code, state)    │
                │         or r.code in watchlist              │
                │    ]                                        │
                │    assign_global_signals(results, ...)      │
                └─────────────────────────────────────────────┘
```

---

## 3. 算法详情

### 3.1 miss streak 计算

按 `(code, state)` 分组,扫近 `lookback_days=90` 日每日 `pred[scene=scan]`:

```python
# 伪代码
streaks = {}  # (code, state) → [{date, tier, signal}, ...]

for d in last_90_days:
    state = load_state_on_date(d)
    for p in load_predictions_by_scene(d, "scan"):
        if p["signal"] != "买入":
            continue
        o = load_outcomes(d).get(p["code"])
        if not o:
            continue
        tier = o.get("hit_tier") or compute_hit_tier(o["actual_pct"])
        if tier is None:
            continue
        key = (p["code"], state)
        streaks.setdefault(key, []).append({
            "date": d, "tier": tier, "signal": p["signal"],
        })

# 计算每组"最近连续 miss 次数"(从最近一条往前数,连续 miss 直到遇到 good/great)
for key, events in streaks.items():
    events.sort(key=lambda x: x["date"], reverse=True)  # 降序
    consecutive_miss = 0
    for e in events:
        if e["tier"] in ("miss", "weak"):   # weak 也算不达预期
            consecutive_miss += 1
        else:
            break
    streaks[key] = {"count": consecutive_miss, "last_date": events[0]["date"]}
```

### 3.2 拉黑判定

```python
# 伪代码
for (code, state), info in streaks.items():
    if info["count"] < cfg.streak_threshold:   # 默认 3
        continue
    # "最近活跃性":最后一次 miss 必须在 7 天内,避免把早已不推荐的股票也拉黑
    if (today - info["last_date"]).days > cfg.last_activity_max_days:   # 默认 7
        continue
    # 白名单 override
    if code in watchlist:
        continue
    # 写入黑名单
    blacklist[(code, state)] = {
        "until":       today + cfg.block_days,   # 默认 30
        "streak_count": info["count"],
        "last_miss":   info["last_date"],
        "added_at":    today,
    }
```

### 3.3 自动过期

每次 `blacklist.run()` 在拉黑新条目之前,先清理:

```python
today_str = today.strftime("%Y-%m-%d")
blacklist = {k: v for k, v in blacklist.items() if v["until"] > today_str}
```

### 3.4 推荐路径过滤

`cmd_scan_bot` 在 `assign_global_signals` 之前插入一段过滤:

```python
from learning.blacklist import load_blacklist
from learning import market_state

state = market_state.load_current_state().get("current", "range")
bl = load_blacklist()
watchlist = set(cfg.get("universe", {}).get("watchlist", []))

before_count = len(results)
results = [
    r for r in results
    if r["code"] in watchlist
       or not bl.is_blocked(r["code"], state)
]
after_count = len(results)
log.info(f"[blacklist] filtered {before_count - after_count} stocks")
```

### 3.5 配置 `learning/blacklist.yaml`

```yaml
streak_threshold:       3      # 连续 N 次 buy+miss 触发
block_days:             30     # 拉黑天数
last_activity_max_days: 7      # 最后一次 miss 必须在 N 天内
lookback_days:          90     # 扫 pred/outcome 回溯天数
enable_scan_filter:     true   # 是否在 cmd_scan_bot 过滤
enable_tactic_filter:   false  # 是否在 cmd_tactic 过滤(B3 决定 false)
enable_predict_filter:  false  # cmd_predict 不过滤
# 可选:按 market_state 独立配置 streak_threshold
# 留空则所有 state 用上面的默认值
per_state_threshold:
  bull:  3
  bear:  2    # bear 更严格,2 次即拉黑
  range: 3
```

### 3.6 产物 `learning/blacklist.json`

```json
{
  "version": 1,
  "updated_at": "2026-05-11T15:36:01",
  "entries": {
    "300785|range": {
      "until":        "2026-06-10",
      "streak_count": 3,
      "last_miss":    "2026-05-09",
      "added_at":     "2026-05-11",
      "reason":       "bull 状态下 3 次 buy+miss"
    },
    "002174|bull":  { ... }
  },
  "history": [
    {"date": "2026-05-11", "action": "add", "code": "300785", "state": "range",
     "streak": 3, "until": "2026-06-10"},
    {"date": "2026-05-11", "action": "expire", "code": "600000", "state": "bull"},
    ...
  ]
}
```

history 保留最近 200 条,原子写 + 7 份历史快照。

---

## 4. API 签名

```python
# learning/blacklist.py

from dataclasses import dataclass
from typing import Literal

MarketState = Literal["bull", "bear", "range"]

@dataclass(frozen=True)
class BlacklistConfig:
    streak_threshold:       int
    block_days:             int
    last_activity_max_days: int
    lookback_days:          int
    enable_scan_filter:     bool
    per_state_threshold:    dict[str, int]


def load_config(path: str | None = None) -> BlacklistConfig: ...

def evaluate_streaks(date_str: str, cfg: BlacklistConfig) -> dict:
    """遍历近 lookback 日 pred[scene=scan] × outcome,
    返回 {(code, state): {count, last_date}}。"""

def apply_decay(blacklist: dict, today: str) -> tuple[dict, list[dict]]:
    """清理已过期条目。返回 (新 blacklist, 过期 history 记录列表)。"""

def fit_blacklist(date_str: str, cfg: BlacklistConfig | None = None) -> dict:
    """主编排入口:evaluate + decay + 产物落盘。返回 summary。"""

# 供 predict_cmd 调用
class Blacklist:
    def is_blocked(self, code: str, state: MarketState) -> bool: ...

def load_blacklist() -> Blacklist:
    """永不抛异常,文件缺失/损坏返回空黑名单。"""

def run(date_str: str | None = None) -> dict:
    """编排器入口。"""
```

---

## 5. 失败隔离与告警

| 故障 | 行为 | 告警级别 |
|-----|-----|-----|
| pred/outcome 缺失 | 跳过该日,streak 不计 | info |
| `blacklist.json` 损坏 | `load_blacklist()` 返回空黑名单,推荐路径不过滤 | **error + 飞书** |
| watchlist 加载失败 | 视作空集(所有股票都参与拉黑判定) | warn |
| market_state 不可用 | 用 `DEFAULT_STATE="range"` 兜底 | warn |

沿用 P0 `learning/alerts.py` 告警链路。

---

## 6. 测试策略

### 6.1 分层测试

| 层 | 位置 | 覆盖 |
|---|---|---|
| 纯函数 | `tests/test_blacklist.py` | `evaluate_streaks` / `apply_decay` / `Blacklist.is_blocked` |
| 冷启动 | 同上 | 无历史数据 / 损坏 json / watchlist 白名单 override |
| 集成 | 同上 | `cmd_scan_bot` 调用前后过滤前后行为对比 |

### 6.2 覆盖率目标

- `learning/blacklist.py` ≥ 85%

### 6.3 关键测试用例

- streak=2 不触发(< 阈值 3)
- streak=3 触发 + until 正确
- streak=5 触发 + 记录 streak_count=5
- 最后一次 miss > 7 天 → 跳过
- watchlist 股票始终不上榜
- expires < today 自动清理 + history 记录 action=expire
- bear 状态独立阈值(按 per_state_threshold 的 2)

---

## 7. 验收硬指标

| 指标 | 阈值 | 测量 |
|------|-----|------|
| 黑名单命中率 | 拉黑股后续 30 日精准率 ≤ 全市场基线 -5pt | 3 个月观察后统计 |
| 推荐质量 | scan 推荐的"连错熟股"减少 ≥ 80% | 对比上线前后 1 周 |
| 推荐数量 | cmd_scan_bot 过滤导致有效候选减少 ≤ 15% | 单日样本对比 |
| 黑名单规模 | 任意时刻 < 200 只(总 5193 只的 4%) | 生产监测 |

---

## 8. 实施 Sprint

| Sprint | 内容 | LOC 估 | 工作量 |
|--------|------|------|------|
| A | `blacklist.yaml` + `blacklist.py` 核心 + 单测 | 280 | 0.5 天 |
| B | `predict_cmd.cmd_scan_bot` 过滤集成 + 编排器挂载 + e2e | 100 | 0.3 天 |
| C | README 更新 + memory 同步 + commit | 50 | 0.2 天 |
| 合计 | | **~430** | **~1 天** |

---

## 9. 路线图影响

| 阶段 | 状态 | 备注 |
|------|------|------|
| P0 地基层 | ✅ 已上线 | — |
| P1 模型层 | ✅ 已上线 | — |
| P2 战法层 | ✅ 已上线 | — |
| **横向黑名单** | **📐 设计完成,待实施** | 本 spec |
| P3 特征层 | ⏳ 仅大纲 | 需要先做 alt_data |
| P4 价位层 | ⏳ 仅大纲 | 数据已就绪,等黑名单后推进 |

---

**end of spec**
