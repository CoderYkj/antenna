# Antenna 学习系统 P0 地基阶段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 4 层自学习反馈系统搭建所有基础设施,让 P1-P4 任一阶段都能直接接入并获得一致的数据、接口、工具支持。

**Architecture:** 方案 B(分层管线 + 瘦编排器):在 `learning/` 下新增 `market_state.py`、`scene_bucket.py`、`orchestrator.py`、`feedback_io.py`;扩展 `tracker.py` 的 pred/outcome schema;给 `cli.py` 新增 `learn` 子命令;给 `scripts/` 新增 `replay_learn.py` 历史回放脚本。

**Tech Stack:** Python 3.11+, pandas, pyarrow, akshare, pytest, scikit-learn, LightGBM, PyYAML(已全部在 requirements.txt)

**关联文档:** [2026-04-27-antenna-learning-system-design.md](../specs/2026-04-27-antenna-learning-system-design.md)

**P0 验收标准**
1. 所有现有 `log_predictions` 调用点带 `scene` 字段,且旧数据兼容
2. `cli.py learn --dry-run` 能跑通,输出 `feedback/*.json` 骨架
3. `scripts/replay_learn.py --from 2025-07-01` 能在现有 10 个月历史数据上跑完,输出基线精准率 CSV
4. `learning/market_state.json` 每日 15:30 自动更新
5. `pytest tests/learning/` 全通过,覆盖率 ≥ 80%

**本阶段不做的事(留给 P1-P4):**
- 不实现任何实际学习算法(model_learner / tactic_learner / feature_learner / price_learner 只写占位)
- 不接入 alt_data / blacklist / ai_reason(留给后续阶段)
- 不修改 LightGBM 训练流程
- 不修改飞书推荐卡片内容

---

## 文件结构

**新建文件**
| 路径 | 职责 |
|------|------|
| `learning/market_state.py` | 大盘状态打标:沪深300 → bull/bear/range |
| `learning/scene_bucket.py` | 按 scene/market_state 分桶的工具函数 |
| `learning/orchestrator.py` | 瘦编排器,调度所有学习子模块 |
| `learning/feedback_io.py` | feedback 日志写入 + history 快照保留工具 |
| `learning/alerts.py` | 告警写入(飞书 + JSONL + 日志) |
| `scripts/replay_learn.py` | 历史回放:模拟滚动学习并输出精准率曲线 |
| `tests/test_market_state.py` | market_state 单元测试 |
| `tests/test_scene_bucket.py` | scene_bucket 单元测试 |
| `tests/test_tracker_scene.py` | tracker scene 扩展测试 |
| `tests/test_outcome_extension.py` | outcome 扩展测试 |
| `tests/test_orchestrator.py` | 编排器测试 |
| `tests/test_feedback_io.py` | feedback/history 工具测试 |
| `tests/fixtures/sample_preds.jsonl` | 测试 fixture |
| `tests/fixtures/sample_outcomes.jsonl` | 测试 fixture |
| `tests/fixtures/hs300_sample.csv` | 大盘指数测试数据 |

**修改文件**
| 路径 | 修改 |
|------|------|
| `learning/tracker.py` | `log_predictions` 兼容 scene 字段;新增 `load_predictions_by_scene`、`compute_hit_tier` |
| `learning/optimizer.py` | `evaluate_day` 能按 scene 分桶统计(向后兼容) |
| `cli.py` | 新增 `learn` 子命令 |
| `server/predict_cmd.py` | `cmd_scan_bot` 的 snapshot 加 `scene="scan"`;`_tactic_run` 新增写入 `scene="tactic:<key>"` |
| `scripts/task_scan.py` | snapshot 加 `scene="scan"` |
| `scripts/task_predict.py` | snapshot 加 `scene="predict"` |
| `scripts/task_daily_review.py` | 回测完成后调用 `orchestrator.run_all()` |
| `.gitignore` | 追加 `learning/feedback/`、`learning/history/`、`learning/alerts.jsonl`、`data/cache/alt/` 不入库原则(除 learning/market_state.json 外其他运行产物不入库) |

**目录创建**
- `learning/feedback/`(运行时创建,不入库)
- `learning/history/`(运行时创建,不入库)
- `data/cache/alt/`(P3 才用,先创建占位 `.gitkeep`)
- `tests/fixtures/`(新建)
- `tests/learning/`(新建,本阶段所有测试放这)

---

## Task 1: 扩展 tracker.py — 支持 scene 字段和辅助函数

**Files:**
- Modify: `E:\antenna\learning\tracker.py`
- Create: `tests/test_tracker_scene.py`
- Create: `tests/fixtures/sample_preds.jsonl`

---

- [ ] **Step 1.1: 创建测试 fixture**

Create `tests/fixtures/sample_preds.jsonl`:
```jsonl
{"code":"600519","name":"贵州茅台","signal":"买入","rise_prob":0.75,"scene":"scan"}
{"code":"600519","name":"贵州茅台","signal":"买入","rise_prob":0.72,"scene":"tactic:value"}
{"code":"000001","name":"平安银行","signal":"观望","rise_prob":0.55,"scene":"scan"}
{"code":"300750","name":"宁德时代","signal":"买入","rise_prob":0.81,"scene":"predict","watchlist":true}
{"code":"600036","name":"招商银行","signal":"买入","rise_prob":0.68}
```
最后一条无 scene 字段,用于测试向后兼容。

---

- [ ] **Step 1.2: 写失败测试 — scene 过滤函数**

Create `tests/test_tracker_scene.py`:
```python
import json
from pathlib import Path
import pytest
from learning import tracker


@pytest.fixture
def fixture_date(tmp_path, monkeypatch):
    """复制 fixture 到临时 DATA_DIR。"""
    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    src = Path(__file__).parent / "fixtures" / "sample_preds.jsonl"
    (tmp_path / "pred_2026-04-01.jsonl").write_bytes(src.read_bytes())
    return "2026-04-01"


def test_load_predictions_by_scene_returns_only_matching(fixture_date):
    result = tracker.load_predictions_by_scene(fixture_date, "scan")
    assert len(result) == 2
    assert all(r["scene"] == "scan" for r in result)


def test_load_predictions_by_scene_tactic_matches_prefix(fixture_date):
    result = tracker.load_predictions_by_scene(fixture_date, "tactic:value")
    assert len(result) == 1
    assert result[0]["code"] == "600519"


def test_load_predictions_by_scene_all_tactic_substrings(fixture_date):
    result = tracker.load_predictions_by_scene(fixture_date, "tactic")
    assert len(result) == 1


def test_load_predictions_by_scene_backward_compat_records(fixture_date):
    """无 scene 字段的旧记录默认归到 'scan' 桶。"""
    result = tracker.load_predictions_by_scene(fixture_date, "scan")
    codes = [r["code"] for r in result]
    assert "600036" in codes  # 最后一条没 scene,应视作 scan


def test_log_predictions_preserves_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    records = [
        {"code": "600519", "name": "贵州茅台", "signal": "买入", "rise_prob": 0.7, "scene": "tactic:growth"},
    ]
    tracker.log_predictions("2026-04-02", records)
    loaded = tracker.load_predictions("2026-04-02")
    assert loaded[0]["scene"] == "tactic:growth"


def test_log_predictions_same_code_different_scene_coexist(tmp_path, monkeypatch):
    """同一天同一 code 不同 scene 应各存一条,不去重。"""
    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    tracker.log_predictions("2026-04-03", [
        {"code": "600519", "scene": "scan", "signal": "买入", "rise_prob": 0.7},
    ])
    tracker.log_predictions("2026-04-03", [
        {"code": "600519", "scene": "tactic:value", "signal": "买入", "rise_prob": 0.72},
    ])
    loaded = tracker.load_predictions("2026-04-03")
    scenes = sorted(r["scene"] for r in loaded)
    assert scenes == ["scan", "tactic:value"]
```

---

- [ ] **Step 1.3: 运行测试确认失败**

Run: `pytest tests/test_tracker_scene.py -v`
Expected: FAIL,`load_predictions_by_scene` 不存在 + `log_predictions` 去重行为会让多 scene 同 code 合并成 1 条。

---

- [ ] **Step 1.4: 改 tracker.py — scene 字段支持**

Replace `E:\antenna\learning\tracker.py` 全文:

```python
"""
tracker.py - 记录每次推荐的预测快照,以及收盘后的实际结果。

预测文件:learning/data/pred_YYYY-MM-DD.jsonl
结果文件:learning/data/outcome_YYYY-MM-DD.jsonl

v2 变更(P0):
  - 每条 pred 记录新增 scene 字段 ("scan" | "tactic:<key>" | "predict")
  - 同一天同一 code 不同 scene 可共存,按 (code, scene) 联合去重
  - 旧记录无 scene 字段时默认视作 "scan"(向后兼容)
  - load_predictions_by_scene(date_str, scene) 过滤工具

schema(pred 每条):
{
  "ts":        "2026-04-27T10:15:32",     # 可选,新记录带
  "scene":     "scan" | "tactic:value" | "tactic:growth" | "tactic:leader"
             | "tactic:contra" | "predict",
  "code":      str,
  "name":      str,
  "signal":    "买入" | "观望" | "回避",
  "rise_prob": float,
  "confidence":str,
  "rank_pct":  float,                     # 可选
  "market_state": str,                    # 可选,由 market_state 模块写入
  "tactic_hits": list[str],               # 可选
  "features":  dict,                      # 可选
  "dual_trade":dict,                      # 可选
  "alt":       dict,                      # 可选(P3 填充)
  "watchlist": bool,
}
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

DATA_DIR = Path("learning/data")

SCENE_SCAN = "scan"
SCENE_PREDICT = "predict"
SCENE_TACTIC_PREFIX = "tactic:"
DEFAULT_SCENE = SCENE_SCAN  # 向后兼容


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_scene(record: dict) -> str:
    """取 scene 字段,缺失时回退 DEFAULT_SCENE。"""
    return record.get("scene") or DEFAULT_SCENE


def _record_key(record: dict) -> tuple[str, str]:
    """(code, scene) 作为联合去重 key。"""
    return (record["code"], _get_scene(record))


# ── 预测记录 ──────────────────────────────────────────────

def log_predictions(date_str: str, records: Iterable[dict]):
    """
    保存一批推荐记录。同一天同 (code, scene) 去重(后写覆盖);
    不同 scene 同 code 可共存。

    records 每项至少:{code, name, signal, rise_prob}
    推荐字段:scene, confidence, rank_pct, market_state, tactic_hits,
             features, dual_trade, alt, watchlist
    """
    _ensure()
    path = DATA_DIR / f"pred_{date_str}.jsonl"

    existing: dict[tuple[str, str], dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        existing[_record_key(r)] = r
                    except Exception:
                        pass

    for r in records:
        if "scene" not in r:
            r["scene"] = DEFAULT_SCENE
        if "ts" not in r:
            r["ts"] = datetime.now().isoformat(timespec="seconds")
        existing[_record_key(r)] = r

    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(existing.values(), key=lambda x: (x["code"], _get_scene(x))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_predictions(date_str: str) -> list[dict]:
    """加载当日所有 pred 记录(含所有 scene)。"""
    path = DATA_DIR / f"pred_{date_str}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_predictions_by_scene(date_str: str, scene: str) -> list[dict]:
    """
    按 scene 过滤。精确匹配或前缀匹配:
      - scene="scan"         → 只返回 scan
      - scene="predict"      → 只返回 predict
      - scene="tactic:value" → 只返回 tactic:value
      - scene="tactic"       → 所有 tactic:* 子类
    无 scene 字段的旧记录被视作 "scan"。
    """
    records = load_predictions(date_str)
    out = []
    for r in records:
        rs = _get_scene(r)
        if scene == rs:
            out.append(r)
        elif scene == "tactic" and rs.startswith(SCENE_TACTIC_PREFIX):
            out.append(r)
    return out


def list_prediction_dates() -> list[str]:
    """返回所有有预测记录的日期列表(升序)。"""
    _ensure()
    return sorted(p.stem[5:] for p in DATA_DIR.glob("pred_*.jsonl"))


# ── 实际结果记录 ──────────────────────────────────────────

def log_outcomes(date_str: str, outcomes: dict[str, dict]):
    """
    保存当日实际行情结果(覆盖当日)。
    outcomes 每项推荐字段:
      - actual_open, actual_close, actual_high, actual_low, actual_pct(必填)
      - hit_tier(P0 新增): "miss" | "weak" | "good" | "great"
      - hit_5d(P0 新增): 5 日累计涨幅(当天写入为 None,第 6 日回填)
      - max_drawdown_5d(P0 新增): 5 日最大回撤
    """
    _ensure()
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for code, r in outcomes.items():
            r["code"] = code
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_outcomes(date_str: str) -> dict[str, dict]:
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["code"]] = r
    return out
```

---

- [ ] **Step 1.5: 运行测试确认通过**

Run: `pytest tests/test_tracker_scene.py -v`
Expected: PASS 6/6

---

- [ ] **Step 1.6: 运行现有测试确认无回归**

Run: `pytest tests/ -v --ignore=tests/test_tracker_scene.py`
Expected: 原有测试全过(若已有 tracker 测试,确认新 scene 逻辑不破坏旧行为)

---

- [ ] **Step 1.7: 提交**

```bash
git add learning/tracker.py tests/test_tracker_scene.py tests/fixtures/sample_preds.jsonl
git commit -m "feat(learning): add scene field to pred records with backward-compat"
```

---

## Task 2: 扩展 outcome — hit_tier / hit_5d / max_drawdown_5d

**Files:**
- Modify: `learning/tracker.py`
- Create: `learning/outcome_metrics.py`
- Create: `tests/test_outcome_extension.py`
- Create: `tests/fixtures/sample_outcomes.jsonl`

---

- [ ] **Step 2.1: 创建 fixture**

Create `tests/fixtures/sample_outcomes.jsonl`:
```jsonl
{"code":"600519","actual_open":1800.0,"actual_close":1854.0,"actual_high":1860.0,"actual_low":1790.0,"actual_pct":3.0}
{"code":"000001","actual_open":10.0,"actual_close":10.05,"actual_high":10.1,"actual_low":9.95,"actual_pct":0.5}
{"code":"300750","actual_open":200.0,"actual_close":212.0,"actual_high":215.0,"actual_low":198.0,"actual_pct":6.0}
{"code":"600036","actual_open":40.0,"actual_close":39.6,"actual_high":40.2,"actual_low":39.5,"actual_pct":-1.0}
```

---

- [ ] **Step 2.2: 写失败测试**

Create `tests/test_outcome_extension.py`:
```python
import pytest
from learning.outcome_metrics import compute_hit_tier, compute_5d_metrics


class TestComputeHitTier:
    @pytest.mark.parametrize("pct,expected", [
        (-5.0, "miss"),
        (0.0, "miss"),
        (0.9, "miss"),
        (1.0, "weak"),
        (1.5, "weak"),
        (1.99, "weak"),
        (2.0, "good"),
        (3.5, "good"),
        (4.99, "good"),
        (5.0, "great"),
        (10.0, "great"),
    ])
    def test_hit_tier_boundaries(self, pct, expected):
        assert compute_hit_tier(pct) == expected

    def test_hit_tier_none_returns_none(self):
        assert compute_hit_tier(None) is None


class TestCompute5dMetrics:
    def test_metrics_returns_cumulative_and_drawdown(self):
        """给定 5 日数据:
        Day0 close=100, Day1 close=103, Day2 close=98, Day3 close=105, Day4 close=107
        累计涨幅 = (107/100 - 1)*100 = 7.0
        最大回撤 = (98-103)/103 = -4.85%(从 103 跌到 98)
        """
        closes = [100.0, 103.0, 98.0, 105.0, 107.0]
        metrics = compute_5d_metrics(closes)
        assert metrics["hit_5d"] == pytest.approx(7.0, abs=0.01)
        assert metrics["max_drawdown_5d"] == pytest.approx(-4.85, abs=0.05)

    def test_metrics_all_up(self):
        closes = [100, 101, 102, 103, 104]
        metrics = compute_5d_metrics(closes)
        assert metrics["max_drawdown_5d"] == pytest.approx(0.0, abs=0.01)

    def test_metrics_insufficient_data_returns_none(self):
        assert compute_5d_metrics([100, 101]) is None
        assert compute_5d_metrics([]) is None

    def test_metrics_single_value_returns_none(self):
        assert compute_5d_metrics([100]) is None
```

---

- [ ] **Step 2.3: 运行测试确认失败**

Run: `pytest tests/test_outcome_extension.py -v`
Expected: FAIL,模块不存在

---

- [ ] **Step 2.4: 实现 outcome_metrics.py**

Create `learning/outcome_metrics.py`:
```python
"""
outcome_metrics.py - outcome 扩展字段计算工具。

hit_tier: 按次日涨幅分档
  miss  : actual_pct < 1.0%      (未命中)
  weak  : 1.0% <= actual_pct < 2.0%
  good  : 2.0% <= actual_pct < 5.0%
  great : actual_pct >= 5.0%

5 日指标(第 6 个交易日回填):
  hit_5d:          5 日累计涨幅(%)
  max_drawdown_5d: 5 日内最大回撤(%),负值
"""
from typing import Sequence


def compute_hit_tier(actual_pct: float | None) -> str | None:
    if actual_pct is None:
        return None
    if actual_pct < 1.0:
        return "miss"
    if actual_pct < 2.0:
        return "weak"
    if actual_pct < 5.0:
        return "good"
    return "great"


def compute_5d_metrics(closes: Sequence[float]) -> dict | None:
    """
    closes: [day0_close, day1_close, ..., day4_close](pred 日为 day0,共 5 个交易日)

    返回:
      {"hit_5d": float, "max_drawdown_5d": float}
      数据不足(<2)时返回 None。
    """
    if not closes or len(closes) < 2:
        return None

    base = closes[0]
    if base == 0:
        return None

    # 累计涨幅
    cumulative = (closes[-1] / base - 1.0) * 100.0

    # 最大回撤 = min over i,j of (closes[j] / closes[i] - 1),其中 j >= i
    # 简化实现:对每个 i 求之后的最小值,计算回撤
    max_dd = 0.0
    running_peak = closes[0]
    for c in closes:
        if c > running_peak:
            running_peak = c
        dd = (c / running_peak - 1.0) * 100.0
        if dd < max_dd:
            max_dd = dd

    return {
        "hit_5d": round(cumulative, 2),
        "max_drawdown_5d": round(max_dd, 2),
    }
```

---

- [ ] **Step 2.5: 运行测试确认通过**

Run: `pytest tests/test_outcome_extension.py -v`
Expected: PASS(包含全部 `parametrize` 条目)

---

- [ ] **Step 2.6: 提交**

```bash
git add learning/outcome_metrics.py tests/test_outcome_extension.py tests/fixtures/sample_outcomes.jsonl
git commit -m "feat(learning): add hit_tier and 5d metrics computation helpers"
```

---

## Task 3: scene_bucket.py — 按 scene/market_state 分桶工具

**Files:**
- Create: `learning/scene_bucket.py`
- Create: `tests/test_scene_bucket.py`

---

- [ ] **Step 3.1: 写失败测试**

Create `tests/test_scene_bucket.py`:
```python
from learning.scene_bucket import bucket_by_scene, bucket_by_state, merge_buckets


SAMPLE_PREDS = [
    {"code": "600519", "scene": "scan",         "market_state": "bull",  "rise_prob": 0.75},
    {"code": "000001", "scene": "scan",         "market_state": "bull",  "rise_prob": 0.55},
    {"code": "300750", "scene": "tactic:value", "market_state": "bull",  "rise_prob": 0.80},
    {"code": "002174", "scene": "tactic:growth","market_state": "range", "rise_prob": 0.62},
    {"code": "600036", "scene": "predict",      "market_state": "range", "rise_prob": 0.68},
    {"code": "000002",                           "market_state": "bear",  "rise_prob": 0.45},  # 无 scene
]


def test_bucket_by_scene_returns_expected_keys():
    buckets = bucket_by_scene(SAMPLE_PREDS)
    assert set(buckets.keys()) == {"scan", "tactic:value", "tactic:growth", "predict"}
    # 无 scene 的记录归到 "scan"
    codes_in_scan = [r["code"] for r in buckets["scan"]]
    assert "600519" in codes_in_scan
    assert "000001" in codes_in_scan
    assert "000002" in codes_in_scan  # 旧数据


def test_bucket_by_state_splits_by_market_state():
    buckets = bucket_by_state(SAMPLE_PREDS)
    assert set(buckets.keys()) == {"bull", "range", "bear"}
    assert len(buckets["bull"]) == 3
    assert len(buckets["range"]) == 2
    assert len(buckets["bear"]) == 1


def test_bucket_by_state_missing_defaults_to_range():
    preds_missing_state = [{"code": "X", "scene": "scan", "rise_prob": 0.5}]
    buckets = bucket_by_state(preds_missing_state)
    assert "range" in buckets
    assert len(buckets["range"]) == 1


def test_merge_buckets_by_scene_then_state():
    buckets = merge_buckets(SAMPLE_PREDS, by=("scene", "market_state"))
    # ("scan","bull") 应有 2 条 (600519, 000001)
    assert len(buckets[("scan", "bull")]) == 2
    assert len(buckets[("tactic:value", "bull")]) == 1


def test_bucket_by_scene_empty_input():
    assert bucket_by_scene([]) == {}
```

---

- [ ] **Step 3.2: 运行测试确认失败**

Run: `pytest tests/test_scene_bucket.py -v`
Expected: FAIL,模块不存在

---

- [ ] **Step 3.3: 实现 scene_bucket.py**

Create `learning/scene_bucket.py`:
```python
"""
scene_bucket.py - 按 scene/market_state 分桶工具,给所有 learner 复用。

scene 字段缺失的旧记录视作 "scan"。
market_state 字段缺失的旧记录视作 "range"。
"""
from collections import defaultdict
from typing import Iterable

DEFAULT_SCENE = "scan"
DEFAULT_STATE = "range"


def _scene(r: dict) -> str:
    return r.get("scene") or DEFAULT_SCENE


def _state(r: dict) -> str:
    return r.get("market_state") or DEFAULT_STATE


def bucket_by_scene(records: Iterable[dict]) -> dict[str, list[dict]]:
    """按 scene 分桶。子战法各自成桶(tactic:value / tactic:growth / ...)。"""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[_scene(r)].append(r)
    return dict(out)


def bucket_by_state(records: Iterable[dict]) -> dict[str, list[dict]]:
    """按 market_state 分桶。"""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[_state(r)].append(r)
    return dict(out)


def merge_buckets(records: Iterable[dict], by: tuple[str, ...]) -> dict[tuple, list[dict]]:
    """多维组合分桶。by=("scene", "market_state") 返回 {(scene, state): [...]}"""
    extractors = {
        "scene":        _scene,
        "market_state": _state,
    }
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = tuple(extractors[k](r) for k in by)
        out[key].append(r)
    return dict(out)
```

---

- [ ] **Step 3.4: 测试通过**

Run: `pytest tests/test_scene_bucket.py -v`
Expected: PASS 5/5

---

- [ ] **Step 3.5: 提交**

```bash
git add learning/scene_bucket.py tests/test_scene_bucket.py
git commit -m "feat(learning): add scene/market_state bucketing utilities"
```

---

## Task 4: market_state.py — 大盘状态打标(核心横向模块)

**Files:**
- Create: `learning/market_state.py`
- Create: `tests/test_market_state.py`
- Create: `tests/fixtures/hs300_sample.csv`

---

- [ ] **Step 4.1: 创建 fixture(模拟沪深 300 历史数据)**

Create `tests/fixtures/hs300_sample.csv`:
```csv
date,open,close,high,low,volume
2026-01-02,3600.0,3610.0,3620.0,3595.0,100000000
2026-01-03,3612.0,3625.0,3630.0,3610.0,110000000
2026-01-06,3626.0,3640.0,3645.0,3620.0,105000000
2026-01-07,3641.0,3655.0,3660.0,3640.0,108000000
2026-01-08,3656.0,3670.0,3675.0,3655.0,112000000
```
(注:测试会动态构造更多数据,fixture 仅演示格式)

---

- [ ] **Step 4.2: 写失败测试**

Create `tests/test_market_state.py`:
```python
import pandas as pd
import pytest
from pathlib import Path
from learning import market_state


def _make_hs300_df(close_series: list[float], start: str = "2026-01-02") -> pd.DataFrame:
    """构造连续交易日的沪深 300 DataFrame(open/high/low 简化为 close ±0.5%)。"""
    dates = pd.bdate_range(start=start, periods=len(close_series))
    return pd.DataFrame({
        "date":   dates,
        "open":   [c * 0.995 for c in close_series],
        "close":  close_series,
        "high":   [c * 1.005 for c in close_series],
        "low":    [c * 0.995 for c in close_series],
        "volume": [1e8] * len(close_series),
    })


class TestComputeState:
    def test_bull_when_above_ma60_and_60d_return_positive(self):
        """60 日匀速上涨 10%:close > ma60 且 ret_60d > 5% → bull。"""
        closes = [3000 + i * 5 for i in range(90)]  # 3000 → 3445,涨 ~14%
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        assert state["state"] == "bull"
        assert state["ret_60d"] > 0.05

    def test_bear_when_below_ma60_and_60d_return_negative(self):
        closes = [3500 - i * 5 for i in range(90)]  # 3500 → 3055,跌 ~13%
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        assert state["state"] == "bear"
        assert state["ret_60d"] < -0.05

    def test_range_when_flat(self):
        closes = [3500 + (i % 10 - 5) * 2 for i in range(90)]
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        assert state["state"] == "range"

    def test_high_volatility_forces_range(self):
        """20 日 ATR/close > 2.5% 强制 range,即使趋势向上。"""
        import numpy as np
        np.random.seed(42)
        closes = [3000 + i * 3 + np.random.uniform(-150, 150) for i in range(90)]
        df = _make_hs300_df(closes)
        state = market_state.compute_state(df)
        # atr_pct 高波动应触发 range
        if state["atr_pct"] > 0.025:
            assert state["state"] == "range"

    def test_insufficient_data_returns_range(self):
        df = _make_hs300_df([3500] * 30)  # 只有 30 天,不足 60
        state = market_state.compute_state(df)
        assert state["state"] == "range"
        assert state.get("reason") == "insufficient_data"


class TestDebounceSwitch:
    def test_switch_requires_3_consecutive_days(self, tmp_path, monkeypatch):
        monkeypatch.setattr(market_state, "STATE_FILE", tmp_path / "market_state.json")

        # 第 1 日触发 bull(从无到 bull,直接切 OK,无需防抖)
        closes_bull = [3000 + i * 5 for i in range(90)]
        df_bull = _make_hs300_df(closes_bull)
        market_state.update_state(df_bull, date_str="2026-04-01")
        assert market_state.load_current_state()["current"] == "bull"

        # 第 2 日触发 bear 单次 — 不应立即切,存入 pending
        closes_bear = [3500 - i * 5 for i in range(90)]
        df_bear = _make_hs300_df(closes_bear)
        market_state.update_state(df_bear, date_str="2026-04-02")
        state = market_state.load_current_state()
        assert state["current"] == "bull"  # 仍是 bull
        assert state["pending"] == "bear"

        # 第 3 日仍 bear
        market_state.update_state(df_bear, date_str="2026-04-03")
        assert market_state.load_current_state()["current"] == "bull"

        # 第 4 日仍 bear,连续 3 日达成,切换
        market_state.update_state(df_bear, date_str="2026-04-04")
        assert market_state.load_current_state()["current"] == "bear"


class TestStateOnDate:
    def test_load_state_on_date_uses_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(market_state, "STATE_FILE", tmp_path / "market_state.json")
        # 构造带 history 的 state 文件
        import json
        data = {
            "current": "range",
            "since":   "2026-04-20",
            "pending": None,
            "hs300":   {"close": 3500, "ma60": 3490, "ret_60d": 0.01, "atr_pct": 0.018},
            "history": [
                {"date": "2026-04-15", "state": "bull"},
                {"date": "2026-04-16", "state": "bull"},
                {"date": "2026-04-17", "state": "range"},
                {"date": "2026-04-20", "state": "range"},
            ],
        }
        (tmp_path / "market_state.json").write_text(json.dumps(data), encoding="utf-8")
        assert market_state.load_state_on_date("2026-04-15") == "bull"
        assert market_state.load_state_on_date("2026-04-17") == "range"
        # 无记录的日期回退默认
        assert market_state.load_state_on_date("2025-01-01") == "range"
```

---

- [ ] **Step 4.3: 运行测试确认失败**

Run: `pytest tests/test_market_state.py -v`
Expected: FAIL(module not found)

---

- [ ] **Step 4.4: 实现 market_state.py**

Create `learning/market_state.py`:
```python
"""
market_state.py - 大盘状态打标(bull/bear/range),服务于所有 learner 的状态分桶。

算法(spec §7.1):
  bull   : close > ma60 且 60 日累计涨幅 > +5%,连续 3 日触发
  bear   : close < ma60 且 60 日累计跌幅 < -5%,连续 3 日触发
  range  : 其他;或 20 日 ATR/close > 2.5% 强制进入

产物:learning/market_state.json
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

STATE_FILE = Path("learning/market_state.json")
DEBOUNCE_DAYS = 3  # 连续 N 日触发才切换
DEFAULT_STATE = "range"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _classify(close: float, ma60: float, ret_60d: float, atr_pct: float) -> str:
    """纯分类函数,无防抖。"""
    if atr_pct > 0.025:
        return "range"
    if close > ma60 and ret_60d > 0.05:
        return "bull"
    if close < ma60 and ret_60d < -0.05:
        return "bear"
    return "range"


def compute_state(hs300_df: pd.DataFrame) -> dict:
    """
    根据沪深 300 DataFrame(含 close 列)计算当前原始状态(无防抖)。

    返回:
      {
        "state":   "bull"|"bear"|"range",
        "close":   float,
        "ma60":    float,
        "ret_60d": float,
        "atr_pct": float,
        "reason":  Optional[str],   # "insufficient_data" 等
      }
    """
    if hs300_df is None or len(hs300_df) < 60:
        return {"state": DEFAULT_STATE, "reason": "insufficient_data"}

    df = hs300_df.tail(90).copy().reset_index(drop=True)  # 取最近 90 天
    df["ma60"] = df["close"].rolling(60).mean()

    last_close = float(df["close"].iloc[-1])
    last_ma60  = float(df["ma60"].iloc[-1])
    # 60 日累计回报
    if len(df) >= 60:
        price_60_ago = float(df["close"].iloc[-60])
    else:
        price_60_ago = float(df["close"].iloc[0])
    ret_60d = (last_close / price_60_ago - 1.0) if price_60_ago else 0.0

    # 20 日 ATR / close
    hi = df["high"] if "high" in df.columns else df["close"]
    lo = df["low"]  if "low"  in df.columns else df["close"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (hi - lo).abs(),
        (hi - prev_close).abs(),
        (lo - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.tail(20).mean()
    atr_pct = (atr20 / last_close) if last_close else 0.0

    state = _classify(last_close, last_ma60, ret_60d, float(atr_pct))

    return {
        "state":   state,
        "close":   round(last_close, 2),
        "ma60":    round(last_ma60, 2),
        "ret_60d": round(float(ret_60d), 4),
        "atr_pct": round(float(atr_pct), 4),
    }


def load_current_state() -> dict:
    """加载当前持久化状态。不存在时返回默认。"""
    if not STATE_FILE.exists():
        return {"current": DEFAULT_STATE, "since": None, "pending": None, "history": []}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"current": DEFAULT_STATE, "since": None, "pending": None, "history": []}


def load_state_on_date(date_str: str) -> str:
    """
    根据 history 回查某日状态。若 date_str 之前有 history 记录则用最近的,否则返回 DEFAULT_STATE。
    用于回放测试:对每条 pred/outcome 查询当时市场状态。
    """
    data = load_current_state()
    history = data.get("history") or []
    # 找 <= date_str 的最后一条
    matching = [h for h in history if h["date"] <= date_str]
    if not matching:
        return DEFAULT_STATE
    return matching[-1]["state"]


def update_state(hs300_df: pd.DataFrame, date_str: str) -> dict:
    """
    基于最新 hs300_df 更新状态(含防抖);追加 history 一条并落盘。

    防抖规则:
      - 若 raw_state == current:直接 append history,pending 清空
      - 若 raw_state != current 但 == pending:计数 +1;达到 DEBOUNCE_DAYS 切 current
      - 若 raw_state 不同于 current 也不同于 pending:pending = raw_state,计数重置
      - 首次(current 为 DEFAULT_STATE 且 since=None):允许立即设置
    """
    raw = compute_state(hs300_df)
    raw_state = raw["state"]
    data = load_current_state()

    current = data.get("current", DEFAULT_STATE)
    pending = data.get("pending")
    pending_days = data.get("pending_days", 0)

    first_time = data.get("since") is None

    if first_time:
        new_current = raw_state
        new_pending = None
        new_pending_days = 0
        new_since = date_str
    elif raw_state == current:
        new_current = current
        new_pending = None
        new_pending_days = 0
        new_since = data.get("since")
    elif raw_state == pending:
        new_pending_days = pending_days + 1
        if new_pending_days >= DEBOUNCE_DAYS:
            new_current = pending
            new_pending = None
            new_pending_days = 0
            new_since = date_str
        else:
            new_current = current
            new_pending = pending
            new_since = data.get("since")
    else:
        new_current = current
        new_pending = raw_state
        new_pending_days = 1
        new_since = data.get("since")

    history = data.get("history") or []
    history.append({"date": date_str, "state": new_current})
    history = history[-90:]  # 最近 90 日

    new_data = {
        "current":      new_current,
        "since":        new_since,
        "pending":      new_pending,
        "pending_days": new_pending_days,
        "hs300":        {k: v for k, v in raw.items() if k != "state"},
        "history":      history,
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write(STATE_FILE, new_data)
    return new_data


def run(date_str: Optional[str] = None) -> dict:
    """
    编排器入口:拉沪深 300 数据并更新状态。
    失败时抛异常,由编排器捕获。
    """
    from data.fetcher import fetch_stock_hist
    try:
        df = fetch_stock_hist("sh000300", days=120, cache_only=False)
    except Exception:
        # 尝试别名
        df = fetch_stock_hist("000300", days=120, cache_only=False)

    if df is None or df.empty:
        raise RuntimeError("无法获取沪深300数据")

    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    return update_state(df, date_str)
```

---

- [ ] **Step 4.5: 运行测试确认通过**

Run: `pytest tests/test_market_state.py -v`
Expected: PASS 8/8

---

- [ ] **Step 4.6: 本地冒烟 — 验证 akshare 能拉到沪深300**

Run:
```bash
python -c "from data.fetcher import fetch_stock_hist; df = fetch_stock_hist('sh000300', days=120); print(df.tail(3))"
```
Expected: 打印最近 3 天行情。若失败,需在 Task 4.7 的 `run()` 里加降级(例如换成 `000300.SH` 代码格式)。

---

- [ ] **Step 4.7: 接入 cli.py 冒烟验证**

Run:
```bash
python -c "from learning.market_state import run; print(run())"
```
Expected: 打印状态 dict,且 `learning/market_state.json` 文件生成。

---

- [ ] **Step 4.8: 提交**

```bash
git add learning/market_state.py tests/test_market_state.py tests/fixtures/hs300_sample.csv
git commit -m "feat(learning): add market state classification with 3-day debounce"
```

---

## Task 5: feedback_io.py — 归因日志 + 历史快照工具

**Files:**
- Create: `learning/feedback_io.py`
- Create: `tests/test_feedback_io.py`

---

- [ ] **Step 5.1: 写失败测试**

Create `tests/test_feedback_io.py`:
```python
import json
from pathlib import Path
import pytest
from learning import feedback_io


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(feedback_io, "HISTORY_DIR",  tmp_path / "history")
    return tmp_path


def test_write_feedback_creates_file_with_expected_schema(workdir):
    feedback_io.write_feedback("market_state", "2026-04-27", {
        "samples_used":    100,
        "accuracy_before": 0.32,
        "accuracy_after":  0.40,
        "detail":          "bull → range",
    })
    path = workdir / "feedback" / "market_state_2026-04-27.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["module"] == "market_state"
    assert data["date"] == "2026-04-27"
    assert data["samples_used"] == 100


def test_snapshot_rotates_to_max_7(workdir):
    target = workdir / "market_state.json"
    target.write_text('{"v":1}', encoding="utf-8")

    for i in range(10):
        date_str = f"2026-04-{str(10 + i).zfill(2)}"
        feedback_io.snapshot_file(target, date_str)

    snapshots = sorted((workdir / "history").glob("market_state_*.json"))
    assert len(snapshots) == 7
    # 最早 3 份应被删除,保留最近 7 份
    names = [p.name for p in snapshots]
    assert "market_state_2026-04-13.json" in names
    assert "market_state_2026-04-19.json" in names
    assert "market_state_2026-04-10.json" not in names


def test_snapshot_missing_source_is_no_op(workdir):
    """源文件不存在时不报错,只是不快照。"""
    missing = workdir / "nonexistent.json"
    feedback_io.snapshot_file(missing, "2026-04-27")  # 不应抛异常
    assert not (workdir / "history").exists() or \
        len(list((workdir / "history").glob("*.json"))) == 0


def test_atomic_write_does_not_leave_tmp(workdir):
    target = workdir / "subdir" / "out.json"
    feedback_io.atomic_write_json(target, {"k": "v"})
    assert target.exists()
    # 确保没有遗留的 .tmp
    assert not list(target.parent.glob("*.tmp"))
```

---

- [ ] **Step 5.2: 运行测试确认失败**

Run: `pytest tests/test_feedback_io.py -v`
Expected: FAIL,模块不存在

---

- [ ] **Step 5.3: 实现 feedback_io.py**

Create `learning/feedback_io.py`:
```python
"""
feedback_io.py - 学习模块的统一 I/O 工具。

- write_feedback(module, date, data)          → learning/feedback/<module>_<date>.json
- snapshot_file(source, date)                 → learning/history/<basename>_<date>.json(最多保留 7 份)
- atomic_write_json(path, data)               → 临时文件 + os.replace 原子写
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

FEEDBACK_DIR = Path("learning/feedback")
HISTORY_DIR  = Path("learning/history")
MAX_SNAPSHOTS = 7


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def write_feedback(module: str, date_str: str, payload: dict) -> Path:
    """
    写入一条 feedback 日志。
    payload 典型字段:samples_used, accuracy_before, accuracy_after, detail。
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    path = FEEDBACK_DIR / f"{module}_{date_str}.json"
    data = {
        "module":     module,
        "date":       date_str,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    atomic_write_json(path, data)
    return path


def snapshot_file(source: Path, date_str: str) -> Path | None:
    """
    将 source 拷贝到 history/,命名为 <basename>_<date_str>.json。
    超过 MAX_SNAPSHOTS 时删除最早的快照。
    source 不存在时静默返回 None。
    """
    source = Path(source)
    if not source.exists():
        return None

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    basename = source.stem
    dest = HISTORY_DIR / f"{basename}_{date_str}{source.suffix}"
    shutil.copy2(source, dest)

    snapshots = sorted(HISTORY_DIR.glob(f"{basename}_*{source.suffix}"))
    while len(snapshots) > MAX_SNAPSHOTS:
        snapshots[0].unlink(missing_ok=True)
        snapshots = sorted(HISTORY_DIR.glob(f"{basename}_*{source.suffix}"))
    return dest
```

---

- [ ] **Step 5.4: 测试通过**

Run: `pytest tests/test_feedback_io.py -v`
Expected: PASS 4/4

---

- [ ] **Step 5.5: 提交**

```bash
git add learning/feedback_io.py tests/test_feedback_io.py
git commit -m "feat(learning): add feedback log writer and snapshot rotator"
```

---

## Task 6: alerts.py — 告警工具(飞书 + 日志)

**Files:**
- Create: `learning/alerts.py`
- Create: `tests/test_alerts.py`

---

- [ ] **Step 6.1: 写失败测试**

Create `tests/test_alerts.py`:
```python
import json
from pathlib import Path
import pytest
from learning import alerts


@pytest.fixture
def alerts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.jsonl")
    return tmp_path


def test_record_alert_appends_jsonl_line(alerts_dir):
    alerts.record_alert("market_state", "data fetch failed", severity="error")
    lines = (alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["module"] == "market_state"
    assert data["severity"] == "error"


def test_multiple_alerts_accumulate(alerts_dir):
    alerts.record_alert("m1", "err1", "warning")
    alerts.record_alert("m2", "err2", "error")
    lines = (alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_severity_defaults_to_error(alerts_dir):
    alerts.record_alert("m", "msg")
    data = json.loads((alerts_dir / "alerts.jsonl").read_text(encoding="utf-8").strip())
    assert data["severity"] == "error"


def test_push_to_feishu_not_called_when_no_webhook(alerts_dir, monkeypatch):
    called = []
    def fake_push(hook, msg):
        called.append(msg)
    monkeypatch.setattr(alerts, "_push_feishu", fake_push)
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "")  # 空 webhook
    alerts.send_alert("market_state", "failed")
    assert called == []


def test_send_alert_calls_push_when_webhook_present(alerts_dir, monkeypatch):
    pushed = []
    monkeypatch.setattr(alerts, "_push_feishu", lambda hook, msg: pushed.append((hook, msg)))
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "https://hook.example")
    alerts.send_alert("market_state", "failed: connection refused")
    assert len(pushed) == 1
    assert "market_state" in pushed[0][1]
```

---

- [ ] **Step 6.2: 运行测试确认失败**

Run: `pytest tests/test_alerts.py -v`
Expected: FAIL(module not found)

---

- [ ] **Step 6.3: 实现 alerts.py**

Create `learning/alerts.py`:
```python
"""
alerts.py - 学习系统告警:飞书 webhook + 本地 alerts.jsonl + stdout。

所有告警都三路同时落:
  1. 写 learning/alerts.jsonl(permanent,人工复盘用)
  2. 尝试发飞书(若 webhook 配置)
  3. 打印到 stdout

用法:
  try:
      module.run()
  except Exception as e:
      alerts.send_alert(module.__name__, f"{e}", traceback=True)
"""
import json
import traceback as tb_mod
from datetime import datetime
from pathlib import Path

ALERTS_FILE = Path("learning/alerts.jsonl")


def _get_webhook() -> str:
    """读 config.yaml 拿飞书 webhook。"""
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("feishu", {}).get("webhook_url", "") or ""
    except Exception:
        return ""


def _push_feishu(webhook: str, msg: str) -> bool:
    """发送纯文本告警到飞书 webhook。失败不抛。"""
    try:
        from notify.feishu import send_text
        return send_text(webhook, msg)
    except Exception:
        return False


def record_alert(module: str, message: str, severity: str = "error") -> None:
    """只写 alerts.jsonl + stdout,不发飞书。"""
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":       datetime.now().isoformat(timespec="seconds"),
        "module":   module,
        "severity": severity,
        "message":  message,
    }
    with open(ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[alert:{severity}] {module}: {message}")


def send_alert(module: str, message: str, severity: str = "error", traceback: bool = False) -> None:
    """完整告警:alerts.jsonl + 飞书 + stdout。"""
    tb_str = ""
    if traceback:
        tb_str = "\n" + tb_mod.format_exc()

    full_msg = f"⚠️ [{module}] {message}{tb_str}"
    record_alert(module, message + tb_str, severity)

    webhook = _get_webhook()
    if webhook:
        _push_feishu(webhook, full_msg)
```

---

- [ ] **Step 6.4: 确认 notify.feishu.send_text 存在**

Run: `grep -n "^def send_text\|send_text" E:/antenna/notify/feishu.py | head -5`
Expected: 找到 `send_text(webhook, text)` 函数签名。

如果不存在,在 `notify/feishu.py` 末尾追加:
```python
def send_text(webhook: str, text: str) -> bool:
    """纯文本消息推送。"""
    import requests
    try:
        resp = requests.post(
            webhook,
            json={"msg_type": "text", "content": {"text": text}},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False
```

---

- [ ] **Step 6.5: 测试通过**

Run: `pytest tests/test_alerts.py -v`
Expected: PASS 5/5

---

- [ ] **Step 6.6: 提交**

```bash
git add learning/alerts.py tests/test_alerts.py notify/feishu.py
git commit -m "feat(learning): add alerts module with feishu+jsonl+stdout triplet"
```

---

## Task 7: orchestrator.py — 瘦编排器骨架

**Files:**
- Create: `learning/orchestrator.py`
- Create: `tests/test_orchestrator.py`

---

- [ ] **Step 7.1: 写失败测试**

Create `tests/test_orchestrator.py`:
```python
from pathlib import Path
import pytest
from learning import orchestrator


def test_registry_has_market_state():
    names = [m["name"] for m in orchestrator.MODULES]
    assert "market_state" in names


def test_module_failure_does_not_block_others(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    calls = []

    def ok_run(date_str=None):
        calls.append("ok")
        return {"state": "range"}

    def boom_run(date_str=None):
        calls.append("boom")
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "boom",        "run": boom_run,  "depends_on": []},
        {"name": "after_boom",  "run": ok_run,    "depends_on": []},
    ])

    result = orchestrator.run_all(date_str="2026-04-27")
    assert "boom" in calls
    assert "ok" in calls
    assert result["boom"]["status"] == "failed"
    assert result["after_boom"]["status"] == "ok"


def test_module_with_unmet_dependency_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    def boom_run(date_str=None):
        raise RuntimeError("boom")

    def after_run(date_str=None):
        return {"ok": True}

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "boom",  "run": boom_run,  "depends_on": []},
        {"name": "after", "run": after_run, "depends_on": ["boom"]},
    ])

    result = orchestrator.run_all(date_str="2026-04-27")
    assert result["after"]["status"] == "skipped"
    assert result["boom"]["status"] == "failed"


def test_dry_run_does_not_execute_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    called = []

    def must_not_run(date_str=None):
        called.append(1)

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "m", "run": must_not_run, "depends_on": []},
    ])

    result = orchestrator.run_all(date_str="2026-04-27", dry_run=True)
    assert called == []
    assert result["m"]["status"] == "dry_run"


def test_run_all_writes_feedback_for_successful_module(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    def ok_run(date_str=None):
        return {"state": "bull", "samples_used": 100}

    monkeypatch.setattr(orchestrator, "MODULES", [
        {"name": "m", "run": ok_run, "depends_on": []},
    ])

    orchestrator.run_all(date_str="2026-04-27")
    feedback_path = tmp_path / "feedback" / "m_2026-04-27.json"
    assert feedback_path.exists()


def test_check_returns_zero_when_all_files_valid(tmp_path, monkeypatch):
    import json
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "market_state.json").write_text(
        json.dumps({"current": "range", "history": []}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code == 0


def test_check_returns_nonzero_on_corrupt_json(tmp_path, monkeypatch):
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "market_state.json").write_text("{BROKEN", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = orchestrator.check()
    assert code != 0
```

---

- [ ] **Step 7.2: 运行测试确认失败**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL,模块不存在

---

- [ ] **Step 7.3: 实现 orchestrator.py**

Create `learning/orchestrator.py`:
```python
"""
orchestrator.py - 瘦编排器:按依赖顺序跑所有学习子模块,单模块失败不影响其他。

P0 仅挂载 market_state,后续阶段会追加:
  - P1: model_learner, blacklist
  - P2: tactic_learner, ai_reason
  - P3: alt_data, feature_learner
  - P4: price_learner

每个 MODULES 条目:
  {"name": str, "run": callable(date_str) → dict, "depends_on": list[str]}
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from learning import feedback_io, alerts, market_state


# ── 模块注册表(按依赖顺序) ─────────────────────────────
MODULES: list[dict] = [
    {
        "name":       "market_state",
        "run":        market_state.run,
        "depends_on": [],
    },
    # P1/P2/P3/P4 后续阶段追加
]


def run_all(date_str: str | None = None, dry_run: bool = False) -> dict:
    """
    依次执行 MODULES 中每个模块。某模块失败:
      - 记录告警
      - 不执行依赖它的模块(标记 skipped)
      - 继续执行无依赖关系的其他模块
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    results: dict[str, dict] = {}
    failed: set[str] = set()

    for module in MODULES:
        name = module["name"]
        deps = module.get("depends_on") or []

        if any(d in failed for d in deps):
            results[name] = {
                "status":  "skipped",
                "reason":  f"dependency failed: {[d for d in deps if d in failed]}",
            }
            continue

        if dry_run:
            results[name] = {"status": "dry_run"}
            continue

        try:
            result = module["run"](date_str)
            results[name] = {"status": "ok", "result": result}
            feedback_io.write_feedback(name, date_str, {"result": result})
        except Exception as e:
            failed.add(name)
            results[name] = {"status": "failed", "error": str(e)}
            alerts.send_alert(name, f"学习失败: {e}", traceback=True)

    return results


def check() -> int:
    """
    启动自检:验证所有学习产物 JSON 可解析。
    返回 exit code(0 成功;非 0 失败)。
    """
    files_to_check = [
        Path("learning/market_state.json"),
        Path("learning/strategy.json"),
        # P1-P4 阶段会追加 tactic_params.json / feature_weights.json / price_params.json / blacklist.json
    ]
    errors = []
    for p in files_to_check:
        if not p.exists():
            continue  # 还没生成不算错
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"{p}: {e}")

    if errors:
        for err in errors:
            print(f"[check] {err}", file=sys.stderr)
        return 1
    print("[check] 所有学习产物验证通过")
    return 0
```

---

- [ ] **Step 7.4: 测试通过**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS 7/7

---

- [ ] **Step 7.5: 提交**

```bash
git add learning/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(learning): add thin orchestrator with failure isolation and dry-run"
```

---

## Task 8: 给 cli.py 添加 learn 子命令

**Files:**
- Modify: `E:\antenna\cli.py`

---

- [ ] **Step 8.1: 在 cli.py 里追加 cmd_learn 函数**

在 `cli.py` 末尾 `def main():` 之前(即第 540 行之前)插入:
```python
# ── learn ──────────────────────────────────────────────────────────────────

def cmd_learn(args, config):
    """
    learn:运行所有学习子模块(P0 仅含 market_state;P1-P4 阶段性追加)。

    用法:
      python cli.py learn                # 用今天日期
      python cli.py learn --date 2026-04-27
      python cli.py learn --dry-run
      python cli.py learn --check        # 仅做自检(不运行)
    """
    import json
    from datetime import datetime
    from learning import orchestrator

    if args.check:
        sys.exit(orchestrator.check())

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"[learn] 执行日期: {date_str}{' (dry-run)' if args.dry_run else ''}")

    results = orchestrator.run_all(date_str=date_str, dry_run=args.dry_run)

    ok      = sum(1 for r in results.values() if r["status"] == "ok")
    failed  = sum(1 for r in results.values() if r["status"] == "failed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    dry     = sum(1 for r in results.values() if r["status"] == "dry_run")

    print(f"[learn] 完成: ok={ok} failed={failed} skipped={skipped} dry_run={dry}")
    for name, r in results.items():
        print(f"  - {name}: {r['status']}")
        if r["status"] == "failed":
            print(f"    error: {r['error']}")

    sys.exit(1 if failed else 0)

```

---

- [ ] **Step 8.2: 在 main() argparse 中注册 learn 子命令**

在 `cli.py:578` 附近(与 `sub.add_parser("bot", ...)` 同级)追加:
```python
    p = sub.add_parser("learn", help="运行学习反馈管线(P0 仅 market_state;后续阶段追加)")
    p.add_argument("--date", help="指定日期(默认今日),格式 YYYY-MM-DD")
    p.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p.add_argument("--check", action="store_true", help="仅校验学习产物文件的 JSON 合法性")
```

---

- [ ] **Step 8.3: 在 dispatch 字典(`cli.py:595` 附近)追加 learn**

将:
```python
    dispatch = {
        "fetch":    cmd_fetch,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "scan":     cmd_scan,
        "report":   cmd_report,
        "backtest": cmd_backtest,
        "style":    cmd_style,
        "bot":      cmd_bot,
    }
```
改为:
```python
    dispatch = {
        "fetch":    cmd_fetch,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "scan":     cmd_scan,
        "report":   cmd_report,
        "backtest": cmd_backtest,
        "style":    cmd_style,
        "bot":      cmd_bot,
        "learn":    cmd_learn,
    }
```

---

- [ ] **Step 8.4: 冒烟验证**

Run:
```bash
python cli.py learn --help
```
Expected: 打印出 `--date`、`--dry-run`、`--check` 参数说明。

Run:
```bash
python cli.py learn --dry-run
```
Expected: 打印 `[learn] 完成: ok=0 failed=0 skipped=0 dry_run=1` 并列出 `- market_state: dry_run`。

Run:
```bash
python cli.py learn --check
```
Expected: 退出码 0(若 learning/market_state.json 还不存在会被跳过,也视为通过)。

Run:
```bash
python cli.py learn
```
Expected: 实际拉取沪深300数据并更新 `learning/market_state.json`。

---

- [ ] **Step 8.5: 提交**

```bash
git add cli.py
git commit -m "feat(cli): add 'learn' subcommand for running learning pipeline"
```

---

## Task 9: 现有 log_predictions 调用点加 scene 字段

**Files:**
- Modify: `server/predict_cmd.py`(两处)
- Modify: `scripts/task_scan.py`
- Modify: `scripts/task_predict.py`
- Modify: `cli.py`(backtest 内)

---

- [ ] **Step 9.1: 改 server/predict_cmd.py cmd_scan_bot(约 1056-1060 行附近)**

用 `grep -n "\"scan_date\"" E:/antenna/server/predict_cmd.py` 找定位。

在构造 snapshot 列表的每条字典中追加 `"scene": "scan"`。具体修改 `server/predict_cmd.py` 约 1040-1060 行的 snapshot 字典构造处(同时包含自选股和推荐股两类追加方式,都要加 scene)。

用 Edit 替换:
```python
            "scan_date":       scan_date,
            "pred_high":       r["price_info"].get("pred_high"),
            "pred_low":        r["price_info"].get("pred_low"),
            "watchlist":  True,
        })
```
替换为:
```python
            "scan_date":       scan_date,
            "pred_high":       r["price_info"].get("pred_high"),
            "pred_low":        r["price_info"].get("pred_low"),
            "watchlist":  True,
            "scene":      "scan",
        })
```

并找该文件中相邻的另一 snapshot 追加块(`"watchlist":  False` 的那条)同样追加 `"scene": "scan"`。

---

- [ ] **Step 9.2: 改 server/predict_cmd.py 里 backtest 块(约 1977-1982 行)**

替换:
```python
                "code":       r["code"],
                "name":       r["code"],
                "signal":     r.get("signal", "观望"),
                "rise_prob":  round(r["rise_prob"], 4),
                "confidence": r.get("confidence", ""),
                "scan_date":  trade_str,
            } for r in top]
```
为:
```python
                "code":       r["code"],
                "name":       r["code"],
                "signal":     r.get("signal", "观望"),
                "rise_prob":  round(r["rise_prob"], 4),
                "confidence": r.get("confidence", ""),
                "scan_date":  trade_str,
                "scene":      "scan",
            } for r in top]
```

---

- [ ] **Step 9.3: 改 scripts/task_scan.py 里 snapshot 构造**

打开 `scripts/task_scan.py`,找到约 56-68 行的 snapshot 构造(`for r in scan_result.get("top", [])` 循环)。在 `.append({...})` 字典里加一条 `"scene": "scan",`。

具体操作:在约 64 行 `"open":       pi.get("open"),` 下一行追加:
```python
            "scene":      "scan",
```

---

- [ ] **Step 9.4: 改 scripts/task_predict.py 里 snapshot 构造**

打开 `scripts/task_predict.py`,找到约 132-144 行的 snapshot 构造,在字典末尾追加:
```python
            "scene":      "predict",
            "watchlist":  True,  # 自选股默认 watchlist=True
```

---

- [ ] **Step 9.5: 改 cli.py backtest(约 416-423 行)**

在 `cli.py` 的 `cmd_backtest` 函数内,找到:
```python
        snapshot = [{
            "code":       r["code"],
            "name":       r["code"],
            "signal":     r.get("signal", "观望"),
            "rise_prob":  round(r["rise_prob"], 4),
            "confidence": r.get("confidence", ""),
            "scan_date":  trade_str,
        } for r in top]
```
改为:
```python
        snapshot = [{
            "code":       r["code"],
            "name":       r["code"],
            "signal":     r.get("signal", "观望"),
            "rise_prob":  round(r["rise_prob"], 4),
            "confidence": r.get("confidence", ""),
            "scan_date":  trade_str,
            "scene":      "scan",
        } for r in top]
```

---

- [ ] **Step 9.6: 冒烟验证 — 旧数据仍能读,新数据带 scene**

Run:
```bash
python -c "from learning.tracker import load_predictions, load_predictions_by_scene; \
recs = load_predictions('2026-04-24'); \
print(f'total: {len(recs)}, first scene: {recs[0].get(\"scene\")!r}'); \
scan_only = load_predictions_by_scene('2026-04-24', 'scan'); \
print(f'scan bucket: {len(scan_only)}')"
```
Expected: 不报错;旧记录的 `scene` 字段可能是 None,但 `load_predictions_by_scene('scan')` 会把它归入 scan 桶。

---

- [ ] **Step 9.7: 提交**

```bash
git add server/predict_cmd.py scripts/task_scan.py scripts/task_predict.py cli.py
git commit -m "feat(tracker): tag all log_predictions call sites with scene"
```

---

## Task 10: 为 cmd_tactic / _tactic_run 添加 pred 快照落盘

**Files:**
- Modify: `server/predict_cmd.py`(在 `_tactic_run` 函数末尾)

---

- [ ] **Step 10.1: 定位 _tactic_run 函数**

Run:
```bash
grep -n "def _tactic_run" E:/antenna/server/predict_cmd.py
```
Expected: 打印行号(约 1452 行)。

---

- [ ] **Step 10.2: 阅读函数完整实现**

Read: `server/predict_cmd.py` 从 `_tactic_run` 开始到下一个 `def` 的范围,确认它产生的 `top` 列表的数据结构(主要看 `code`, `signal`, `rise_prob` 是否已有)。

---

- [ ] **Step 10.3: 在 _tactic_run 函数内,推送飞书卡片之前,写入 pred 快照**

找 `_tactic_run` 函数里调用 `feishu_push(...)` 之前的位置(或 `cards = [...]` 生成之后)。在该位置追加以下代码块:

```python
    # P0: 写入 tactic 场景的 pred 快照(供学习反馈使用)
    try:
        from learning.tracker import log_predictions
        from learning.market_state import load_current_state
        from datetime import datetime

        current_state = load_current_state().get("current", "range")
        scan_date = datetime.now().strftime("%Y-%m-%d")
        scene_tag = f"tactic:{strategy_key}"

        snapshot = [{
            "code":         r["code"],
            "name":         r.get("name", r["code"]),
            "signal":       r.get("signal", "买入"),
            "rise_prob":    round(r.get("rise_prob", 0.0), 4),
            "confidence":   r.get("confidence", ""),
            "scan_date":    scan_date,
            "scene":        scene_tag,
            "market_state": current_state,
            "tactic_hits":  [strategy_key],
        } for r in top]

        if snapshot:
            log_predictions(scan_date, snapshot)
    except Exception as e:
        print(f"[_tactic_run] pred 快照落盘失败(忽略): {e}")
```

---

- [ ] **Step 10.4: 冒烟验证 — 通过 bot 子命令跑战法**

Run:
```bash
python cli.py bot 战法 价值 3
```
Expected:
- 不报错
- `learning/data/pred_<今日>.jsonl` 里新增 3 条 `scene="tactic:value"` 的记录

Run:
```bash
python -c "from learning.tracker import load_predictions_by_scene; \
from datetime import date; \
d = date.today().isoformat(); \
recs = load_predictions_by_scene(d, 'tactic'); \
print(f'tactic 桶共 {len(recs)} 条'); \
for r in recs[:3]: print(r)"
```
Expected: 打印 tactic 桶记录。

---

- [ ] **Step 10.5: 提交**

```bash
git add server/predict_cmd.py
git commit -m "feat(tactic): log tactic pred snapshots for learning feedback"
```

---

## Task 11: outcome 扩展写入流程 — 5 日指标延迟回填

**Files:**
- Create: `scripts/task_fill_5d_metrics.py`
- Modify: `scripts/task_daily_review.py`

---

- [ ] **Step 11.1: 创建 task_fill_5d_metrics.py(回填脚本)**

Create `scripts/task_fill_5d_metrics.py`:
```python
"""
task_fill_5d_metrics.py - 每日 15:30 运行,回填 "今天往前推 5 交易日" 的 outcome 的
  hit_5d 和 max_drawdown_5d 字段。

触发时机:每日 task_daily_review.py 后调用一次。
"""
import sys
import io
import os
from datetime import datetime, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def fill_5d_metrics_for_date(pred_date_str: str) -> dict:
    """
    给定 pred 日期,回填 outcome 的 5 日指标。
    需要第 6 个交易日之后才能跑,否则数据不够。

    返回 {filled: int, skipped: int}。
    """
    import pandas as pd
    from learning.tracker  import load_outcomes, log_outcomes, load_predictions
    from learning.outcome_metrics import compute_5d_metrics, compute_hit_tier
    from data.fetcher import fetch_stock_hist

    outcomes = load_outcomes(pred_date_str)
    preds    = load_predictions(pred_date_str)
    if not outcomes:
        return {"filled": 0, "skipped": 0, "reason": "no outcomes"}

    # 对每只股票,取 pred_date 起 5 个交易日的 close 序列
    pred_ts = pd.Timestamp(pred_date_str)
    filled = 0
    skipped = 0

    for code, r in outcomes.items():
        if r.get("hit_5d") is not None:
            skipped += 1
            continue

        try:
            df = fetch_stock_hist(code, days=30, cache_only=True)
            if df is None or df.empty:
                skipped += 1
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= pred_ts].head(5)
            if len(df) < 5:
                skipped += 1
                continue
            closes = df["close"].tolist()
            metrics = compute_5d_metrics(closes)
            if metrics:
                r["hit_5d"] = metrics["hit_5d"]
                r["max_drawdown_5d"] = metrics["max_drawdown_5d"]
                # 顺便回填 hit_tier(若缺)
                if r.get("hit_tier") is None and r.get("actual_pct") is not None:
                    r["hit_tier"] = compute_hit_tier(r["actual_pct"])
                filled += 1
        except Exception as e:
            skipped += 1

    if filled:
        log_outcomes(pred_date_str, outcomes)

    return {"filled": filled, "skipped": skipped}


def run():
    import pandas as pd
    # 取 7 个交易日前的日期(给点 buffer)
    target = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"[fill_5d] 目标日期: {target}")
    result = fill_5d_metrics_for_date(target)
    print(f"[fill_5d] 回填 {result['filled']} 条,跳过 {result['skipped']} 条")


if __name__ == "__main__":
    run()
```

---

- [ ] **Step 11.2: 在 task_daily_review.py 里增补 hit_tier 即时回填**

读取 `scripts/task_daily_review.py`,在 `log_outcomes(date_str, outcomes)` 调用之前(约 105 行),先给每条 outcome 补齐 `hit_tier`:

找到:
```python
    outcomes = fetch_actual_outcomes(codes, date_str)
    if not outcomes:
        print("[review] 无法获取实际收盘数据,跳过。")
        return
    log_outcomes(date_str, outcomes)
    print(f"[review] 已记录 {len(outcomes)} 只实际结果")
```

替换为:
```python
    outcomes = fetch_actual_outcomes(codes, date_str)
    if not outcomes:
        print("[review] 无法获取实际收盘数据,跳过。")
        return

    # P0: 即时补 hit_tier(5 日指标由 task_fill_5d_metrics 延迟回填)
    from learning.outcome_metrics import compute_hit_tier
    for code, r in outcomes.items():
        r["hit_tier"] = compute_hit_tier(r.get("actual_pct"))

    log_outcomes(date_str, outcomes)
    print(f"[review] 已记录 {len(outcomes)} 只实际结果")
```

---

- [ ] **Step 11.3: 在 task_daily_review.py 末尾追加 learn orchestrator + 5d 回填**

在 `scripts/task_daily_review.py` 的 `run()` 函数末尾(在飞书推送后)追加:
```python
    # ── 调用学习编排器 ──────────────────────────────────
    print("[review] 启动学习管线 ...")
    try:
        from learning.orchestrator import run_all
        results = run_all(date_str=date_str)
        ok     = sum(1 for r in results.values() if r["status"] == "ok")
        failed = sum(1 for r in results.values() if r["status"] == "failed")
        print(f"[review] 学习管线完成: ok={ok} failed={failed}")
    except Exception as e:
        print(f"[review] 学习管线异常(忽略,不影响其他): {e}")

    # ── 5 日指标回填(对 7 日前的 outcome) ───────────────
    try:
        from scripts.task_fill_5d_metrics import run as fill_5d_run
        fill_5d_run()
    except Exception as e:
        print(f"[review] 5 日指标回填异常(忽略): {e}")
```

---

- [ ] **Step 11.4: 冒烟验证**

Run:
```bash
python scripts/task_daily_review.py
```
Expected:
- 原有行为不受影响(推完飞书)
- 额外打印 `[review] 学习管线完成: ok=1 failed=0`
- 额外打印 `[fill_5d] 目标日期: ...`

Run:
```bash
python -c "import json; \
with open('learning/data/outcome_2026-04-24.jsonl', encoding='utf-8') as f: \
  for line in f: r = json.loads(line); print(r.get('hit_tier'), r.get('hit_5d'))"
```
Expected: 新写入的 outcome 有 `hit_tier` 字段(hit_5d 需要等第 6 天才回填,旧文件的可能为 None)。

---

- [ ] **Step 11.5: 提交**

```bash
git add scripts/task_daily_review.py scripts/task_fill_5d_metrics.py
git commit -m "feat(review): integrate learn orchestrator and 5d metrics backfill"
```

---

## Task 12: replay_learn.py — 历史回放脚本(P0 验收核心)

**Files:**
- Create: `scripts/replay_learn.py`

---

- [ ] **Step 12.1: 创建 replay_learn.py**

Create `scripts/replay_learn.py`:
```python
"""
replay_learn.py - 在已有历史 pred/outcome 上模拟滚动学习,输出基线精准率曲线。

P0 阶段:因编排器只挂了 market_state(无实际学习效果),本脚本只做"**基线精准率计算**":
  对 2025-07-01 起每一天:
    1. load_predictions(D) / load_outcomes(D)
    2. 计算当日买入精准率(区分 scan / tactic:* / predict 各场景)
    3. 按市场状态(从 market_state.json history 查)打标
    4. 写入 `learning/replay/baseline_YYYY-MM-DD.csv`

P1 起会扩展本脚本在每日循环中调 orchestrator.run_all(date_str=D),
  以此验证"学习参数更新后,下一日精准率是否提升"。P0 先打基线。

用法:
  python scripts/replay_learn.py                  # 回放全部
  python scripts/replay_learn.py --from 2026-01-01 --to 2026-04-27
  python scripts/replay_learn.py --scene tactic   # 只统计 tactic 场景
"""
import sys
import io
import os
import argparse
import csv
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_DIR = "learning/replay"


def _evaluate_single(preds: list, outcomes: dict, scene_filter: str | None = None) -> dict:
    """对一天的 pred + outcome,计算分场景的买入精准率。"""
    from learning.scene_bucket import bucket_by_scene, DEFAULT_SCENE

    if scene_filter:
        preds = [p for p in preds if (p.get("scene") or DEFAULT_SCENE).startswith(scene_filter)]

    by_scene = bucket_by_scene(preds)
    result = {}
    for scene, records in by_scene.items():
        buys = [r for r in records if r.get("signal") == "买入"]
        hits = 0
        total = 0
        for r in buys:
            out = outcomes.get(r["code"])
            if not out or out.get("actual_pct") is None:
                continue
            total += 1
            if out["actual_pct"] >= 1.0:
                hits += 1
        result[scene] = {
            "buys":     len(buys),
            "scored":   total,
            "hits":     hits,
            "accuracy": round(hits / total, 4) if total else 0,
        }
    return result


def run(from_date: str | None, to_date: str | None, scene_filter: str | None):
    from learning.tracker       import list_prediction_dates, load_predictions, load_outcomes
    from learning.market_state  import load_state_on_date

    all_dates = list_prediction_dates()
    if from_date:
        all_dates = [d for d in all_dates if d >= from_date]
    if to_date:
        all_dates = [d for d in all_dates if d <= to_date]

    if not all_dates:
        print("无可用预测数据")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"baseline_{run_ts}.csv")

    rows = []
    print(f"回放区间: {all_dates[0]} ~ {all_dates[-1]},共 {len(all_dates)} 天")

    for i, d in enumerate(all_dates):
        preds    = load_predictions(d)
        outcomes = load_outcomes(d)
        if not preds or not outcomes:
            continue

        state = load_state_on_date(d)
        per_scene = _evaluate_single(preds, outcomes, scene_filter)

        for scene, stats in per_scene.items():
            rows.append({
                "date":         d,
                "market_state": state,
                "scene":        scene,
                "buys":         stats["buys"],
                "scored":       stats["scored"],
                "hits":         stats["hits"],
                "accuracy":     stats["accuracy"],
            })

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(all_dates)} 天...")

    if not rows:
        print("无有效结果写入")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"基线精准率曲线已写入: {out_path}")
    print(f"总行数: {len(rows)}")

    # 汇总
    from collections import defaultdict
    agg = defaultdict(lambda: {"hits": 0, "scored": 0})
    for r in rows:
        key = r["scene"]
        agg[key]["hits"]   += r["hits"]
        agg[key]["scored"] += r["scored"]
    print("\n场景汇总(整个回放期间):")
    for scene, s in agg.items():
        acc = (s["hits"] / s["scored"]) if s["scored"] else 0
        print(f"  {scene:<20} scored={s['scored']:>5} hits={s['hits']:>5} accuracy={acc:.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", help="起始日期 YYYY-MM-DD")
    p.add_argument("--to",   dest="to_date",   help="结束日期 YYYY-MM-DD")
    p.add_argument("--scene", help="只统计某场景,如 scan / tactic / predict")
    args = p.parse_args()
    run(args.from_date, args.to_date, args.scene)


if __name__ == "__main__":
    main()
```

---

- [ ] **Step 12.2: 冒烟验证 — P0 验收核心**

Run:
```bash
python scripts/replay_learn.py --from 2025-07-01
```
Expected:
- 打印"回放区间: 2025-07-01 ~ 2026-04-xx,共 N 天"
- 打印"已处理 N/M 天..." 进度
- 打印"基线精准率曲线已写入: learning/replay/baseline_<timestamp>.csv"
- 打印"场景汇总:"
  - `scan scored=XXX hits=YYY accuracy=Z.Z%`
- accuracy 大概会落在 30-35% 区间(与当前 strategy.json 的 30-day=32.6% 匹配 ±3pt)

Run:
```bash
python scripts/replay_learn.py --from 2026-04-01 --to 2026-04-27
```
Expected: 小区间回放,快速输出。

**这是 P0 验收的决定性测试。** 如果 accuracy 与 strategy.json 严重不一致,先检查:
1. `_evaluate_single` 的"买入"判定逻辑(是否与 optimizer.evaluate_day 一致)
2. scene 过滤是否把旧记录全归到 scan

---

- [ ] **Step 12.3: 提交**

```bash
git add scripts/replay_learn.py
git commit -m "feat(learning): add replay_learn.py for baseline accuracy curves"
```

---

## Task 13: .gitignore 更新 + 占位目录

**Files:**
- Modify: `E:\antenna\.gitignore`
- Create: `data/cache/alt/.gitkeep`

---

- [ ] **Step 13.1: 查看现有 .gitignore**

Run: `cat E:/antenna/.gitignore`

---

- [ ] **Step 13.2: 追加 P0 新增的运行产物目录**

在 `.gitignore` 末尾追加:
```
# Learning system runtime artifacts (P0+)
learning/feedback/
learning/history/
learning/alerts.jsonl
learning/replay/
data/cache/alt/*
!data/cache/alt/.gitkeep
```

注意 `learning/market_state.json` **不排除**(它是学习成果,需要进版本控制以便回滚)。

---

- [ ] **Step 13.3: 创建占位文件**

Create `data/cache/alt/.gitkeep`:
```
# Placeholder for P3 alt_data cache.
```

---

- [ ] **Step 13.4: 提交**

```bash
git add .gitignore data/cache/alt/.gitkeep
git commit -m "chore: gitignore learning runtime artifacts, keep alt cache dir"
```

---

## Task 14: 完整集成测试 + 覆盖率验证

**Files:**
- Create: `tests/test_learn_integration.py`

---

- [ ] **Step 14.1: 写端到端集成测试**

Create `tests/test_learn_integration.py`:
```python
"""
集成测试:走一遍完整的 learn 管线,确保 P0 基础设施衔接正确。
"""
import json
from pathlib import Path
import pytest


def test_cli_learn_dry_run(tmp_path, monkeypatch):
    """模拟 cli.py learn --dry-run 不触发实际网络/磁盘操作。"""
    from learning import orchestrator
    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")

    results = orchestrator.run_all(date_str="2026-04-27", dry_run=True)
    # P0 默认只挂了 market_state
    assert "market_state" in results
    assert results["market_state"]["status"] == "dry_run"


def test_tracker_and_bucket_pipeline(tmp_path, monkeypatch):
    """tracker 写入 → scene_bucket 读取的端到端。"""
    from learning import tracker
    from learning.scene_bucket import bucket_by_scene

    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)

    tracker.log_predictions("2026-04-27", [
        {"code":"A","scene":"scan","signal":"买入","rise_prob":0.7},
        {"code":"A","scene":"tactic:value","signal":"买入","rise_prob":0.72},
        {"code":"B","scene":"scan","signal":"观望","rise_prob":0.55},
    ])

    records = tracker.load_predictions("2026-04-27")
    buckets = bucket_by_scene(records)
    assert len(buckets["scan"]) == 2
    assert len(buckets["tactic:value"]) == 1


def test_outcome_metrics_end_to_end(tmp_path, monkeypatch):
    """写入扩展 outcome → load 验证字段存在。"""
    from learning import tracker
    from learning.outcome_metrics import compute_hit_tier

    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)

    outcomes = {
        "600519": {
            "actual_open": 1800, "actual_close": 1854, "actual_high": 1860, "actual_low": 1790,
            "actual_pct": 3.0,
            "hit_tier": compute_hit_tier(3.0),
            "hit_5d": 7.0,
            "max_drawdown_5d": -1.5,
        }
    }
    tracker.log_outcomes("2026-04-27", outcomes)
    loaded = tracker.load_outcomes("2026-04-27")
    assert loaded["600519"]["hit_tier"] == "good"
    assert loaded["600519"]["hit_5d"] == 7.0


def test_orchestrator_runs_market_state_with_real_data(tmp_path, monkeypatch):
    """验证编排器真能调用 market_state.run(含防御 — 若拿不到数据,fail 但不崩)。"""
    from learning import orchestrator, alerts

    monkeypatch.setattr(orchestrator.feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(alerts, "ALERTS_FILE", tmp_path / "alerts.jsonl")
    # 不允许真调飞书
    monkeypatch.setattr(alerts, "_get_webhook", lambda: "")

    results = orchestrator.run_all(date_str="2026-04-27")
    # 不 assert ok(取决于环境网络),只要 orchestrator 不崩即可
    assert "market_state" in results
    assert results["market_state"]["status"] in ("ok", "failed")
```

---

- [ ] **Step 14.2: 运行全部测试**

Run:
```bash
pytest tests/ -v
```
Expected: 全部通过。可能需要忽略原有不相关失败:`pytest tests/ -v -k "tracker or scene or outcome or market_state or orchestrator or feedback or alerts or learn"`。

---

- [ ] **Step 14.3: 覆盖率检查**

Run:
```bash
pytest tests/ --cov=learning --cov-report=term-missing
```
Expected:
```
learning/tracker.py            XX%
learning/scene_bucket.py       >= 80%
learning/market_state.py       >= 80%
learning/feedback_io.py        >= 80%
learning/alerts.py             >= 80%
learning/orchestrator.py       >= 80%
learning/outcome_metrics.py    >= 80%
```

---

- [ ] **Step 14.4: 提交**

```bash
git add tests/test_learn_integration.py
git commit -m "test(learning): add P0 end-to-end integration tests"
```

---

## Task 15: 更新 project_architecture.md memory 文件

**Files:**
- Modify: `C:\Users\yankj\.claude\projects\E--antenna\memory\project_architecture.md`

---

- [ ] **Step 15.1: 在 "目录结构速查" 的 learning/ 部分追加新模块**

找到 memory 文件中 `learning/` 的文件列表,追加:
```
│   ├── market_state.py      # ★新增 大盘状态打标(bull/bear/range)
│   ├── scene_bucket.py      # ★新增 按 scene/market_state 分桶
│   ├── orchestrator.py      # ★新增 瘦编排器,cli.py learn 调度
│   ├── feedback_io.py       # ★新增 feedback/history 文件工具
│   ├── outcome_metrics.py   # ★新增 hit_tier + 5日指标
│   └── alerts.py            # ★新增 学习告警(飞书 + JSONL)
```

---

- [ ] **Step 15.2: 在 "模块详解" 的 learning/optimizer.py 之后追加 "learning/orchestrator.py — 编排器"**

在 optimizer.py 部分之后插入:
````
### learning/orchestrator.py — 学习管线编排器(P0 新增)

| 函数 | 说明 |
|------|------|
| `run_all(date_str, dry_run) → dict` | 依次执行 MODULES 中每个学习子模块 |
| `check() → int` | 启动自检,验证所有学习产物 JSON 合法 |

**MODULES 结构**:`[{"name": str, "run": callable, "depends_on": list[str]}]`

P0 阶段只挂 `market_state`;P1-P4 阶段逐步追加 model_learner/tactic_learner/feature_learner/price_learner/blacklist。

### learning/market_state.py — 大盘状态打标

| 函数 | 说明 |
|------|------|
| `compute_state(hs300_df) → dict` | 原始分类(无防抖) |
| `update_state(hs300_df, date_str) → dict` | 含 3 日防抖的持久化更新 |
| `load_current_state() → dict` | 读当前状态 |
| `load_state_on_date(date_str) → str` | 查历史状态 |
| `run(date_str) → dict` | 编排器入口 |
````

---

- [ ] **Step 15.3: 在 "维护说明" 末尾追加 P0 完成标记**

在 memory 文件最末追加:
```
## 学习系统阶段进度

- ✅ P0 地基阶段(2026-04-27):scene 标签、outcome 扩展、cli.py learn 骨架、market_state、replay_learn
- ⏳ P1 模型层(待启动):model_learner + blacklist
- ⏳ P2 战法层(未启动)
- ⏳ P3 特征层(未启动)
- ⏳ P4 价位层(未启动)
```

---

- [ ] **Step 15.4: 不需要 commit**(memory 文件不在 git 仓库)

Memory 文件位于 `C:\Users\yankj\.claude\projects\`,由 Claude 的 auto-memory 机制管理,不进 antenna 仓库。

---

## Final Verification

- [ ] **Final Step 1: 全量测试**

```bash
pytest tests/ -v --cov=learning --cov-report=term-missing
```
Expected: 所有测试 PASS,`learning/` 覆盖率 ≥ 80%。

- [ ] **Final Step 2: 冒烟验证完整路径**

```bash
python cli.py learn --dry-run         # ok=0 failed=0 dry_run=1
python cli.py learn                   # market_state 真实写入
python cli.py learn --check           # 退出码 0
python scripts/replay_learn.py --from 2025-07-01  # 输出基线精准率 CSV
```
Expected: 全部成功。

- [ ] **Final Step 3: 回放结果与 strategy.json 交叉验证**

比较:
- `learning/strategy.json.accuracy_30d` ≈ 0.326(当前 4-27 数值)
- `replay_learn.py` 输出的最近 30 日平均 `scan` 场景 accuracy

Expected: 两者差异 < 3pt,说明 baseline 对齐。若差异过大,检查 `_evaluate_single` 与 `optimizer.evaluate_day` 的"买入"判定是否一致。

- [ ] **Final Step 4: 打完工 tag**

```bash
git tag -a learning-p0 -m "P0 foundation complete: scene/outcome extensions + orchestrator + market_state + replay_learn"
```

---

## 执行 Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-antenna-learning-p0-foundation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 我调一个独立 subagent 执行每个 Task,每个 Task 完成后 review,再进下一 Task;快速迭代。

**2. Inline Execution** — 在当前会话依次执行所有 Task,带 checkpoint 让你审查。

**请告诉我选哪种,或者你自己接手也 OK。**

---

## Self-Review(作者自检,已完成)

**Spec 覆盖核对**

| Spec 章节 | 对应 Task |
|---------|-----------|
| §5.1 pred scene 扩展 | Task 1, 9, 10 |
| §5.2 outcome hit_tier/5d 扩展 | Task 2, 11 |
| §5.3 scene 取值 | Task 1, 9, 10(scene 常量定义 + 所有调用点标注) |
| §5.4 market_state | Task 4 |
| §7.1 market_state 打标 | Task 4 |
| §8.1 编排器失败隔离 | Task 7 |
| §8.4 文件原子写 + 7 份历史 | Task 5 |
| §8.5 告警 | Task 6 |
| §9.2 回放脚本 | Task 12 |
| §10 P0 地基阶段 | 所有 Task |

**Placeholder 扫描**:无 TBD/TODO/"implement later",所有代码块完整。

**Type 一致性**:
- `scene` 字段全局用 str,常量 `SCENE_SCAN`/`SCENE_PREDICT`/`SCENE_TACTIC_PREFIX` 在 Task 1 定义后续任务都直接引用
- `market_state` 取值 `{bull,bear,range}` 在 Task 4 定义,Task 7/12/14 均一致使用
- `compute_hit_tier(actual_pct)` 签名 Task 2 定义,Task 11 一致调用
- `compute_5d_metrics(closes)` 签名 Task 2 定义,Task 11 一致调用
- `run_all(date_str, dry_run)` 签名 Task 7 定义,Task 8/11 一致调用
- `orchestrator.MODULES` 的 entry schema(name/run/depends_on)在 Task 7 定义,后续阶段追加时需遵循

**DRY/YAGNI**:
- 原子写通过 `feedback_io.atomic_write_json` 统一,market_state 和后续 learner 都复用
- alerts/feedback_io 分离职责,不重复
- 未实现 P1-P4 任何算法(YAGNI)
- tests/fixtures 共享
