# Antenna Learning P3 + P4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 5 个 alt_data 资金/情绪特征，实现特征重要性动态淘汰（P3），以及基于历史命中的价位参数网格搜索（P4）。

**Architecture:** 三个独立模块（`data/alt_fetcher.py`、`learning/feature_learner.py`、`learning/price_learner.py`）均通过 orchestrator 注册，互不依赖，失败不相互阻塞。推理层新增 `get_active_feature_cols()` 工具函数，所有 FEATURE_COLS 硬引用改为该函数调用；`suggest_dual_period_trades()` 读 `price_params.json` 替代硬编码常量。

**Tech Stack:** Python 3.12+, LightGBM, scikit-learn（`permutation_importance`）, akshare, pandas, PyYAML, pytest

---

## 文件清单

| 路径 | 操作 | 说明 |
|------|------|------|
| `features/technical.py` | 修改 | 新增 `get_active_feature_cols()` |
| `features/builder.py` | 修改 | `build_features(df, alt=None)` |
| `models/predictor.py` | 修改 | dropped 特征补 0，改用 `get_active_feature_cols()` |
| `cli.py` | 修改 | train/predict 路径改用 `get_active_feature_cols()` |
| `server/predict_cmd.py` | 修改 | cmd_predict / cmd_scan_bot 改用 `get_active_feature_cols()`；cmd_scan_bot 内集成 alt_fetcher |
| `features/analyser.py` | 修改 | `suggest_dual_period_trades()` 读 price_params |
| `learning/orchestrator.py` | 修改 | 注册 feature_learner、price_learner |
| `data/alt_fetcher.py` | 新增 | 5 个 alt_data 特征拉取 + parquet 缓存 |
| `learning/feature_learner.py` | 新增 | Permutation Importance + drop tracker |
| `learning/feature_learner.yaml` | 新增 | 重要性阈值、淘汰周数 |
| `learning/price_learner.py` | 新增 | 价位参数网格搜索 |
| `learning/price_learner.yaml` | 新增 | 搜索范围、默认值 |
| `tests/test_feature_learner.py` | 新增 | feature_learner + get_active_feature_cols 单测 |
| `tests/test_price_learner.py` | 新增 | price_learner 单测 |
| `tests/test_alt_fetcher.py` | 新增 | alt_fetcher 单测 |

---

## Task 1：get_active_feature_cols() + predictor 补 0

**Files:**
- Modify: `features/technical.py`
- Modify: `models/predictor.py:82-86`
- Test: `tests/test_feature_learner.py`（本文件贯穿 Task 1、4、5）

- [ ] **Step 1.1: 写失败测试**

新建 `tests/test_feature_learner.py`：

```python
"""tests/test_feature_learner.py — feature_learner + get_active_feature_cols 单测。"""
import json
from pathlib import Path

import pytest


# ── get_active_feature_cols ─────────────────────────────────

def test_get_active_feature_cols_no_file(tmp_path, monkeypatch):
    """文件缺失时回退完整 FEATURE_COLS。"""
    monkeypatch.chdir(tmp_path)
    from features.technical import get_active_feature_cols, FEATURE_COLS
    assert get_active_feature_cols() == FEATURE_COLS


def test_get_active_feature_cols_reads_active(tmp_path, monkeypatch):
    """feature_weights.json 存在时返回 active 列表。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"active": ["ma5", "rsi6"]}), encoding="utf-8"
    )
    from importlib import reload
    import features.technical as t
    reload(t)
    assert t.get_active_feature_cols() == ["ma5", "rsi6"]


def test_get_active_feature_cols_empty_active_fallback(tmp_path, monkeypatch):
    """active 列表为空时回退 FEATURE_COLS。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "feature_weights.json").write_text(
        json.dumps({"active": []}), encoding="utf-8"
    )
    from importlib import reload
    import features.technical as t
    reload(t)
    from features.technical import FEATURE_COLS
    assert t.get_active_feature_cols() == FEATURE_COLS
```

- [ ] **Step 1.2: 运行，确认失败**

```bash
pytest tests/test_feature_learner.py::test_get_active_feature_cols_no_file -v
```
预期：`AttributeError: module 'features.technical' has no attribute 'get_active_feature_cols'`

- [ ] **Step 1.3: 实现 get_active_feature_cols()**

在 `features/technical.py` 末尾追加：

```python
import json as _json
from pathlib import Path as _Path

_FEATURE_WEIGHTS_PATH = _Path("learning/feature_weights.json")


def get_active_feature_cols() -> list[str]:
    """返回当前激活特征列表。文件缺失 / active 为空 / 解析失败 → 回退完整 FEATURE_COLS。"""
    try:
        if not _FEATURE_WEIGHTS_PATH.exists():
            return FEATURE_COLS
        data = _json.loads(_FEATURE_WEIGHTS_PATH.read_text(encoding="utf-8"))
        active = data.get("active", [])
        return active if active else FEATURE_COLS
    except Exception:
        return FEATURE_COLS
```

- [ ] **Step 1.4: 运行，确认通过**

```bash
pytest tests/test_feature_learner.py -k "get_active" -v
```
预期：3 PASSED

- [ ] **Step 1.5: 写 predictor 补 0 测试**

在 `tests/test_feature_learner.py` 末尾追加：

```python
def test_predictor_fills_missing_feature_with_zero(tmp_path, monkeypatch):
    """predict() 对 df 中不存在的特征自动补 0，而非抛 KeyError。"""
    import pandas as pd
    import numpy as np
    monkeypatch.chdir(tmp_path)

    class _FakeModel:
        def predict(self, X):
            return np.array([0.5] * len(X))

    df = pd.DataFrame({"close": [10.0], "ma5": [9.8]})
    from models.predictor import predict
    # ma10 等大量列缺失，predict 应补 0 而不抛异常
    result = predict(df, feature_cols=["ma5", "ma10"], model=_FakeModel(), buy_top_pct=None)
    assert "rise_prob" in result
```

- [ ] **Step 1.6: 运行，确认失败**

```bash
pytest tests/test_feature_learner.py::test_predictor_fills_missing_feature_with_zero -v
```
预期：`KeyError: 'ma10'`

- [ ] **Step 1.7: 修改 predictor.predict() 补 0**

在 `models/predictor.py` 的 `predict()` 函数内，`valid = df[feature_cols].dropna()` **之前**插入：

```python
    # 补齐缺失特征列（dropped 特征不在 df 中时填 0）
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
```

- [ ] **Step 1.8: 运行，确认通过**

```bash
pytest tests/test_feature_learner.py -v
```
预期：4 PASSED

- [ ] **Step 1.9: 更新 cli.py 和 predict_cmd.py 中的 FEATURE_COLS 引用**

`cli.py` 中所有 `from features.technical import FEATURE_COLS` 改为：
```python
from features.technical import get_active_feature_cols
```
并将 `feature_cols=FEATURE_COLS` 的调用改为 `feature_cols=get_active_feature_cols()`。（搜索 `FEATURE_COLS` 确认修改完整）

`server/predict_cmd.py` 中 `cmd_predict` 和 `_scan_one` 两处：
```python
# 改前
from features.technical import FEATURE_COLS
result = predict(df, FEATURE_COLS, ...)

# 改后
from features.technical import get_active_feature_cols
result = predict(df, get_active_feature_cols(), ...)
```

- [ ] **Step 1.10: 全量测试确认无回归**

```bash
pytest tests/ -q
```
预期：316+ PASSED，0 FAILED

- [ ] **Step 1.11: Commit**

```bash
git add features/technical.py models/predictor.py cli.py server/predict_cmd.py tests/test_feature_learner.py
git commit -m "feat(p3): get_active_feature_cols() + predictor 补 0 for dropped features"
```

---

## Task 2：build_features alt= 参数

**Files:**
- Modify: `features/builder.py`
- Test: `tests/test_feature_learner.py`（追加）

- [ ] **Step 2.1: 写失败测试**

追加到 `tests/test_feature_learner.py`：

```python
def test_build_features_appends_alt_to_last_row():
    """build_features(df, alt={...}) 把 alt 值追加到 df 最后一行。"""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "open": [10.0] * 5, "high": [11.0] * 5,
        "low":  [9.0]  * 5, "close": [10.5] * 5,
        "volume": [1e6] * 5,
    })
    from features.builder import build_features
    result = build_features(df, alt={"main_net_in_1d": 0.3, "sector_heat_rank": -0.5})
    assert result["main_net_in_1d"].iloc[-1] == pytest.approx(0.3)
    assert result["sector_heat_rank"].iloc[-1] == pytest.approx(-0.5)
    # 非最后行保持 NaN
    assert pd.isna(result["main_net_in_1d"].iloc[0])


def test_build_features_no_alt_unchanged():
    """不传 alt 时行为与原版一致。"""
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5),
        "open": [10.0] * 5, "high": [11.0] * 5,
        "low":  [9.0]  * 5, "close": [10.5] * 5,
        "volume": [1e6] * 5,
    })
    from features.builder import build_features
    result = build_features(df)
    assert "main_net_in_1d" not in result.columns
```

- [ ] **Step 2.2: 运行，确认失败**

```bash
pytest tests/test_feature_learner.py -k "build_features" -v
```
预期：`TypeError: build_features() got an unexpected keyword argument 'alt'`

- [ ] **Step 2.3: 修改 builder.py**

将 `features/builder.py` 全文替换为：

```python
import pandas as pd
from features.technical import add_indicators


def build_features(df: pd.DataFrame, alt: dict | None = None) -> pd.DataFrame:
    """在原始 OHLCV 数据上计算所有技术指标，返回完整 DataFrame。

    alt: {feature_name: value} 追加到 df 最后一行（predictor 只用最后一行）。
         None 或空 dict 时行为与原版一致。
    """
    df = add_indicators(df)
    if alt:
        last_idx = df.index[-1]
        for col, val in alt.items():
            if col not in df.columns:
                df[col] = float("nan")
            df.loc[last_idx, col] = float(val) if val is not None else 0.0
    return df
```

- [ ] **Step 2.4: 运行，确认通过**

```bash
pytest tests/test_feature_learner.py -k "build_features" -v
```
预期：2 PASSED

- [ ] **Step 2.5: 全量测试**

```bash
pytest tests/ -q
```
预期：全绿

- [ ] **Step 2.6: Commit**

```bash
git add features/builder.py tests/test_feature_learner.py
git commit -m "feat(p3): build_features 接受 alt= 参数追加 alt_data 到最后一行"
```

---

## Task 3：alt_fetcher.py

**Files:**
- Create: `data/alt_fetcher.py`
- Create: `tests/test_alt_fetcher.py`

- [ ] **Step 3.1: 写失败测试**

新建 `tests/test_alt_fetcher.py`：

```python
"""tests/test_alt_fetcher.py — alt_fetcher 单元测试。"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# ── 缓存命中 ────────────────────────────────────────────────

def test_returns_cached_result(tmp_path, monkeypatch):
    """缓存文件存在时直接读取，不调 akshare。"""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "data" / "cache" / "alt"
    cache_dir.mkdir(parents=True)
    cached = pd.DataFrame([{
        "code": "600519",
        "main_net_in_1d": 0.3, "main_net_in_5d": 0.1,
        "dragon_top_cnt_10d": 0.0, "sector_heat_rank": 0.5,
        "north_hold_chg_5d": None,
    }])
    cached.to_parquet(cache_dir / "2026-05-18.parquet", index=False)

    from data.alt_fetcher import fetch_alt_features
    result = fetch_alt_features(["600519"], "2026-05-18")
    assert "600519" in result
    assert result["600519"]["main_net_in_1d"] == pytest.approx(0.3)


# ── 单接口失败降级 ───────────────────────────────────────────

def test_single_feature_failure_returns_none_for_that_feature(tmp_path, monkeypatch):
    """某个接口抛异常时，对应特征返回 None，其余特征正常。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache" / "alt").mkdir(parents=True)

    def _raise(*a, **kw):
        raise RuntimeError("akshare timeout")

    with patch("data.alt_fetcher._fetch_fund_flow", side_effect=_raise), \
         patch("data.alt_fetcher._fetch_dragon_board", return_value={"000001": 2}), \
         patch("data.alt_fetcher._fetch_sector_heat", return_value={"000001": 0.7}), \
         patch("data.alt_fetcher._fetch_north_flow", return_value={"000001": None}):
        from importlib import reload
        import data.alt_fetcher as m; reload(m)
        result = m.fetch_alt_features(["000001"], "2026-05-18")

    assert result["000001"]["main_net_in_1d"] is None
    assert result["000001"]["dragon_top_cnt_10d"] == pytest.approx(2.0 / 10)  # normalized


# ── 全部失败返回空 dict ──────────────────────────────────────

def test_all_features_fail_returns_empty_dict_per_code(tmp_path, monkeypatch):
    """所有接口都失败时，code 对应的 dict 全是 None。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache" / "alt").mkdir(parents=True)

    with patch("data.alt_fetcher._fetch_fund_flow", side_effect=Exception), \
         patch("data.alt_fetcher._fetch_dragon_board", side_effect=Exception), \
         patch("data.alt_fetcher._fetch_sector_heat", side_effect=Exception), \
         patch("data.alt_fetcher._fetch_north_flow", side_effect=Exception):
        from importlib import reload
        import data.alt_fetcher as m; reload(m)
        result = m.fetch_alt_features(["600519"], "2026-05-18")

    assert result["600519"]["main_net_in_1d"] is None
    assert result["600519"]["north_hold_chg_5d"] is None


# ── 结果写入 parquet 缓存 ────────────────────────────────────

def test_result_cached_to_parquet(tmp_path, monkeypatch):
    """成功拉取后结果写入 data/cache/alt/{date}.parquet。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache" / "alt").mkdir(parents=True)

    with patch("data.alt_fetcher._fetch_fund_flow", return_value={"600519": (0.2, 0.1)}), \
         patch("data.alt_fetcher._fetch_dragon_board", return_value={"600519": 1}), \
         patch("data.alt_fetcher._fetch_sector_heat", return_value={"600519": 0.6}), \
         patch("data.alt_fetcher._fetch_north_flow", return_value={"600519": 0.05}):
        from importlib import reload
        import data.alt_fetcher as m; reload(m)
        m.fetch_alt_features(["600519"], "2026-05-19")

    cache_path = tmp_path / "data" / "cache" / "alt" / "2026-05-19.parquet"
    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert "600519" in df["code"].values


# ── 空 codes 列表 ────────────────────────────────────────────

def test_empty_codes_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache" / "alt").mkdir(parents=True)
    from data.alt_fetcher import fetch_alt_features
    assert fetch_alt_features([], "2026-05-18") == {}
```

- [ ] **Step 3.2: 运行，确认失败**

```bash
pytest tests/test_alt_fetcher.py -v
```
预期：`ModuleNotFoundError: No module named 'data.alt_fetcher'`

- [ ] **Step 3.3: 创建 data/alt_fetcher.py**

```python
"""
data/alt_fetcher.py - alt_data 特征拉取与缓存。

5 个资金/情绪特征，每日盘前（task_scan.py 调用）批量拉取并缓存。
每个接口独立 try/except，失败返回 None，不阻断扫描路径。

缓存: data/cache/alt/{date_str}.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ALT_COLS = [
    "main_net_in_1d",
    "main_net_in_5d",
    "dragon_top_cnt_10d",
    "sector_heat_rank",
    "north_hold_chg_5d",
]

_CACHE_DIR = Path("data/cache/alt")


# ── 私有接口封装 ─────────────────────────────────────────────

def _fetch_fund_flow(codes: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """返回 {code: (net_1d_normalized, net_5d_normalized)}。"""
    import akshare as ak
    result: dict[str, tuple] = {}
    for code in codes:
        try:
            market = "sh" if code.startswith("6") else "sz"
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is None or df.empty:
                continue
            df = df.sort_values("日期") if "日期" in df.columns else df
            row = df.iloc[-1]
            turnover = abs(float(row.get("成交额", 0) or 1e8)) or 1e8
            net_1d = float(row.get("主力净流入-净额", 0) or 0.0)
            net_5d = float(df.tail(5)["主力净流入-净额"].sum()) if len(df) >= 5 else net_1d
            result[code] = (
                max(-1.0, min(1.0, net_1d / turnover)),
                max(-1.0, min(1.0, net_5d / (turnover * 5))),
            )
        except Exception:
            pass
    return result


def _fetch_dragon_board(codes: list[str]) -> dict[str, int]:
    """返回 {code: 近10日龙虎榜出现次数}。"""
    import akshare as ak
    try:
        df = ak.stock_lhb_detail_em(symbol="近10日")
        if df is None or df.empty:
            return {}
        col = "代码" if "代码" in df.columns else df.columns[0]
        counts = df[col].value_counts().to_dict()
        return {c: int(counts.get(c, 0)) for c in codes}
    except Exception:
        return {}


def _fetch_sector_heat(codes: list[str]) -> dict[str, float | None]:
    """返回 {code: 所属行业10日涨幅排名分位 0~1}。"""
    import akshare as ak
    from data.universe import load_sector_map  # {code: sector_name}
    try:
        sector_map = load_sector_map()
        df = ak.stock_board_industry_hist_em(symbol="沪深两市", period="10")
        if df is None or df.empty:
            return {}
        name_col = "板块名称" if "板块名称" in df.columns else df.columns[0]
        chg_col  = "涨跌幅"  if "涨跌幅"  in df.columns else df.columns[2]
        df = df.sort_values(chg_col, ascending=False).reset_index(drop=True)
        n = len(df)
        rank_map = {row[name_col]: (i + 1) / n for i, row in df.iterrows()}
        result = {}
        for code in codes:
            sector = sector_map.get(code)
            result[code] = rank_map.get(sector) if sector else None
        return result
    except Exception:
        return {}


def _fetch_north_flow(codes: list[str]) -> dict[str, float | None]:
    """返回 {code: 北向5日持股变动 normalized}。"""
    import akshare as ak
    try:
        df = ak.stock_hsgt_hold_stock_em(market="沪股通", indicator="5日")
        if df is None or df.empty:
            return {}
        code_col = "股票代码" if "股票代码" in df.columns else df.columns[0]
        chg_col  = "5日涨跌" if "5日涨跌" in df.columns else df.columns[-1]
        mapping = {}
        for _, row in df.iterrows():
            c = str(row.get(code_col, "")).zfill(6)
            val = row.get(chg_col)
            try:
                mapping[c] = max(-1.0, min(1.0, float(val) / 100)) if val is not None else None
            except (TypeError, ValueError):
                mapping[c] = None
        return {code: mapping.get(code) for code in codes}
    except Exception:
        return {}


# ── 公开接口 ─────────────────────────────────────────────────

def fetch_alt_features(
    codes: list[str],
    date_str: str,
) -> dict[str, dict[str, float | None]]:
    """批量拉取 5 个 alt_data 特征，缓存到 parquet，返回 {code: {feature: value}}。

    - 每个接口独立失败，失败特征返回 None
    - 同日已有缓存直接读取，不重复拉取
    - codes 为空时返回 {}
    """
    if not codes:
        return {}

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{date_str}.parquet"

    # ── 缓存命中 ────────────────────────────────────────────
    if cache_path.exists():
        try:
            cached_df = pd.read_parquet(cache_path)
            result = {}
            for _, row in cached_df.iterrows():
                code = str(row.get("code", ""))
                result[code] = {col: row.get(col) for col in ALT_COLS}
            if result:
                return result
        except Exception:
            pass

    # ── 拉取各接口（串行，避免频率限制）──────────────────────
    logger.info("[alt_fetcher] 拉取 %d 支股票 alt_data (%s)", len(codes), date_str)

    fund_flow  = _try_call(_fetch_fund_flow, codes)
    dragon     = _try_call(_fetch_dragon_board, codes)
    sector     = _try_call(_fetch_sector_heat, codes)
    north      = _try_call(_fetch_north_flow, codes)

    # ── 合并结果 ─────────────────────────────────────────────
    result: dict[str, dict] = {}
    for code in codes:
        ff = fund_flow.get(code) if fund_flow else None
        result[code] = {
            "main_net_in_1d":     ff[0] if ff else None,
            "main_net_in_5d":     ff[1] if ff else None,
            "dragon_top_cnt_10d": float(dragon.get(code, 0)) / 10 if dragon and code in dragon else None,
            "sector_heat_rank":   sector.get(code) if sector else None,
            "north_hold_chg_5d":  north.get(code)  if north  else None,
        }

    # ── 写缓存 ───────────────────────────────────────────────
    try:
        rows = [{"code": code, **feats} for code, feats in result.items()]
        pd.DataFrame(rows).to_parquet(cache_path, index=False)
    except Exception as e:
        logger.warning("[alt_fetcher] 缓存写入失败: %s", e)

    return result


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning("[alt_fetcher] %s 失败: %s", fn.__name__, e)
        return None
```

- [ ] **Step 3.4: 检查 data/universe.py 是否有 load_sector_map()**

```bash
grep -n "load_sector_map\|sector_map" data/universe.py
```

若不存在，在 `data/universe.py` 末尾追加：

```python
def load_sector_map() -> dict:
    """加载代码→行业映射。"""
    import json
    p = Path("data/sector_map.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
```

- [ ] **Step 3.5: 运行测试**

```bash
pytest tests/test_alt_fetcher.py -v
```
预期：5 PASSED

- [ ] **Step 3.6: 全量测试**

```bash
pytest tests/ -q
```
预期：全绿

- [ ] **Step 3.7: Commit**

```bash
git add data/alt_fetcher.py tests/test_alt_fetcher.py data/universe.py
git commit -m "feat(p3): alt_fetcher - 5 个 alt_data 特征拉取与缓存"
```

---

## Task 4：feature_learner.py

**Files:**
- Create: `learning/feature_learner.py`
- Create: `learning/feature_learner.yaml`
- Test: `tests/test_feature_learner.py`（追加）

- [ ] **Step 4.1: 创建 learning/feature_learner.yaml**

```yaml
# learning/feature_learner.yaml
permutation:
  n_repeats: 10          # permutation_importance 重复次数
  lookback_days: 90      # 样本窗口（日历天数）

drop:
  importance_threshold: 0.05   # 低于此值计入"低重要"
  candidate_weeks: 4           # 连续低重要 → candidate_drop
  drop_weeks: 8                # 连续低重要 → dropped

alt_data:
  ic_threshold: 0.02     # |IC| 达标才加入 active
  ic_lookback_days: 90
```

- [ ] **Step 4.2: 写失败测试（drop tracker + config）**

追加到 `tests/test_feature_learner.py`：

```python
# ── feature_learner config ──────────────────────────────────

def test_load_config_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from learning.feature_learner import load_config
    with pytest.raises(ValueError, match="feature_learner.yaml"):
        load_config()


def test_load_config_reads_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    cfg_path = tmp_path / "learning" / "feature_learner.yaml"
    cfg_path.write_text(
        "permutation:\n  n_repeats: 5\n  lookback_days: 60\n"
        "drop:\n  importance_threshold: 0.05\n  candidate_weeks: 4\n  drop_weeks: 8\n"
        "alt_data:\n  ic_threshold: 0.02\n  ic_lookback_days: 90\n",
        encoding="utf-8",
    )
    from importlib import reload
    import learning.feature_learner as m; reload(m)
    cfg = m.load_config(cfg_path)
    assert cfg.n_repeats == 5
    assert cfg.candidate_weeks == 4


# ── drop tracker ────────────────────────────────────────────

def test_update_drop_tracker_normal_feature_stays_active():
    """重要性 >= threshold 的特征保持 active。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {}, "dropped": []}
    importances = {"ma5": 0.10, "cci": 0.08}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert "ma5" not in new["candidate_drop"]
    assert "ma5" not in new["dropped"]


def test_update_drop_tracker_accumulates_candidate():
    """低重要性特征连续累积 weeks_below 计数。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {"cci": {"weeks_below": 3}}, "dropped": []}
    importances = {"cci": 0.01}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert new["candidate_drop"]["cci"]["weeks_below"] == 4


def test_update_drop_tracker_promotes_to_dropped():
    """连续 drop_weeks 后移入 dropped。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {"cci": {"weeks_below": 7}}, "dropped": []}
    importances = {"cci": 0.01}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert "cci" in new["dropped"]
    assert "cci" not in new["candidate_drop"]


def test_update_drop_tracker_resets_on_recovery():
    """特征重要性恢复后，candidate_drop 计数清零。"""
    from learning.feature_learner import _update_drop_tracker
    state = {"candidate_drop": {"cci": {"weeks_below": 3}}, "dropped": []}
    importances = {"cci": 0.10}
    new = _update_drop_tracker(state, importances, threshold=0.05, candidate_weeks=4, drop_weeks=8)
    assert "cci" not in new["candidate_drop"]


# ── weekly skip ─────────────────────────────────────────────

def test_run_skips_on_non_sunday(tmp_path, monkeypatch):
    """非周日调用 run() 返回 skipped。"""
    monkeypatch.chdir(tmp_path)
    from learning.feature_learner import run
    # 2026-05-18 is Monday
    result = run("2026-05-18")
    assert result["status"] == "skipped"


def test_run_cold_start_no_model(tmp_path, monkeypatch):
    """无模型时返回 cold_start（不抛异常）。2026-05-17 是周日。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "data").mkdir()
    cfg_path = tmp_path / "learning" / "feature_learner.yaml"
    cfg_path.write_text(
        "permutation:\n  n_repeats: 3\n  lookback_days: 90\n"
        "drop:\n  importance_threshold: 0.05\n  candidate_weeks: 4\n  drop_weeks: 8\n"
        "alt_data:\n  ic_threshold: 0.02\n  ic_lookback_days: 90\n",
        encoding="utf-8",
    )
    (tmp_path / "models" / "saved").mkdir(parents=True)
    from importlib import reload
    import learning.feature_learner as m; reload(m)
    result = m.run("2026-05-17")  # Sunday
    assert result["status"] in ("cold_start", "no_model")
```

- [ ] **Step 4.3: 运行，确认失败**

```bash
pytest tests/test_feature_learner.py -k "feature_learner or drop_tracker or weekly" -v
```
预期：`ModuleNotFoundError: No module named 'learning.feature_learner'`

- [ ] **Step 4.4: 创建 learning/feature_learner.py**

```python
"""
learning/feature_learner.py - P3 特征层:Permutation Importance + 动态淘汰。

职责:
  - 每周日运行（内部检查 weekday，非周日返回 skipped）
  - 对最新 LightGBM 模型跑 sklearn permutation_importance
  - 连续 candidate_weeks 低重要 → candidate_drop；连续 drop_weeks → dropped
  - alt_data 特征按 IC 阈值决定是否加入 active
  - 产物: learning/feature_weights.json

读取: learning/feature_learner.yaml, learning/data/pred_*.jsonl,
      learning/data/outcome_*.jsonl, data/cache/{code}.parquet
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from features.technical import FEATURE_COLS, get_active_feature_cols
from learning import feedback_io

logger = logging.getLogger(__name__)

CONFIG_PATH         = Path("learning/feature_learner.yaml")
FEATURE_WEIGHTS_PATH = Path("learning/feature_weights.json")
DATA_DIR            = Path("learning/data")
ALT_COLS            = [
    "main_net_in_1d", "main_net_in_5d",
    "dragon_top_cnt_10d", "sector_heat_rank", "north_hold_chg_5d",
]


# ── 配置 ────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureLearnerConfig:
    n_repeats:            int
    lookback_days:        int
    importance_threshold: float
    candidate_weeks:      int
    drop_weeks:           int
    ic_threshold:         float
    ic_lookback_days:     int


def load_config(path: str | Path | None = None) -> FeatureLearnerConfig:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise ValueError(f"feature_learner.yaml 未找到: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FeatureLearnerConfig(
        n_repeats=raw["permutation"]["n_repeats"],
        lookback_days=raw["permutation"]["lookback_days"],
        importance_threshold=raw["drop"]["importance_threshold"],
        candidate_weeks=raw["drop"]["candidate_weeks"],
        drop_weeks=raw["drop"]["drop_weeks"],
        ic_threshold=raw["alt_data"]["ic_threshold"],
        ic_lookback_days=raw["alt_data"]["ic_lookback_days"],
    )


# ── 状态读写 ─────────────────────────────────────────────────

def _load_feature_weights() -> dict:
    if not FEATURE_WEIGHTS_PATH.exists():
        return {"candidate_drop": {}, "dropped": [], "candidates": [], "importance": {}}
    try:
        return json.loads(FEATURE_WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"candidate_drop": {}, "dropped": [], "candidates": [], "importance": {}}


# ── drop tracker（纯函数，便于单测）────────────────────────

def _update_drop_tracker(
    state: dict,
    importances: dict[str, float],
    threshold: float,
    candidate_weeks: int,
    drop_weeks: int,
) -> dict:
    """返回更新后的 {candidate_drop, dropped} state（不修改原 state）。"""
    candidate_drop: dict = dict(state.get("candidate_drop") or {})
    dropped: list = list(state.get("dropped") or [])

    for feat, score in importances.items():
        if feat in dropped:
            continue
        if score < threshold:
            entry = candidate_drop.get(feat, {"weeks_below": 0})
            entry = {"weeks_below": entry["weeks_below"] + 1}
            if entry["weeks_below"] >= drop_weeks:
                dropped.append(feat)
                candidate_drop.pop(feat, None)
            else:
                candidate_drop[feat] = entry
        else:
            # 重要性恢复，清零
            candidate_drop.pop(feat, None)

    return {"candidate_drop": candidate_drop, "dropped": dropped}


# ── 样本构建 ─────────────────────────────────────────────────

def _rebuild_features_row(code: str, day: str) -> dict | None:
    """从 parquet 缓存复原 day 当天特征（dict 形式）。"""
    from features.builder import build_features
    cache = Path("data/cache") / f"{code}.parquet"
    if not cache.exists():
        return None
    try:
        df = pd.read_parquet(cache)
        if df is None or df.empty or "date" not in df.columns:
            return None
        df["date"] = df["date"].astype(str).str[:10]
        row_df = df[df["date"] <= day]
        if row_df.empty:
            return None
        row_df = build_features(row_df)
        last = row_df.iloc[-1]
        return {col: last.get(col) for col in FEATURE_COLS}
    except Exception:
        return None


def _load_outcomes(date_str: str) -> dict[str, dict]:
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                result[r["code"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def _load_alt_for(code: str, date_str: str) -> dict:
    """从 alt 缓存读单只股票当日 alt 特征。"""
    path = Path("data/cache/alt") / f"{date_str}.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        row = df[df["code"] == code]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
    except Exception:
        return {}


def _date_range(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _build_sample_matrix(date_str: str, lookback_days: int) -> tuple:
    """返回 (X: pd.DataFrame, y: pd.Series, alt_df: pd.DataFrame)。"""
    from learning.tracker import load_predictions_by_scene
    end = datetime.strptime(date_str, "%Y-%m-%d")
    start = end - timedelta(days=lookback_days)
    rows = []
    for d in _date_range(start, end):
        d_str = d.strftime("%Y-%m-%d")
        preds = {r["code"]: r for r in load_predictions_by_scene(d_str, "scan")}
        outcomes = _load_outcomes(d_str)
        for code, pred in preds.items():
            outcome = outcomes.get(code)
            if outcome is None:
                continue
            hit = 1 if (outcome.get("actual_pct") or 0.0) >= 0.02 else 0
            feats = _rebuild_features_row(code, d_str)
            if feats is None:
                continue
            row = feats.copy()
            row["hit"] = hit
            alt = _load_alt_for(code, d_str)
            for col in ALT_COLS:
                row[col] = alt.get(col, 0.0) if alt.get(col) is not None else 0.0
            rows.append(row)
    if not rows:
        return pd.DataFrame(), pd.Series([], dtype=int), pd.DataFrame()
    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS]
    y = df["hit"]
    alt_df = df[["hit"] + [c for c in ALT_COLS if c in df.columns]]
    return X, y, alt_df


# ── permutation importance ───────────────────────────────────

class _LGBEstimator:
    def __init__(self, booster):
        self._b = booster

    def predict_proba(self, X):
        import numpy as np
        preds = self._b.predict(np.asarray(X, dtype=float))
        return np.column_stack([1 - preds, preds])


def _compute_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int) -> dict[str, float]:
    from sklearn.inspection import permutation_importance
    est = _LGBEstimator(model)
    result = permutation_importance(
        est, X.values, y.values,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )
    return {col: float(v) for col, v in zip(X.columns, result.importances_mean)}


def _compute_ic(alt_df: pd.DataFrame, ic_threshold: float) -> list[str]:
    """返回 IC >= threshold 的 alt 特征列表。"""
    active_alt = []
    for col in ALT_COLS:
        if col not in alt_df.columns or "hit" not in alt_df.columns:
            continue
        valid = alt_df[[col, "hit"]].dropna()
        if len(valid) < 20:
            continue
        ic = abs(float(valid[col].corr(valid["hit"])))
        if ic >= ic_threshold:
            active_alt.append(col)
    return active_alt


# ── 主入口 ───────────────────────────────────────────────────

def fit_feature_weights(date_str: str, cfg: FeatureLearnerConfig | None = None) -> dict:
    """主编排函数。返回状态 dict。"""
    cfg = cfg or load_config()
    try:
        from models.trainer import load_latest_model
        model = load_latest_model("models/saved")
    except Exception:
        return {"status": "no_model"}

    X, y, alt_df = _build_sample_matrix(date_str, cfg.lookback_days)
    if len(X) < 30:
        return {"status": "cold_start", "samples": len(X)}

    importances = _compute_importance(model, X, y, cfg.n_repeats)

    current = _load_feature_weights()
    new_state = _update_drop_tracker(
        current, importances,
        cfg.importance_threshold, cfg.candidate_weeks, cfg.drop_weeks,
    )

    active_alt = _compute_ic(alt_df, cfg.ic_threshold)
    active = [f for f in FEATURE_COLS if f not in new_state["dropped"]] + active_alt

    state = {
        "version":    1,
        "updated_at": datetime.now().isoformat(),
        "active":     active,
        "candidate_drop": new_state["candidate_drop"],
        "dropped":    new_state["dropped"],
        "candidates": [c for c in ALT_COLS if c not in active_alt],
        "importance": {k: round(v, 4) for k, v in importances.items()},
    }
    feedback_io.atomic_write_json(FEATURE_WEIGHTS_PATH, state)
    feedback_io.write_feedback("feature_learner", date_str, {"active": len(active), "dropped": len(new_state["dropped"])})
    return {"status": "ok", "active": len(active), "dropped": len(new_state["dropped"])}


def run(date_str: str | None = None) -> dict:
    """编排器入口。非周日返回 skipped；异常上抛，由 orchestrator 捕获。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    if datetime.strptime(date_str, "%Y-%m-%d").weekday() != 6:  # 0=Mon, 6=Sun
        return {"status": "skipped", "reason": "weekly only (Sunday)"}
    cfg = load_config()
    return fit_feature_weights(date_str, cfg)
```

- [ ] **Step 4.5: 运行测试**

```bash
pytest tests/test_feature_learner.py -v
```
预期：全部 PASSED（包括 cold_start/no_model 不抛异常）

- [ ] **Step 4.6: 全量测试**

```bash
pytest tests/ -q
```
预期：全绿

- [ ] **Step 4.7: Commit**

```bash
git add learning/feature_learner.py learning/feature_learner.yaml tests/test_feature_learner.py
git commit -m "feat(p3): feature_learner - Permutation Importance + 动态特征淘汰"
```

---

## Task 5：price_learner.py

**Files:**
- Create: `learning/price_learner.py`
- Create: `learning/price_learner.yaml`
- Create: `tests/test_price_learner.py`

- [ ] **Step 5.1: 创建 learning/price_learner.yaml**

```yaml
# learning/price_learner.yaml
lookback_days: 90
min_samples: 30

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

- [ ] **Step 5.2: 写失败测试**

新建 `tests/test_price_learner.py`：

```python
"""tests/test_price_learner.py — price_learner 单元测试。"""
import json
from pathlib import Path

import pandas as pd
import pytest


# ── load_price_params ────────────────────────────────────────

def test_load_price_params_missing_file_returns_defaults(tmp_path, monkeypatch):
    """price_params.json 缺失时返回 yaml 中的默认值，不抛异常。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    cfg_path = tmp_path / "learning" / "price_learner.yaml"
    cfg_path.write_text(
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n",
        encoding="utf-8",
    )
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("range")
    assert params["short_atr_mult"] == pytest.approx(1.5)
    assert params["long_amp_mult"] == pytest.approx(3.0)


def test_load_price_params_reads_json(tmp_path, monkeypatch):
    """price_params.json 存在时读取对应 state 的参数。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    (tmp_path / "learning" / "price_learner.yaml").write_text(
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n",
        encoding="utf-8",
    )
    (tmp_path / "learning" / "price_params.json").write_text(
        json.dumps({"bull": {"short_atr_mult": 1.8, "short_gain_mult": 2.5,
                             "long_amp_mult": 3.2, "long_ma60_buffer": 0.98}}),
        encoding="utf-8",
    )
    from importlib import reload
    import learning.price_learner as m; reload(m)
    params = m.load_price_params("bull")
    assert params["short_atr_mult"] == pytest.approx(1.8)


# ── _grid_score ─────────────────────────────────────────────

def test_grid_score_win_triggers(tmp_path):
    """hit_5d >= sell threshold 计为 win。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.06,          # 6% return
        "max_drawdown_5d": -0.01,
        "atr_pct": 0.015,        # atr/close = 1.5%
        "up_ratio": 2.0,         # 2% daily move estimate
    }])
    # short_gain_mult=2.0 → sell_thresh = 2.0 * 0.02 = 4% → win (hit_5d=6%)
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score > 0.0


def test_grid_score_loss_triggers():
    """max_drawdown_5d <= -stop threshold 计为 loss。"""
    from learning.price_learner import _score_params
    samples = pd.DataFrame([{
        "hit_5d": 0.01,
        "max_drawdown_5d": -0.04,  # -4% drawdown
        "atr_pct": 0.015,
        "up_ratio": 2.0,
    }])
    # short_atr_mult=1.5 → stop_thresh = 1.5 * 0.015 = 2.25% → loss (-4% < -2.25%)
    score = _score_params(samples, short_atr_mult=1.5, short_gain_mult=2.0,
                          long_amp_mult=3.0, long_ma60_buffer=0.97)
    assert score < 0.0


# ── fit_price_params cold start ──────────────────────────────

def test_fit_price_params_cold_start_uses_defaults(tmp_path, monkeypatch):
    """样本 < min_samples 时各桶都回退默认值。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning" / "data").mkdir(parents=True)
    cfg_yaml = (
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n"
    )
    (tmp_path / "learning" / "price_learner.yaml").write_text(cfg_yaml, encoding="utf-8")
    from importlib import reload
    import learning.price_learner as m; reload(m)
    cfg = m.load_config()
    result = m.fit_price_params("2026-05-17", cfg)
    # 无数据 → cold_start
    assert result["status"] in ("cold_start", "ok")


# ── weekly skip ─────────────────────────────────────────────

def test_run_skips_on_non_sunday(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from learning.price_learner import run
    result = run("2026-05-18")  # Monday
    assert result["status"] == "skipped"
```

- [ ] **Step 5.3: 运行，确认失败**

```bash
pytest tests/test_price_learner.py -v
```
预期：`ModuleNotFoundError: No module named 'learning.price_learner'`

- [ ] **Step 5.4: 创建 learning/price_learner.py**

```python
"""
learning/price_learner.py - P4 价位层:ATR/振幅系数网格搜索。

职责:
  - 每周日运行（内部检查 weekday，非周日返回 skipped）
  - 读最近 90 日买入信号 + outcome[hit_5d, max_drawdown_5d]
  - 对 short_atr_mult / short_gain_mult / long_amp_mult / long_ma60_buffer 做网格搜索
  - 分 bull/bear/range 各搜一套参数，样本 < min_samples 回退默认值
  - 产物: learning/price_params.json

读取: learning/price_learner.yaml, learning/data/pred_*.jsonl,
      learning/data/outcome_*.jsonl, data/cache/{code}.parquet
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from learning import feedback_io

logger = logging.getLogger(__name__)

CONFIG_PATH        = Path("learning/price_learner.yaml")
PRICE_PARAMS_PATH  = Path("learning/price_params.json")
DATA_DIR           = Path("learning/data")


# ── 配置 ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PriceLearnerConfig:
    lookback_days: int
    min_samples:   int
    grid:          dict
    defaults:      dict


def load_config(path: str | Path | None = None) -> PriceLearnerConfig:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise ValueError(f"price_learner.yaml 未找到: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PriceLearnerConfig(
        lookback_days=raw["lookback_days"],
        min_samples=raw["min_samples"],
        grid=raw["grid"],
        defaults=raw["defaults"],
    )


# ── 参数读取（供 analyser 调用）─────────────────────────────

def load_price_params(state: str) -> dict:
    """读 price_params.json[state]；文件缺失或解析失败 → 从 yaml 读 defaults。永不抛。"""
    try:
        if PRICE_PARAMS_PATH.exists():
            data = json.loads(PRICE_PARAMS_PATH.read_text(encoding="utf-8"))
            if state in data:
                return data[state]
    except Exception:
        pass
    try:
        cfg = load_config()
        return dict(cfg.defaults)
    except Exception:
        return {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
                "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}


# ── 样本构建 ─────────────────────────────────────────────────

def _load_outcomes(date_str: str) -> dict[str, dict]:
    path = DATA_DIR / f"outcome_{date_str}.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                result[r["code"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def _get_atr_pct(code: str, day: str) -> float | None:
    """从 parquet 缓存读 atr/close。"""
    cache = Path("data/cache") / f"{code}.parquet"
    if not cache.exists():
        return None
    try:
        df = pd.read_parquet(cache)
        df["date"] = df["date"].astype(str).str[:10]
        row = df[df["date"] <= day]
        if row.empty:
            return None
        last = row.iloc[-1]
        atr   = float(last.get("atr") or 0)
        close = float(last.get("close") or 1)
        return atr / close if close > 0 else None
    except Exception:
        return None


def _build_buy_samples(date_str: str, state: str, lookback_days: int) -> pd.DataFrame:
    """返回 (code, hit_5d, max_drawdown_5d, atr_pct, up_ratio) 的 DataFrame。"""
    from learning.tracker import load_predictions_by_scene
    end = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=7)  # 5日滞后缓冲
    start = end - timedelta(days=lookback_days)

    rows = []
    d = start
    while d <= end:
        d_str = d.strftime("%Y-%m-%d")
        preds = load_predictions_by_scene(d_str, "scan")
        outcomes = _load_outcomes(d_str)
        for pred in preds:
            if pred.get("signal") != "买入":
                continue
            if pred.get("market_state") != state:
                continue
            code = pred["code"]
            outcome = outcomes.get(code)
            if outcome is None:
                continue
            hit_5d     = outcome.get("hit_5d")
            drawdown   = outcome.get("max_drawdown_5d")
            if hit_5d is None or drawdown is None:
                continue
            atr_pct = _get_atr_pct(code, d_str)
            pi = pred.get("price_info") or pred.get("dual_trade") or {}
            up_ratio = float(pi.get("up_ratio") or 2.0) / 100
            rows.append({
                "hit_5d":          float(hit_5d),
                "max_drawdown_5d": float(drawdown),
                "atr_pct":         float(atr_pct) if atr_pct else 0.015,
                "up_ratio":        up_ratio,
            })
        d += timedelta(days=1)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── 网格搜索 ─────────────────────────────────────────────────

def _arange(lo: float, hi: float, step: float):
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def _score_params(
    samples: pd.DataFrame,
    short_atr_mult: float,
    short_gain_mult: float,
    long_amp_mult: float,   # noqa: ARG001  (reserved for future long scoring)
    long_ma60_buffer: float,  # noqa: ARG001
) -> float:
    """score = (wins - losses) / n。win: hit_5d >= 止盈%; loss: drawdown <= -止损%。"""
    wins = losses = 0
    for _, row in samples.iterrows():
        sell_pct = float(row["up_ratio"]) * short_gain_mult
        stop_pct = float(row["atr_pct"]) * short_atr_mult
        if float(row["hit_5d"]) >= sell_pct:
            wins += 1
        elif float(row["max_drawdown_5d"]) <= -stop_pct:
            losses += 1
    n = len(samples)
    return (wins - losses) / n if n > 0 else -1.0


def _grid_search(samples: pd.DataFrame, cfg: PriceLearnerConfig) -> dict:
    g = cfg.grid
    best_params = dict(cfg.defaults)
    best_score = -float("inf")
    for s_atr in _arange(g["short_atr_mult"]["min"],   g["short_atr_mult"]["max"],   g["short_atr_mult"]["step"]):
        for s_gain in _arange(g["short_gain_mult"]["min"], g["short_gain_mult"]["max"], g["short_gain_mult"]["step"]):
            for l_amp in _arange(g["long_amp_mult"]["min"], g["long_amp_mult"]["max"], g["long_amp_mult"]["step"]):
                for l_buf in _arange(g["long_ma60_buffer"]["min"], g["long_ma60_buffer"]["max"], g["long_ma60_buffer"]["step"]):
                    score = _score_params(samples, s_atr, s_gain, l_amp, l_buf)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "short_atr_mult":   s_atr,
                            "short_gain_mult":  s_gain,
                            "long_amp_mult":    l_amp,
                            "long_ma60_buffer": l_buf,
                        }
    return best_params


# ── 主入口 ───────────────────────────────────────────────────

def fit_price_params(date_str: str, cfg: PriceLearnerConfig | None = None) -> dict:
    cfg = cfg or load_config()
    state_params: dict[str, dict] = {}
    cold_states = []

    for state in ("bull", "bear", "range"):
        samples = _build_buy_samples(date_str, state, cfg.lookback_days)
        if len(samples) < cfg.min_samples:
            state_params[state] = dict(cfg.defaults)
            cold_states.append(state)
        else:
            state_params[state] = _grid_search(samples, cfg)

    result = {
        "version":    1,
        "updated_at": datetime.now().isoformat(),
        **state_params,
    }
    feedback_io.atomic_write_json(PRICE_PARAMS_PATH, result)
    feedback_io.write_feedback("price_learner", date_str, {"cold_states": cold_states})
    status = "cold_start" if len(cold_states) == 3 else "ok"
    return {"status": status, "cold_states": cold_states}


def run(date_str: str | None = None) -> dict:
    """编排器入口。非周日返回 skipped；异常上抛，由 orchestrator 捕获。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    if datetime.strptime(date_str, "%Y-%m-%d").weekday() != 6:
        return {"status": "skipped", "reason": "weekly only (Sunday)"}
    cfg = load_config()
    return fit_price_params(date_str, cfg)
```

- [ ] **Step 5.5: 运行测试**

```bash
pytest tests/test_price_learner.py -v
```
预期：全部 PASSED

- [ ] **Step 5.6: 全量测试**

```bash
pytest tests/ -q
```
预期：全绿

- [ ] **Step 5.7: Commit**

```bash
git add learning/price_learner.py learning/price_learner.yaml tests/test_price_learner.py
git commit -m "feat(p4): price_learner - ATR/振幅系数网格搜索"
```

---

## Task 6：analyser.py 价位参数化

**Files:**
- Modify: `features/analyser.py:530-595`

- [ ] **Step 6.1: 写失败测试**

追加到 `tests/test_price_learner.py`：

```python
# ── analyser integration ─────────────────────────────────────

def test_suggest_dual_period_uses_price_params(tmp_path, monkeypatch):
    """suggest_dual_period_trades 使用 price_params 中的 short_atr_mult。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    cfg_yaml = (
        "lookback_days: 90\nmin_samples: 30\n"
        "grid:\n  short_atr_mult: {min: 1.0, max: 2.5, step: 0.1}\n"
        "  short_gain_mult: {min: 1.5, max: 3.0, step: 0.1}\n"
        "  long_amp_mult: {min: 2.0, max: 4.0, step: 0.2}\n"
        "  long_ma60_buffer: {min: 0.95, max: 0.99, step: 0.01}\n"
        "defaults:\n  short_atr_mult: 1.5\n  short_gain_mult: 2.2\n"
        "  long_amp_mult: 3.0\n  long_ma60_buffer: 0.97\n"
    )
    (tmp_path / "learning" / "price_learner.yaml").write_text(cfg_yaml, encoding="utf-8")
    # 写 price_params.json 指定 short_atr_mult=2.0
    (tmp_path / "learning" / "price_params.json").write_text(
        json.dumps({"range": {"short_atr_mult": 2.0, "short_gain_mult": 2.2,
                              "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}}),
        encoding="utf-8",
    )
    last = {"ma5": 10.0, "ma20": 9.5, "ma60": 9.0,
            "bb_upper": 11.0, "bb_lower": 9.0, "atr": 0.2}
    price_info = {"price": 10.5, "pred_high": 11.2, "pred_low": 9.8,
                  "up_ratio": 2.0, "open": 10.3}

    # monkeypatch load_current_state to return range
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        import learning.market_state as ms
        mp.setattr(ms, "load_current_state", lambda: {"current": "range"})
        from importlib import reload
        import features.analyser as a; reload(a)
        result = a.suggest_dual_period_trades(last, price_info, 0.6)

    short = result["short"]
    # ATR=0.2, short_atr_mult=2.0 → stop = buy - 0.4（比默认 1.5 更宽）
    assert short["stop_price"] < short["buy_price"]
    assert "ATR" in short["stop_desc"]


def test_suggest_dual_period_fallback_no_file(tmp_path, monkeypatch):
    """price_params.json 不存在时使用硬编码默认值，不抛异常。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "learning").mkdir()
    last = {"ma5": 10.0, "ma20": 9.5, "ma60": 9.0,
            "bb_upper": 11.0, "bb_lower": 9.0, "atr": 0.2}
    price_info = {"price": 10.5, "pred_high": 11.2, "pred_low": 9.8,
                  "up_ratio": 2.0, "open": 10.3}
    from importlib import reload
    import features.analyser as a; reload(a)
    result = a.suggest_dual_period_trades(last, price_info, 0.6)
    assert "short" in result
    assert "long"  in result
```

- [ ] **Step 6.2: 运行，确认失败**

```bash
pytest tests/test_price_learner.py -k "analyser" -v
```
预期：`AssertionError`（analyser 还没读 price_params）

- [ ] **Step 6.3: 修改 analyser.py**

在 `features/analyser.py` 的 `suggest_dual_period_trades()` 函数体**最开头**（`def _f` 定义之前）插入：

```python
    # P4: 从 price_params.json 读参数，文件缺失则使用硬编码默认值
    try:
        from learning.price_learner import load_price_params
        from learning.market_state import load_current_state
        _state = load_current_state().get("current", "range")
        _pp = load_price_params(_state)
    except Exception:
        _pp = {"short_atr_mult": 1.5, "short_gain_mult": 2.2,
               "long_amp_mult": 3.0, "long_ma60_buffer": 0.97}
    _SHORT_ATR_MULT   = float(_pp.get("short_atr_mult",  1.5))
    _SHORT_GAIN_MULT  = float(_pp.get("short_gain_mult", 2.2))
    _LONG_AMP_MULT    = float(_pp.get("long_amp_mult",   3.0))
    _LONG_MA60_BUFFER = float(_pp.get("long_ma60_buffer", 0.97))
```

然后替换函数体内的硬编码常量：

| 原代码 | 替换为 |
|--------|--------|
| `s_buy * (1 + up_ratio)` (短线 sell fallback，共 2 处) | `s_buy * (1 + up_ratio * _SHORT_GAIN_MULT)` |
| `atr * 1.5` | `atr * _SHORT_ATR_MULT` |
| `s_stop_desc = f"ATR×1.5 {s_stop}"` | `s_stop_desc = f"ATR×{_SHORT_ATR_MULT} {s_stop}"` |
| `up_ratio * 3` (长线 sell，共 2 处) | `up_ratio * _LONG_AMP_MULT` |
| `ma60 * 0.98` (长线止损) | `ma60 * _LONG_MA60_BUFFER` |
| `l_buy * 0.95` (长线止损 fallback) | `l_buy * (2 - _LONG_MA60_BUFFER)` （等价: 0.97 → 0.03 止损 = 3%） |

> 注意：`l_buy * (2 - _LONG_MA60_BUFFER)` 仅当 `_LONG_MA60_BUFFER` 在 [0.95, 0.99] 时有意义（止损 = 1%-5%）。保留行为一致性。

- [ ] **Step 6.4: 运行测试**

```bash
pytest tests/test_price_learner.py -v
```
预期：全部 PASSED

- [ ] **Step 6.5: 全量测试**

```bash
pytest tests/ -q
```
预期：全绿（analyser 原有测试不应受影响）

- [ ] **Step 6.6: Commit**

```bash
git add features/analyser.py tests/test_price_learner.py
git commit -m "feat(p4): suggest_dual_period_trades 读 price_params.json 替代硬编码常量"
```

---

## Task 7：cmd_scan_bot 集成 alt_data

**Files:**
- Modify: `server/predict_cmd.py`（cmd_scan_bot 函数内部）

- [ ] **Step 7.1: 定位 _scan_one 定义位置**

```bash
grep -n "_scan_one\|def cmd_scan_bot" server/predict_cmd.py | head -20
```

确认 `_scan_one` 定义在 `cmd_scan_bot` 内部（闭包），且 `build_features(df)` 在其中被调用。

- [ ] **Step 7.2: 修改 cmd_scan_bot**

在 `cmd_scan_bot` 函数内，**codes 列表确定之后**、**线程池启动之前**，插入 alt_data 拉取：

```python
    # P3: 批量拉取 alt_data（失败静默降级）
    from datetime import datetime as _dt
    _today = _dt.now().strftime("%Y-%m-%d")
    try:
        from data.alt_fetcher import fetch_alt_features
        alt_cache = fetch_alt_features(codes, _today)
    except Exception:
        alt_cache = {}
```

然后在 `_scan_one` 内的 `build_features(df)` 调用改为：

```python
        df = build_features(df, alt=alt_cache.get(code, {}))
```

- [ ] **Step 7.3: 手动验证（可选）**

```bash
python -c "from server.predict_cmd import cmd_scan_bot; print('import ok')"
```

- [ ] **Step 7.4: 全量测试**

```bash
pytest tests/ -q
```
预期：全绿

- [ ] **Step 7.5: Commit**

```bash
git add server/predict_cmd.py
git commit -m "feat(p3): cmd_scan_bot 扫描前拉取 alt_data，注入 build_features"
```

---

## Task 8：orchestrator 注册 + 最终验证

**Files:**
- Modify: `learning/orchestrator.py`

- [ ] **Step 8.1: 修改 orchestrator.py**

将 `learning/orchestrator.py` 的 import 行改为：

```python
from learning import (
    feedback_io, alerts, market_state, model_learner,
    tactic_learner, blacklist, feature_learner, price_learner,
)
```

在 `MODULES` 列表末尾追加（替换 `# P3/P4 后续阶段追加` 注释）：

```python
    {
        "name":       "feature_learner",
        "run":        feature_learner.run,
        "depends_on": ["market_state"],
    },
    {
        "name":       "price_learner",
        "run":        price_learner.run,
        "depends_on": ["market_state"],
    },
```

在 `check()` 函数的 `files_to_check` 列表末尾追加：

```python
        Path("learning/feature_weights.json"),  # P3 产物
        Path("learning/price_params.json"),      # P4 产物
```

将文件头注释中的 `# P3/P4 后续阶段追加` 行替换为：

```python
    # P3: feature_learner, alt_data
    # P4: price_learner
```

- [ ] **Step 8.2: 运行全量测试**

```bash
pytest tests/ -q
```
预期：全绿，316+ PASSED

- [ ] **Step 8.3: 验证 import 正常**

```bash
python -c "from learning.orchestrator import MODULES; print([m['name'] for m in MODULES])"
```

预期输出包含 `'feature_learner'` 和 `'price_learner'`。

- [ ] **Step 8.4: 验证 learn --check 通过**

```bash
python cli.py learn --check
```
预期：`[check] 所有学习产物验证通过`（或因新 json 文件不存在而跳过，不报错）

- [ ] **Step 8.5: 最终 Commit**

```bash
git add learning/orchestrator.py
git commit -m "feat(p3p4): orchestrator 注册 feature_learner + price_learner，check() 追加产物校验"
```

---

## 验收标准

```bash
# 1. 全量测试通过（包含 P3/P4 新测试）
pytest tests/ -q
# 预期: 345+ PASSED (316 原有 + 29 新增), 0 FAILED

# 2. 模块可正常 import
python -c "from data.alt_fetcher import fetch_alt_features; print('alt_fetcher ok')"
python -c "from learning.feature_learner import run; print('feature_learner ok')"
python -c "from learning.price_learner import run, load_price_params; print('price_learner ok')"

# 3. 编排器模块列表正确
python -c "from learning.orchestrator import MODULES; names=[m['name'] for m in MODULES]; print(names); assert 'feature_learner' in names and 'price_learner' in names"

# 4. 周日手动触发（无数据时返回 cold_start，不抛异常）
python -c "from learning.feature_learner import run; print(run('2026-05-17'))"
python -c "from learning.price_learner import run; print(run('2026-05-17'))"
```
