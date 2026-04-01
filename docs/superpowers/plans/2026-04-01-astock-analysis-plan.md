# A股分析&预测工具 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一款本地运行的 A 股分析与预测命令行工具，支持单股涨跌概率预测与全市场候选股筛选，结果以 HTML 报告形式在浏览器展示。

**Architecture:** 数据层（AKShare 抓取 + Parquet 本地缓存）→ 特征层（pandas-ta 技术指标）→ 模型层（LightGBM 二分类）→ 报告层（Plotly HTML），通过 `cli.py` 统一入口串联各层，各层单向依赖，互不耦合。

**Tech Stack:** Python 3.11+, akshare, pandas, pandas-ta, lightgbm, scikit-learn, plotly, pyarrow, PyYAML, pytest

---

## 文件结构

```
anatent/
├── cli.py                      # 命令行入口（fetch/train/predict/scan/report）
├── config.yaml                 # 股票池、模型参数、路径配置
├── requirements.txt            # 依赖
├── pytest.ini                  # pytest 配置
├── data/
│   ├── __init__.py
│   ├── fetcher.py              # AKShare 数据抓取 + Parquet 缓存
│   └── universe.py             # 股票池读取
├── features/
│   ├── __init__.py
│   ├── technical.py            # 技术指标计算 + FEATURE_COLS 常量
│   └── builder.py              # 特征矩阵构建（调用 technical.py）
├── models/
│   ├── __init__.py
│   ├── trainer.py              # 模型训练（LightGBM）+ 保存
│   └── predictor.py            # 推理：涨跌概率 + 信号
├── reports/
│   ├── __init__.py
│   └── renderer.py             # Plotly HTML 报告生成
└── tests/
    ├── conftest.py             # sys.path 配置
    ├── test_fetcher.py
    ├── test_universe.py
    ├── test_technical.py
    ├── test_builder.py
    ├── test_trainer.py
    ├── test_predictor.py
    └── test_renderer.py
```

---

## Task 1: 项目脚手架

**Files:**
- Create: `anatent/requirements.txt`
- Create: `anatent/pytest.ini`
- Create: `anatent/config.yaml`
- Create: `anatent/data/__init__.py`
- Create: `anatent/features/__init__.py`
- Create: `anatent/models/__init__.py`
- Create: `anatent/reports/__init__.py`
- Create: `anatent/tests/conftest.py`

- [ ] **Step 1: 创建项目根目录结构**

```bash
cd E:\anatent
mkdir -p data/cache features models/saved reports/output tests
touch data/__init__.py features/__init__.py models/__init__.py reports/__init__.py
```

- [ ] **Step 2: 写入 requirements.txt**

```
akshare>=1.12.0
pandas>=2.0.0
numpy>=1.24.0
pandas-ta>=0.3.14b
lightgbm>=4.0.0
scikit-learn>=1.3.0
plotly>=5.18.0
pyarrow>=14.0.0
PyYAML>=6.0
pytest>=7.4.0
pytest-mock>=3.12.0
```

- [ ] **Step 3: 写入 pytest.ini**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 4: 写入 config.yaml**

```yaml
data:
  source: akshare
  cache_dir: data/cache
  default_days: 365

universe:
  watchlist:
    - "600519"   # 贵州茅台
    - "000858"   # 五粮液
    - "601318"   # 中国平安
    - "000001"   # 平安银行
    - "600036"   # 招商银行
  scan_pool: watchlist  # watchlist 或 hs300（后续扩展）

model:
  target_days: 5
  threshold: 0.02
  train_years: 3
  test_months: 3
  saved_dir: models/saved

reports:
  output_dir: reports/output
```

- [ ] **Step 5: 写入 tests/conftest.py**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 6: 安装依赖**

```bash
cd E:\anatent
pip install -r requirements.txt
```

预期：依赖全部安装成功，无报错。

- [ ] **Step 7: 验证 pytest 可运行**

```bash
cd E:\anatent
pytest --collect-only
```

预期：`no tests ran`，无报错。

- [ ] **Step 8: 提交**

```bash
cd E:\anatent
git init
git add .
git commit -m "chore: project scaffold"
```

---

## Task 2: 数据抓取层（fetcher.py）

**Files:**
- Create: `anatent/data/fetcher.py`
- Create: `anatent/tests/test_fetcher.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_fetcher.py`:
```python
import pandas as pd
import pytest
from unittest.mock import patch


def _mock_akshare_df():
    return pd.DataFrame({
        "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "开盘": [1700.0, 1710.0, 1720.0],
        "收盘": [1710.0, 1720.0, 1730.0],
        "最高": [1720.0, 1730.0, 1740.0],
        "最低": [1695.0, 1705.0, 1715.0],
        "成交量": [50000, 55000, 48000],
        "成交额": [8.5e10, 9.4e10, 8.3e10],
        "换手率": [0.4, 0.44, 0.38],
    })


def test_fetch_returns_correct_columns(tmp_path):
    with patch("data.fetcher.CACHE_DIR", tmp_path), \
         patch("data.fetcher._fetch_from_akshare", return_value=_mock_akshare_df().rename(
             columns={"日期": "date", "开盘": "open", "收盘": "close",
                      "最高": "high", "最低": "low", "成交量": "volume",
                      "成交额": "amount", "换手率": "turnover"}
         ).assign(date=lambda df: pd.to_datetime(df["date"]))[
             ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
         ]):
        from data.fetcher import fetch_stock_hist
        df = fetch_stock_hist("600519", days=30)

    assert not df.empty
    for col in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]:
        assert col in df.columns, f"Missing column: {col}"


def test_fetch_caches_to_parquet(tmp_path):
    normalized = _mock_akshare_df().rename(
        columns={"日期": "date", "开盘": "open", "收盘": "close",
                 "最高": "high", "最低": "low", "成交量": "volume",
                 "成交额": "amount", "换手率": "turnover"}
    ).assign(date=lambda df: pd.to_datetime(df["date"]))[
        ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    ]

    with patch("data.fetcher.CACHE_DIR", tmp_path), \
         patch("data.fetcher._fetch_from_akshare", return_value=normalized):
        from data.fetcher import fetch_stock_hist
        fetch_stock_hist("600519", days=30)

    assert (tmp_path / "600519.parquet").exists()


def test_fetch_uses_cache_on_second_call(tmp_path):
    normalized = _mock_akshare_df().rename(
        columns={"日期": "date", "开盘": "open", "收盘": "close",
                 "最高": "high", "最低": "low", "成交量": "volume",
                 "成交额": "amount", "换手率": "turnover"}
    ).assign(date=lambda df: pd.to_datetime(df["date"]))[
        ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    ]
    # Make last date = today so cache is fresh
    normalized["date"] = pd.Timestamp.today().normalize()

    with patch("data.fetcher.CACHE_DIR", tmp_path), \
         patch("data.fetcher._fetch_from_akshare", return_value=normalized) as mock_fetch:
        from data.fetcher import fetch_stock_hist
        fetch_stock_hist("600519", days=30)
        fetch_stock_hist("600519", days=30)

    assert mock_fetch.call_count == 1  # 第二次应命中缓存
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd E:\anatent
pytest tests/test_fetcher.py -v
```

预期：`ImportError: No module named 'data.fetcher'`

- [ ] **Step 3: 实现 fetcher.py**

`data/fetcher.py`:
```python
import akshare as ak
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"

_RENAME_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount", "换手率": "turnover",
}
_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def fetch_stock_hist(code: str, days: int = 365) -> pd.DataFrame:
    """拉取 A 股日线数据，优先使用本地 Parquet 缓存，增量更新。"""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{code}.parquet"
    today = pd.Timestamp.today().normalize()

    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached["date"] = pd.to_datetime(cached["date"])
        last_date = cached["date"].max()
        if last_date >= today - pd.Timedelta(days=1):
            return cached
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        new_data = _fetch_from_akshare(code, start_date, today.strftime("%Y%m%d"))
        if not new_data.empty:
            df = pd.concat([cached, new_data]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
            df.to_parquet(cache_file, index=False)
            return df
        return cached

    start = (today - pd.Timedelta(days=days)).strftime("%Y%m%d")
    df = _fetch_from_akshare(code, start, today.strftime("%Y%m%d"))
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def _fetch_from_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start_date, end_date=end_date,
        adjust="qfq",
    )
    df = raw.rename(columns=_RENAME_MAP)
    df["date"] = pd.to_datetime(df["date"])
    return df[[c for c in _COLS if c in df.columns]]
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd E:\anatent
pytest tests/test_fetcher.py -v
```

预期：3 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: data fetcher with akshare and parquet cache"
```

---

## Task 3: 股票池管理（universe.py）

**Files:**
- Create: `anatent/data/universe.py`
- Create: `anatent/tests/test_universe.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_universe.py`:
```python
import pytest


def test_load_universe_returns_watchlist():
    from data.universe import load_universe
    config = {
        "universe": {
            "watchlist": ["600519", "000858"],
            "scan_pool": "watchlist",
        }
    }
    result = load_universe(config)
    assert result == ["600519", "000858"]


def test_load_universe_unknown_pool_raises():
    from data.universe import load_universe
    config = {
        "universe": {
            "watchlist": ["600519"],
            "scan_pool": "unknown_pool",
        }
    }
    with pytest.raises(ValueError, match="Unknown scan_pool"):
        load_universe(config)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_universe.py -v
```

预期：`ImportError: No module named 'data.universe'`

- [ ] **Step 3: 实现 universe.py**

`data/universe.py`:
```python
def load_universe(config: dict) -> list:
    """根据 config 返回股票代码列表。"""
    pool = config["universe"].get("scan_pool", "watchlist")
    if pool == "watchlist":
        return config["universe"]["watchlist"]
    raise ValueError(f"Unknown scan_pool: {pool}. Supported: watchlist")
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_universe.py -v
```

预期：2 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add data/universe.py tests/test_universe.py
git commit -m "feat: stock universe management"
```

---

## Task 4: 技术指标计算（technical.py）

**Files:**
- Create: `anatent/features/technical.py`
- Create: `anatent/tests/test_technical.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_technical.py`:
```python
import pandas as pd
import numpy as np
import pytest


def _make_ohlcv(n=120):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - np.random.uniform(0, 1, n),
        "high": close + np.random.uniform(0, 2, n),
        "low": close - np.random.uniform(0, 2, n),
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })


def test_add_indicators_returns_ma_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["ma5", "ma10", "ma20", "ma60"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_macd_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["macd_dif", "macd_dea", "macd_hist"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_rsi_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["rsi6", "rsi12", "rsi24"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_bollinger_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["bb_upper", "bb_mid", "bb_lower", "bb_width"]:
        assert col in result.columns, f"Missing {col}"


def test_add_indicators_returns_volume_columns():
    from features.technical import add_indicators
    df = _make_ohlcv()
    result = add_indicators(df)
    for col in ["vol_ratio", "obv", "atr"]:
        assert col in result.columns, f"Missing {col}"


def test_feature_cols_all_present_after_indicators():
    from features.technical import add_indicators, FEATURE_COLS
    df = _make_ohlcv()
    result = add_indicators(df)
    missing = [c for c in FEATURE_COLS if c not in result.columns]
    assert not missing, f"FEATURE_COLS missing from output: {missing}"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_technical.py -v
```

预期：`ImportError: No module named 'features.technical'`

- [ ] **Step 3: 实现 technical.py**

`features/technical.py`:
```python
import pandas as pd
import pandas_ta as ta

FEATURE_COLS = [
    # 趋势均线
    "ma5", "ma10", "ma20", "ma60",
    # 均线偏离度
    "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev",
    # MACD
    "macd_dif", "macd_dea", "macd_hist",
    # RSI
    "rsi6", "rsi12", "rsi24",
    # KDJ
    "kdj_k", "kdj_d", "kdj_j",
    # CCI
    "cci",
    # 布林带
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    # 波动
    "atr",
    # 成交量
    "vol_ratio", "obv",
    # 价格衍生
    "pct_change", "turnover",
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标，返回带有所有指标列的 DataFrame（不删除 NaN 行）。"""
    df = df.copy()

    # 均线
    for period in [5, 10, 20, 60]:
        df[f"ma{period}"] = ta.sma(df["close"], length=period)

    # 均线偏离度
    for period in [5, 10, 20, 60]:
        df[f"ma{period}_dev"] = (df["close"] - df[f"ma{period}"]) / df[f"ma{period}"]

    # MACD
    macd = ta.macd(df["close"])
    df["macd_dif"] = macd["MACD_12_26_9"]
    df["macd_dea"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # RSI
    for period in [6, 12, 24]:
        df[f"rsi{period}"] = ta.rsi(df["close"], length=period)

    # KDJ（用 Stochastic 近似）
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["kdj_k"] = stoch.iloc[:, 0]
        df["kdj_d"] = stoch.iloc[:, 1]
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    else:
        df["kdj_k"] = float("nan")
        df["kdj_d"] = float("nan")
        df["kdj_j"] = float("nan")

    # CCI
    df["cci"] = ta.cci(df["high"], df["low"], df["close"])

    # 布林带（20 日，2 倍标准差）
    bb = ta.bbands(df["close"], length=20)
    if bb is not None and not bb.empty:
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_mid"] = bb["BBM_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    else:
        df["bb_upper"] = float("nan")
        df["bb_mid"] = float("nan")
        df["bb_lower"] = float("nan")
        df["bb_width"] = float("nan")

    # ATR
    df["atr"] = ta.atr(df["high"], df["low"], df["close"])

    # 量比（当日成交量 / 5 日均量）
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(5).mean()

    # OBV
    df["obv"] = ta.obv(df["close"], df["volume"])

    # 价格衍生
    df["pct_change"] = df["close"].pct_change()

    return df
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_technical.py -v
```

预期：6 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add features/technical.py tests/test_technical.py
git commit -m "feat: technical indicators with FEATURE_COLS"
```

---

## Task 5: 特征矩阵构建（builder.py）

**Files:**
- Create: `anatent/features/builder.py`
- Create: `anatent/tests/test_builder.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_builder.py`:
```python
import pandas as pd
import numpy as np


def _make_ohlcv(n=120):
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })


def test_build_features_has_no_leading_nan_rows():
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    df = _make_ohlcv(120)
    result = build_features(df)
    # 所有 FEATURE_COLS 在结果中无全空列
    for col in FEATURE_COLS:
        assert col in result.columns
    # 最后几行不应全是 NaN
    last = result.tail(10)[FEATURE_COLS]
    assert not last.isnull().all(axis=None)


def test_build_features_preserves_ohlcv():
    from features.builder import build_features
    df = _make_ohlcv(120)
    result = build_features(df)
    for col in ["date", "open", "high", "low", "close", "volume"]:
        assert col in result.columns
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_builder.py -v
```

预期：`ImportError: No module named 'features.builder'`

- [ ] **Step 3: 实现 builder.py**

`features/builder.py`:
```python
import pandas as pd
from features.technical import add_indicators


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在原始 OHLCV 数据上计算所有技术指标，返回完整 DataFrame（保留 NaN 行，由调用方处理）。"""
    return add_indicators(df)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_builder.py -v
```

预期：2 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add features/builder.py tests/test_builder.py
git commit -m "feat: feature matrix builder"
```

---

## Task 6: 模型训练（trainer.py）

**Files:**
- Create: `anatent/models/trainer.py`
- Create: `anatent/tests/test_trainer.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_trainer.py`:
```python
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def _make_training_df(n=500):
    np.random.seed(1)
    from features.technical import FEATURE_COLS, add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


def test_build_labels_correct_shape():
    from models.trainer import build_labels
    df = _make_training_df()
    labels = build_labels(df, target_days=5, threshold=0.02)
    assert len(labels) == len(df)
    assert labels.isin([0, 1]).all()


def test_train_returns_booster(tmp_path):
    import lightgbm as lgb
    from models.trainer import train, build_labels
    from features.technical import FEATURE_COLS
    df = _make_training_df(n=500)
    df["code"] = "600519"
    # label 必须在合并前按股票单独计算，否则 shift 会跨越股票边界
    df["label"] = build_labels(df, target_days=5, threshold=0.02)
    model = train(df, feature_cols=FEATURE_COLS)
    assert hasattr(model, "predict")


def test_save_and_load_model(tmp_path):
    import lightgbm as lgb
    from models.trainer import train, build_labels, save_model, load_latest_model
    from features.technical import FEATURE_COLS
    df = _make_training_df(n=500)
    df["code"] = "600519"
    df["label"] = build_labels(df, target_days=5, threshold=0.02)
    model = train(df, feature_cols=FEATURE_COLS)
    path = save_model(model, saved_dir=str(tmp_path))
    loaded = load_latest_model(saved_dir=str(tmp_path))
    assert hasattr(loaded, "predict")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_trainer.py -v
```

预期：`ImportError: No module named 'models.trainer'`

- [ ] **Step 3: 实现 trainer.py**

`models/trainer.py`:
```python
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_labels(df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """计算未来 N 天涨幅是否超过阈值（1=涨，0=不涨）。"""
    future_return = df["close"].shift(-target_days) / df["close"] - 1
    return (future_return > threshold).astype(int)


def train(
    df: pd.DataFrame,
    feature_cols: list,
) -> lgb.Booster:
    """训练 LightGBM 二分类模型。df 须已含 'label' 列（由调用方按股票单独计算）。"""
    df = df.dropna(subset=feature_cols + ["label"])

    split_date = df["date"].max() - pd.DateOffset(months=3)
    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    if not test_df.empty and y_test.nunique() > 1:
        auc = roc_auc_score(y_test, model.predict(X_test))
        print(f"  Test AUC: {auc:.4f}")

    return model


def save_model(model: lgb.Booster, saved_dir: str = "models/saved") -> str:
    Path(saved_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = Path(saved_dir) / f"model_{date_str}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")
    return str(path)


def load_latest_model(saved_dir: str = "models/saved") -> lgb.Booster:
    files = sorted(Path(saved_dir).glob("model_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No model found in {saved_dir}. Run: python cli.py train")
    with open(files[-1], "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_trainer.py -v
```

预期：3 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add models/trainer.py tests/test_trainer.py
git commit -m "feat: LightGBM model training and persistence"
```

---

## Task 7: 模型推理（predictor.py）

**Files:**
- Create: `anatent/models/predictor.py`
- Create: `anatent/tests/test_predictor.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_predictor.py`:
```python
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


def _make_feature_df(n=100):
    np.random.seed(2)
    from features.technical import FEATURE_COLS, add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


class _MockModel:
    def __init__(self, prob):
        self._prob = prob

    def predict(self, X):
        return np.full(len(X), self._prob)


def test_predict_returns_required_keys():
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.7)):
        result = predict(df, FEATURE_COLS)
    for key in ["rise_prob", "fall_prob", "confidence", "signal"]:
        assert key in result


def test_predict_probs_sum_to_one():
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.65)):
        result = predict(df, FEATURE_COLS)
    assert abs(result["rise_prob"] + result["fall_prob"] - 1.0) < 1e-6


def test_predict_signal_buy_when_high_prob():
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.75)):
        result = predict(df, FEATURE_COLS)
    assert result["signal"] == "买入"
    assert result["confidence"] == "高"


def test_predict_signal_avoid_when_low_prob():
    from models.predictor import predict
    from features.technical import FEATURE_COLS
    df = _make_feature_df()
    with patch("models.predictor.load_latest_model", return_value=_MockModel(0.25)):
        result = predict(df, FEATURE_COLS)
    assert result["signal"] == "回避"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_predictor.py -v
```

预期：`ImportError: No module named 'models.predictor'`

- [ ] **Step 3: 实现 predictor.py**

`models/predictor.py`:
```python
from models.trainer import load_latest_model


def predict(df, feature_cols: list, model_dir: str = "models/saved") -> dict:
    """
    对给定股票的最新数据行进行预测。

    Returns:
        dict with keys: rise_prob, fall_prob, confidence, signal
    """
    model = load_latest_model(model_dir)
    valid = df[feature_cols].dropna()
    if valid.empty:
        raise ValueError("No valid rows after dropping NaN — need more history data.")

    prob = float(model.predict(valid.iloc[[-1]])[0])

    if prob >= 0.65:
        confidence = "高"
    elif prob >= 0.5:
        confidence = "中"
    else:
        confidence = "低"

    if prob >= 0.6:
        signal = "买入"
    elif prob >= 0.4:
        signal = "观望"
    else:
        signal = "回避"

    return {
        "rise_prob": round(prob, 4),
        "fall_prob": round(1 - prob, 4),
        "confidence": confidence,
        "signal": signal,
    }
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_predictor.py -v
```

预期：4 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add models/predictor.py tests/test_predictor.py
git commit -m "feat: model predictor with signal classification"
```

---

## Task 8: HTML 报告渲染（renderer.py）

**Files:**
- Create: `anatent/reports/renderer.py`
- Create: `anatent/tests/test_renderer.py`

- [ ] **Step 1: 写入失败测试**

`tests/test_renderer.py`:
```python
import pandas as pd
import numpy as np
from pathlib import Path


def _make_chart_df(n=120):
    np.random.seed(3)
    from features.technical import add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


_PREDICTION = {
    "rise_prob": 0.72,
    "fall_prob": 0.28,
    "confidence": "高",
    "signal": "买入",
}


def test_render_creates_html_file(tmp_path):
    from reports.renderer import render_report
    df = _make_chart_df()
    path = render_report("600519", df, _PREDICTION, output_dir=str(tmp_path))
    assert Path(path).exists()
    assert path.endswith(".html")


def test_render_html_contains_code(tmp_path):
    from reports.renderer import render_report
    df = _make_chart_df()
    path = render_report("600519", df, _PREDICTION, output_dir=str(tmp_path))
    content = Path(path).read_text(encoding="utf-8")
    assert "600519" in content


def test_render_html_contains_signal(tmp_path):
    from reports.renderer import render_report
    df = _make_chart_df()
    path = render_report("600519", df, _PREDICTION, output_dir=str(tmp_path))
    content = Path(path).read_text(encoding="utf-8")
    assert "买入" in content
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_renderer.py -v
```

预期：`ImportError: No module named 'reports.renderer'`

- [ ] **Step 3: 实现 renderer.py**

`reports/renderer.py`:
```python
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_report(
    code: str,
    df: pd.DataFrame,
    prediction: dict,
    output_dir: str = "reports/output",
) -> str:
    """生成包含 K线 + 指标 + 预测结果的单文件 HTML 报告。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.18, 0.16, 0.16],
        subplot_titles=["K线 + 均线", "成交量", "MACD", "RSI6"],
        vertical_spacing=0.04,
    )

    # K线图
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线",
        increasing_line_color="red", decreasing_line_color="green",
    ), row=1, col=1)

    # 均线
    ma_colors = {5: "#1f77b4", 10: "#ff7f0e", 20: "#2ca02c", 60: "#d62728"}
    for period, color in ma_colors.items():
        col_name = f"ma{period}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col_name], name=f"MA{period}",
                line=dict(color=color, width=1),
            ), row=1, col=1)

    # 成交量
    vol_colors = [
        "red" if c >= o else "green"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="成交量",
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # MACD
    if "macd_dif" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_dif"], name="DIF",
            line=dict(color="#1f77b4", width=1),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_dea"], name="DEA",
            line=dict(color="#ff7f0e", width=1),
        ), row=3, col=1)
        hist_colors = ["red" if v >= 0 else "green" for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["macd_hist"], name="HIST",
            marker_color=hist_colors, showlegend=False,
        ), row=3, col=1)

    # RSI
    if "rsi6" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi6"], name="RSI6",
            line=dict(color="#9467bd", width=1),
        ), row=4, col=1)
        # 超买/超卖参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    # 标题含预测结果
    signal_emoji = {"买入": "▲", "观望": "●", "回避": "▼"}
    emoji = signal_emoji.get(prediction["signal"], "")
    title = (
        f"{code}  {emoji} {prediction['signal']}  "
        f"涨概率: {prediction['rise_prob']:.1%}  "
        f"置信度: {prediction['confidence']}"
    )

    fig.update_layout(
        title=title,
        height=900,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(type="category")

    output_path = str(Path(output_dir) / f"{code}_report.html")
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_renderer.py -v
```

预期：3 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add reports/renderer.py tests/test_renderer.py
git commit -m "feat: plotly HTML report renderer"
```

---

## Task 9: CLI 命令行入口（cli.py）

**Files:**
- Create: `anatent/cli.py`

- [ ] **Step 1: 实现 cli.py**

`cli.py`:
```python
"""
A股分析预测工具 CLI

用法:
  python cli.py fetch [--code 600519] [--days 365]
  python cli.py train
  python cli.py predict 600519
  python cli.py scan [--top 20]
  python cli.py report 600519
"""
import argparse
import sys
import webbrowser
from pathlib import Path

import yaml


def _load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── fetch ──────────────────────────────────────────────────────────────────

def cmd_fetch(args, config):
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe

    codes = [args.code] if args.code else load_universe(config)
    days = args.days or config["data"]["default_days"]

    for code in codes:
        print(f"  Fetching {code} ...")
        df = fetch_stock_hist(code, days=days)
        print(f"    {code}: {len(df)} rows, latest {df['date'].max().date()}")
    print("Done.")


# ── train ──────────────────────────────────────────────────────────────────

def cmd_train(args, config):
    import pandas as pd
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.trainer import build_labels, train, save_model

    codes = load_universe(config)
    days = config["data"]["default_days"]
    target_days = config["model"]["target_days"]
    threshold = config["model"]["threshold"]

    dfs = []
    for code in codes:
        print(f"  Loading {code} ...")
        df = fetch_stock_hist(code, days=days)
        df = build_features(df)
        # 必须在合并前按股票单独计算 label，否则 shift 会跨越股票边界
        df["label"] = build_labels(df, target_days=target_days, threshold=threshold)
        df["code"] = code
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Training on {len(combined)} rows ...")
    model = train(combined, feature_cols=FEATURE_COLS)
    save_model(model, saved_dir=config["model"]["saved_dir"])


# ── predict ────────────────────────────────────────────────────────────────

def cmd_predict(args, config):
    from data.fetcher import fetch_stock_hist
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict
    from reports.renderer import render_report

    df = fetch_stock_hist(args.code, days=365)
    df = build_features(df)
    result = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])

    print(f"\n{'='*40}")
    print(f"  股票代码:  {args.code}")
    print(f"  信号:      {result['signal']}")
    print(f"  涨概率:    {result['rise_prob']:.1%}")
    print(f"  跌概率:    {result['fall_prob']:.1%}")
    print(f"  置信度:    {result['confidence']}")
    print(f"{'='*40}\n")

    path = render_report(
        args.code, df.tail(120), result,
        output_dir=config["reports"]["output_dir"],
    )
    print(f"  报告: {path}")
    webbrowser.open(f"file:///{Path(path).resolve()}")


# ── scan ───────────────────────────────────────────────────────────────────

def cmd_scan(args, config):
    from data.fetcher import fetch_stock_hist
    from data.universe import load_universe
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict

    codes = load_universe(config)
    top_n = args.top

    results = []
    for code in codes:
        try:
            df = fetch_stock_hist(code, days=365)
            df = build_features(df)
            r = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])
            results.append({"code": code, **r})
            print(f"  {code}: {r['signal']} ({r['rise_prob']:.1%})")
        except Exception as e:
            print(f"  {code}: 跳过 — {e}")

    results.sort(key=lambda x: x["rise_prob"], reverse=True)
    top = results[:top_n]

    print(f"\n{'='*50}")
    print(f"  Top {top_n} 候选股（按涨概率排序）")
    print(f"{'='*50}")
    print(f"  {'代码':<10}{'信号':<8}{'涨概率':<10}置信度")
    print(f"  {'-'*46}")
    for r in top:
        print(f"  {r['code']:<10}{r['signal']:<8}{r['rise_prob']:.1%}      {r['confidence']}")
    print()


# ── report ─────────────────────────────────────────────────────────────────

def cmd_report(args, config):
    from data.fetcher import fetch_stock_hist
    from features.builder import build_features
    from features.technical import FEATURE_COLS
    from models.predictor import predict
    from reports.renderer import render_report

    df = fetch_stock_hist(args.code, days=365)
    df = build_features(df)
    result = predict(df, FEATURE_COLS, model_dir=config["model"]["saved_dir"])
    path = render_report(
        args.code, df.tail(120), result,
        output_dir=config["reports"]["output_dir"],
    )
    print(f"Report: {path}")
    webbrowser.open(f"file:///{Path(path).resolve()}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="anatent",
        description="A股分析&预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("fetch", help="拉取/更新历史数据")
    p.add_argument("--code", help="股票代码（不指定则更新全部关注列表）")
    p.add_argument("--days", type=int, help="历史天数（默认读取 config.yaml）")

    sub.add_parser("train", help="训练预测模型")

    p = sub.add_parser("predict", help="预测单只股票并生成 HTML 报告")
    p.add_argument("code", help="股票代码，如 600519")

    p = sub.add_parser("scan", help="扫描股票池，输出 Top N 候选股")
    p.add_argument("--top", type=int, default=20, help="输出数量（默认 20）")

    p = sub.add_parser("report", help="生成 HTML 报告（不重新预测）")
    p.add_argument("code", help="股票代码")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = _load_config()
    dispatch = {
        "fetch": cmd_fetch,
        "train": cmd_train,
        "predict": cmd_predict,
        "scan": cmd_scan,
        "report": cmd_report,
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行全量测试，确认全部通过**

```bash
cd E:\anatent
pytest -v
```

预期：所有测试 PASS，无 FAIL/ERROR。

- [ ] **Step 3: 冒烟测试 — 验证帮助信息**

```bash
python cli.py --help
python cli.py fetch --help
python cli.py predict --help
```

预期：每个命令均输出正确的帮助文本，无报错。

- [ ] **Step 4: 提交**

```bash
git add cli.py
git commit -m "feat: CLI entry point connecting all layers"
```

---

## Task 10: 端到端验证

- [ ] **Step 1: 拉取数据**

```bash
cd E:\anatent
python cli.py fetch --code 600519 --days 365
```

预期：终端输出 `600519: NNN rows, latest YYYY-MM-DD`，`data/cache/600519.parquet` 文件存在。

- [ ] **Step 2: 对更多关注列表股票拉取数据**

```bash
python cli.py fetch --days 365
```

预期：config.yaml 中所有股票均被拉取。

- [ ] **Step 3: 训练模型**

```bash
python cli.py train
```

预期：终端输出 `Test AUC: X.XXXX`，`models/saved/model_YYYYMMDD.pkl` 文件存在。

- [ ] **Step 4: 单股预测**

```bash
python cli.py predict 600519
```

预期：终端输出信号/概率/置信度，浏览器自动打开 K 线 HTML 报告。

- [ ] **Step 5: 筛选候选股**

```bash
python cli.py scan --top 5
```

预期：终端输出 Top 5 候选股表格，按涨概率降序排列。

- [ ] **Step 6: 提交最终版本**

```bash
git add .
git commit -m "feat: end-to-end smoke test passed"
```

---

## 快速参考

```bash
# 初次使用
pip install -r requirements.txt
python cli.py fetch --days 365
python cli.py train
python cli.py predict 600519

# 日常使用
python cli.py fetch                  # 增量更新全部关注股
python cli.py scan --top 10          # 今日筛选候选股
python cli.py predict 000858         # 五粮液单股预测

# 运行测试
pytest -v
```
