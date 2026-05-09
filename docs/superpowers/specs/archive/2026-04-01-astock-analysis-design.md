# A股分析&预测工具 — 设计文档

**版本：** 0.1.0  
**日期：** 2026-04-01  
**定位：** 个人投资决策辅助，命令行触发 + 浏览器查看图表

---

## 背景与目标

构建一款本地运行的 A 股分析与预测工具，核心能力：

1. **单股预测**：输入股票代码，输出未来 1-5 天涨跌概率
2. **全市场筛选**：扫描股票池，输出 Top N 候选股
3. **可视化报告**：K线图 + 技术指标 + 预测结果，生成 HTML，浏览器打开

---

## 项目结构

```
anatent/
├── cli.py                  # 命令行入口
├── config.yaml             # 股票池、模型参数、数据源配置
├── data/
│   ├── fetcher.py          # 数据抓取（AKShare/BaoStock 适配层）
│   ├── cache/              # 本地缓存（Parquet 格式）
│   └── universe.py         # 股票池管理
├── features/
│   ├── technical.py        # 技术指标计算（MA、MACD、RSI、布林带、成交量）
│   └── builder.py          # 特征矩阵构建
├── models/
│   ├── trainer.py          # 模型训练（LightGBM）
│   ├── predictor.py        # 推理：涨跌概率 + 置信度
│   └── saved/              # 持久化模型文件（.pkl）
├── reports/
│   ├── renderer.py         # 生成 HTML 报告（Plotly）
│   └── output/             # 输出 HTML 文件
└── requirements.txt
```

---

## 数据层

**数据源（免费优先）：**
- 主力：`akshare`（A股日线 OHLCV、基本面、资金流向）
- 备用：`baostock`（历史数据稳定，速度较快）
- 未来可升级：Tushare Pro（改 fetcher.py 适配层即可，不影响上层）

**缓存策略：**
- 历史数据以 Parquet 格式存储到 `data/cache/`，按股票代码分文件
- 每次 `fetch` 命令增量更新（只拉取缺失日期）

**股票池：**
- `config.yaml` 中配置关注列表
- `scan` 命令默认扫描关注列表，可选扩展到沪深 300 / 全 A

---

## 特征工程

计算以下技术指标作为模型输入特征：

| 类别 | 指标 |
|---|---|
| 趋势 | MA5/10/20/60、EMA12/26、MACD(DIF/DEA/HIST) |
| 震荡 | RSI6/12/24、KDJ(K/D/J)、CCI |
| 波动 | 布林带（上轨/中轨/下轨/带宽）、ATR |
| 成交量 | 量比、换手率、OBV、VWAP |
| 价格衍生 | 涨跌幅、振幅、与各均线偏离度 |

**标签定义（监督学习）：**
- 预测目标：未来 N 天收盘价相对今日涨幅是否超过阈值（二分类）
- N 和阈值通过 `config.yaml` 配置（默认 N=5，阈值=2%）

---

## 模型层

**算法：** LightGBM（梯度提升树）

**选择理由：**
- 在金融时间序列上表现稳健，无需 GPU
- 训练速度快，本地可快速迭代
- 特征重要性可解释

**训练流程：**
1. 按时间序列切分（训练集：最近 3 年，测试集：最近 3 个月），避免未来数据泄露
2. 输出：涨跌概率（0~1）+ 置信度分级（高/中/低）

**模型文件：** `models/saved/model_YYYYMMDD.pkl`，每次训练保存带日期版本

---

## CLI 命令

```bash
# 拉取/更新数据（默认更新关注列表股票）
python cli.py fetch [--code 600519] [--days 365]

# 训练模型
python cli.py train [--target-days 5] [--threshold 0.02]

# 预测单股（终端输出 + 自动打开 HTML 报告）
python cli.py predict 600519

# 全市场筛选，输出 Top N 候选股
python cli.py scan [--top 20]

# 仅生成 HTML 报告（不重新预测）
python cli.py report 600519
```

---

## 报告输出（HTML）

每份报告包含：

1. **K线图** — 日线 OHLCV + 成交量
2. **技术指标图** — MACD、RSI、布林带叠加
3. **预测结果** — 涨跌概率条形图 + 置信度标签
4. **筛选摘要**（scan 命令）— 候选股表格，按概率排序

渲染库：`Plotly`，生成单文件 HTML，双击即可打开，无需服务器。

---

## 技术栈

| 模块 | 技术选型 |
|---|---|
| 语言 | Python 3.11+ |
| 数据抓取 | akshare、baostock |
| 数据处理 | pandas、numpy |
| 技术指标 | ta-lib 或 pandas-ta |
| 机器学习 | lightgbm、scikit-learn |
| 可视化 | plotly |
| CLI | argparse |
| 缓存格式 | parquet（pyarrow） |
| 配置 | PyYAML |

---

## 实现阶段规划

1. **阶段一**：数据层（fetcher + 缓存 + 股票池）
2. **阶段二**：特征工程（技术指标 + 特征矩阵）
3. **阶段三**：模型层（训练 + 推理）
4. **阶段四**：CLI + HTML 报告输出
5. **阶段五**：scan 全市场筛选

每阶段完成后可独立测试，互不依赖。

---

## 未来迭代方向（不在当前范围）

- 接入 Tushare Pro / Wind 付费数据
- 添加 LSTM / Transformer 深度学习模型
- 情感分析（新闻/公告）
- 持仓管理与风险监控
- Streamlit Web 界面
