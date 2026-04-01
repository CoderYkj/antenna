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

    # 布林带（20 日，2 倍标准差）- 使用位置索引兼容不同版本列名
    bb = ta.bbands(df["close"], length=20)
    if bb is not None and not bb.empty:
        df["bb_lower"] = bb.iloc[:, 0]   # BBL
        df["bb_mid"] = bb.iloc[:, 1]     # BBM
        df["bb_upper"] = bb.iloc[:, 2]   # BBU
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
