import pandas as pd
from features.technical import add_indicators


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在原始 OHLCV 数据上计算所有技术指标，返回完整 DataFrame（保留 NaN 行，由调用方处理）。"""
    return add_indicators(df)
