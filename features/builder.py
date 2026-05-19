import pandas as pd
from features.technical import add_indicators


def build_features(df: pd.DataFrame, alt: dict | None = None) -> pd.DataFrame:
    """在原始 OHLCV 数据上计算所有技术指标，返回完整 DataFrame（保留 NaN 行，由调用方处理）。

    Args:
        df: 原始 OHLCV DataFrame
        alt: 可选字典，包含 alt_data 特征值，这些值将追加到 df 最后一行。
             alt 中的 None 值转为 0.0。alt=None 或空 dict 时行为与原版一致。

    Returns:
        完整特征 DataFrame。如果传入 alt，新的 alt 列将被添加并仅在最后一行填充值。
    """
    df = add_indicators(df)

    if alt:
        last_idx = df.index[-1]
        for col, val in alt.items():
            if col not in df.columns:
                df[col] = float("nan")
            df.loc[last_idx, col] = float(val) if val is not None else 0.0

    return df
