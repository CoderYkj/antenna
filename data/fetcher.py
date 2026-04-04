import akshare as ak
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"

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


def _code_to_sina_symbol(code: str) -> str:
    """将纯数字代码转为新浪格式：600519 → sh600519，000858 → sz000858。"""
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def _fetch_from_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """使用新浪数据源拉取日线数据（东方财富源在当前环境不可用）。"""
    symbol = _code_to_sina_symbol(code)
    raw = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    raw["date"] = pd.to_datetime(raw["date"])
    # 按日期范围过滤
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
    return df[[c for c in _COLS if c in df.columns]].reset_index(drop=True)
