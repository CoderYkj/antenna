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
