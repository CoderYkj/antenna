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
