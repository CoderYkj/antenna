"""tests/test_alt_fetcher.py — alt_fetcher 单元测试。"""
from pathlib import Path
from unittest.mock import patch

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

    import data.alt_fetcher as m  # noqa: E402
    with patch("data.alt_fetcher._fetch_fund_flow", side_effect=_raise), \
         patch("data.alt_fetcher._fetch_dragon_board", return_value={"000001": 2}), \
         patch("data.alt_fetcher._fetch_sector_heat", return_value={"000001": 0.7}), \
         patch("data.alt_fetcher._fetch_north_flow", return_value={"000001": None}):
        result = m.fetch_alt_features(["000001"], "2026-05-18")

    assert result["000001"]["main_net_in_1d"] is None
    assert result["000001"]["dragon_top_cnt_10d"] == pytest.approx(2.0 / 10)  # normalized


# ── 全部失败返回空 dict ──────────────────────────────────────

def test_all_features_fail_returns_empty_dict_per_code(tmp_path, monkeypatch):
    """所有接口都失败时，code 对应的 dict 全是 None。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache" / "alt").mkdir(parents=True)

    import data.alt_fetcher as m  # 确保模块已加载，patch 作用于当前函数对象
    with patch("data.alt_fetcher._fetch_fund_flow", side_effect=Exception), \
         patch("data.alt_fetcher._fetch_dragon_board", side_effect=Exception), \
         patch("data.alt_fetcher._fetch_sector_heat", side_effect=Exception), \
         patch("data.alt_fetcher._fetch_north_flow", side_effect=Exception):
        result = m.fetch_alt_features(["600519"], "2026-05-18")

    assert result["600519"]["main_net_in_1d"] is None
    assert result["600519"]["north_hold_chg_5d"] is None


# ── 结果写入 parquet 缓存 ────────────────────────────────────

def test_result_cached_to_parquet(tmp_path, monkeypatch):
    """成功拉取后结果写入 data/cache/alt/{date}.parquet。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache" / "alt").mkdir(parents=True)

    import data.alt_fetcher as m  # 确保模块已加载，patch 作用于当前函数对象
    with patch("data.alt_fetcher._fetch_fund_flow", return_value={"600519": (0.2, 0.1)}), \
         patch("data.alt_fetcher._fetch_dragon_board", return_value={"600519": 1}), \
         patch("data.alt_fetcher._fetch_sector_heat", return_value={"600519": 0.6}), \
         patch("data.alt_fetcher._fetch_north_flow", return_value={"600519": 0.05}):
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
