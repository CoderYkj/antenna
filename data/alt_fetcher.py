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
        except Exception as exc:
            logger.debug("[alt_fetcher] fund_flow %s 失败: %s", code, exc)
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
    from data.universe import load_sector_map
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

def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning("[alt_fetcher] %s 失败: %s", getattr(fn, "__name__", repr(fn)), e)
        return None


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
            # 仅当所有请求的 codes 都在缓存中时才返回，否则补充拉取
            if result and set(codes) <= set(result.keys()):
                return result
        except Exception:
            pass

    # ── 拉取各接口（串行，避免频率限制）──────────────────────
    logger.info("[alt_fetcher] 拉取 %d 支股票 alt_data (%s)", len(codes), date_str)

    fund_flow = _try_call(_fetch_fund_flow, codes)
    dragon    = _try_call(_fetch_dragon_board, codes)
    sector    = _try_call(_fetch_sector_heat, codes)
    north     = _try_call(_fetch_north_flow, codes)

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

    # ── 写缓存（合并已有数据，避免丢失历史 codes）───────────────
    try:
        rows = [{"code": code, **feats} for code, feats in result.items()]
        new_df = pd.DataFrame(rows)
        if cache_path.exists():
            try:
                existing_df = pd.read_parquet(cache_path)
                # 以新数据为优先，保留旧数据中不在本次请求中的 codes
                existing_df = existing_df[~existing_df["code"].isin(result.keys())]
                new_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception:
                pass  # 读旧缓存失败，直接用新数据
        new_df.to_parquet(cache_path, index=False)
    except Exception as e:
        logger.warning("[alt_fetcher] 缓存写入失败: %s", e)

    return result
