import akshare as ak
import pandas as pd
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = Path(__file__).parent / "cache"

_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]

_SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def fetch_stock_hist(code: str, days: int = 365, cache_only: bool = False) -> pd.DataFrame:
    """拉取 A 股日线数据，优先使用本地 Parquet 缓存，增量更新。
    cache_only=True 时跳过网络请求，直接返回本地缓存（用于批量扫描/训练）。
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{code}.parquet"
    today = pd.Timestamp.today().normalize()

    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached["date"] = pd.to_datetime(cached["date"])
        last_date = cached["date"].max()
        if cache_only or last_date >= today - pd.Timedelta(days=1):
            return cached
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        try:
            new_data = _fetch_from_akshare(code, start_date, today.strftime("%Y%m%d"))
            if not new_data.empty:
                df = pd.concat([cached, new_data]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
                df.to_parquet(cache_file, index=False)
                return df
        except Exception as e:
            print(f"[fetcher] {code} 增量更新失败: {e}，使用本地缓存")
        return cached

    if cache_only:
        raise FileNotFoundError(f"本地无缓存：{code}")

    start = (today - pd.Timedelta(days=days)).strftime("%Y%m%d")
    df = _fetch_from_akshare(code, start, today.strftime("%Y%m%d"))
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def fetch_realtime_price(code: str) -> dict:
    """
    获取单只股票实时行情（新浪 hq API）。
    Returns: {"price": float, "pct": float, "open": float, "high": float, "low": float}
    非交易时段或失败返回空 dict。
    """
    try:
        result = fetch_realtime_prices([code])
        return result.get(code, {})
    except Exception as e:
        print(f"[fetcher] {code} 实时行情获取失败: {e}")
        return {}


def fetch_realtime_prices(codes: list) -> dict:
    """
    批量获取多只股票实时行情（新浪 hq API，一次请求）。
    Returns: {code: {"price", "pct", "open", "high", "low"}}
    非交易时段价格为 0，对应股票会被过滤掉。
    """
    if not codes:
        return {}
    symbols_map = {_code_to_sina_symbol(c): c for c in codes}
    symbols_str = ",".join(symbols_map.keys())
    url = f"https://hq.sinajs.cn/list={symbols_str}"
    req = urllib.request.Request(url, headers=_SINA_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode("gbk", errors="replace")

    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or '"' not in line:
            continue
        # var hq_str_sz002174="name,open,prev_close,price,high,low,...,date,time,00"
        sym_part = line.split("=")[0].strip()          # var hq_str_sz002174
        sym = sym_part.split("_")[-1]                  # sz002174
        code = symbols_map.get(sym, sym[-6:])
        value = line.split('"')[1]
        parts = value.split(",")
        if len(parts) < 6:
            continue
        try:
            price = float(parts[3]) if parts[3] else 0.0
        except ValueError:
            continue
        if price == 0:
            continue  # 非交易时段或停牌
        prev_close = float(parts[2]) if parts[2] else 0.0
        pct = (price / prev_close - 1) * 100 if prev_close else 0.0
        result[code] = {
            "name":  parts[0].strip() if parts[0] else code,
            "price": price,
            "open":  float(parts[1]) if parts[1] else price,
            "high":  float(parts[4]) if parts[4] else price,
            "low":   float(parts[5]) if parts[5] else price,
            "pct":   round(pct, 2),
        }
    return result



def fetch_intraday_kline(code: str, period_min: int = 5) -> pd.DataFrame:
    """
    获取当日分时 K 线（tick 聚合为 N 分钟棒）。
    Returns: DataFrame with columns [time, open, high, low, close, volume]
    非交易时段或数据不足时返回空 DataFrame。
    """
    try:
        symbol = _code_to_sina_symbol(code)  # sz002174
        df = ak.stock_zh_a_tick_tx_js(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        # 列：成交时间 成交价格 价格变动 成交量 成交金额 性质
        df = df.rename(columns={
            df.columns[0]: "time",
            df.columns[1]: "price",
            df.columns[3]: "volume",
        })
        df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["price", "time"])
        if df.empty:
            return pd.DataFrame()

        today = pd.Timestamp.today().normalize()
        df["datetime"] = pd.to_datetime(
            today.strftime("%Y-%m-%d") + " " + df["time"].astype(str),
            errors="coerce",
        )
        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

        # 按 N 分钟聚合
        rule = f"{period_min}min"
        agg = df["price"].resample(rule, closed="right", label="right").ohlc()
        vol = df["volume"].resample(rule, closed="right", label="right").sum()
        agg["volume"] = vol
        agg = agg.dropna(subset=["open"])
        agg.index.name = "time"
        return agg.reset_index()
    except Exception as e:
        print(f"[fetcher] {code} 分时数据获取失败: {e}")
        return pd.DataFrame()


def fetch_batch_parallel(codes: list, days: int = 365, workers: int = 8) -> dict:
    """并行拉取多只股票，返回 {code: 成功/跳过/失败} 统计。"""
    CACHE_DIR.mkdir(exist_ok=True)
    stats = {"成功": 0, "跳过": 0, "失败": 0}

    def _fetch_one(code):
        try:
            df = fetch_stock_hist(code, days=days)
            return code, "成功" if not df.empty else "跳过"
        except Exception as e:
            return code, f"失败:{str(e)[:40]}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, code): code for code in codes}
        for future in as_completed(futures):
            code, result = future.result()
            key = result if result in stats else "失败"
            stats[key] += 1
    return stats


def fetch_financial_data(code: str) -> dict:
    """
    拉取 A 股财务数据，返回三张表 + 关键指标摘要。

    Returns: {
        "abstract":  DataFrame,   # 关键指标（新浪）：ROE、毛利率、EPS 等
        "profit":    DataFrame,   # 利润表（同花顺）：营收、净利润等
        "balance":   DataFrame,   # 资产负债表（同花顺）
        "cashflow":  DataFrame,   # 现金流量表（同花顺）
        "indicator": DataFrame,   # 综合财务分析指标（东方财富）
    }
    缺失的表返回空 DataFrame，不抛异常。
    """
    result = {k: pd.DataFrame() for k in ("abstract", "profit", "balance", "cashflow", "indicator")}

    # 同花顺格式代码（不带前缀）
    ths_code = code

    # 东财格式代码（带交易所后缀）
    if code.startswith("6"):
        em_suffix = ".SH"
    elif code.startswith("8") or code.startswith("4"):
        em_suffix = ".BJ"   # 北交所（8x / 4x 开头）
    else:
        em_suffix = ".SZ"
    em_code   = f"{code}{em_suffix}"

    try:
        result["abstract"] = ak.stock_financial_abstract(symbol=code)
    except Exception as e:
        print(f"[fetcher] {code} abstract 失败: {e}")

    try:
        result["profit"] = ak.stock_financial_benefit_ths(symbol=ths_code, indicator="按报告期")
    except Exception as e:
        print(f"[fetcher] {code} profit 失败: {e}")

    try:
        result["balance"] = ak.stock_financial_debt_ths(symbol=ths_code, indicator="按报告期")
    except Exception as e:
        print(f"[fetcher] {code} balance 失败: {e}")

    try:
        result["cashflow"] = ak.stock_financial_cash_ths(symbol=ths_code, indicator="按报告期")
    except Exception as e:
        print(f"[fetcher] {code} cashflow 失败: {e}")

    try:
        result["indicator"] = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
    except Exception as e:
        print(f"[fetcher] {code} indicator 失败: {e}")

    return result


def cached_codes() -> list:
    """返回本地已缓存的股票代码列表。"""
    return [f.stem for f in CACHE_DIR.glob("*.parquet")]


# ── 股票名称映射 ──────────────────────────────────────────────
NAME_MAP_FILE = Path(__file__).parent / "name_map.json"
_NAME_MAP_CACHE: dict = {}       # 进程内缓存
_NAME_MAP_CACHE_TS: float = 0.0  # 最后加载时间


def _load_name_map() -> dict:
    """加载本地股票名称映射（code→name），进程内缓存 1 小时；文件缓存 24 小时。"""
    import json, time
    global _NAME_MAP_CACHE, _NAME_MAP_CACHE_TS
    now = time.time()
    # 进程内缓存有效时直接返回
    if _NAME_MAP_CACHE and now - _NAME_MAP_CACHE_TS < 3600:
        return _NAME_MAP_CACHE
    # 文件存在且 <24h，读文件
    if NAME_MAP_FILE.exists():
        if now - NAME_MAP_FILE.stat().st_mtime < 86400:
            with open(NAME_MAP_FILE, encoding="utf-8") as f:
                _NAME_MAP_CACHE = json.load(f)
            _NAME_MAP_CACHE_TS = now
            return _NAME_MAP_CACHE
    # 文件过期，尝试刷新；失败时 fallback 到旧文件
    fresh = _refresh_name_map()
    if fresh:
        _NAME_MAP_CACHE = fresh
        _NAME_MAP_CACHE_TS = now
        return _NAME_MAP_CACHE
    # akshare 刷新失败，降级读旧文件
    if NAME_MAP_FILE.exists():
        try:
            with open(NAME_MAP_FILE, encoding="utf-8") as f:
                _NAME_MAP_CACHE = json.load(f)
            _NAME_MAP_CACHE_TS = now
            return _NAME_MAP_CACHE
        except Exception:
            pass
    return {}


def _refresh_name_map() -> dict:
    """从 akshare 拉取全量 A 股代码+名称，写入本地缓存。"""
    import json
    try:
        df = ak.stock_info_a_code_name()
        mapping = dict(zip(df["code"].astype(str).str.zfill(6), df["name"]))
    except Exception as e:
        print(f"[fetcher] 名称映射刷新失败: {e}")
        mapping = {}
    if mapping:
        with open(NAME_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
    return mapping


# ── 行业板块映射 ──────────────────────────────────────────────
SECTOR_MAP_FILE = Path(__file__).parent / "sector_map.json"
_SECTOR_CACHE: dict = {}          # 进程内缓存
_HOT_SECTORS_CACHE: tuple = (0.0, [])   # (timestamp, list)


def _load_sector_map() -> dict:
    """加载申万一级行业映射（code_6→industry_name），7 天内复用缓存。"""
    import json, time
    global _SECTOR_CACHE
    if _SECTOR_CACHE:
        return _SECTOR_CACHE
    if SECTOR_MAP_FILE.exists():
        if time.time() - SECTOR_MAP_FILE.stat().st_mtime < 86400 * 7:
            with open(SECTOR_MAP_FILE, encoding="utf-8") as f:
                _SECTOR_CACHE = json.load(f)
                return _SECTOR_CACHE
    _SECTOR_CACHE = _refresh_sector_map()
    return _SECTOR_CACHE


def _refresh_sector_map() -> dict:
    """从申万行业数据构建 {code_6digit: industry_name} 映射，并写入缓存。"""
    import json, time
    try:
        industries = ak.sw_index_first_info()
        codes_col  = industries.columns[0]   # 行业代码
        names_col  = industries.columns[1]   # 行业名称

        mapping: dict = {}
        for _, row in industries.iterrows():
            sw_code  = row[codes_col]
            ind_name = row[names_col]
            try:
                members  = ak.sw_index_third_cons(symbol=sw_code)
                code_col = members.columns[1]   # 股票代码，格式 600519.SH
                for raw_code in members[code_col]:
                    c6 = str(raw_code).split(".")[0].zfill(6)
                    mapping[c6] = ind_name
            except Exception:
                pass
            time.sleep(0.3)   # 避免并发导致服务端拒绝

        if mapping:
            with open(SECTOR_MAP_FILE, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False)
        return mapping
    except Exception as e:
        print(f"[fetcher] 行业映射刷新失败: {e}")
        return {}


def get_stock_sector(code: str) -> str:
    """返回单只股票的申万一级行业名称，未知时返回空字符串。"""
    return _load_sector_map().get(code.zfill(6), "")


def fetch_hot_sectors(top_n: int = 5) -> list[dict]:
    """
    返回今日 THS 行业排行前 top_n 名，格式：
    [{"name": "半导体", "pct": 2.80, "leader": "北方华创"}, ...]
    结果缓存 10 分钟。
    """
    import time
    global _HOT_SECTORS_CACHE
    ts, cached = _HOT_SECTORS_CACHE
    if time.time() - ts < 600 and cached:
        return cached[:top_n]
    try:
        df = ak.stock_board_industry_summary_ths()
        # 列顺序：排名,名称,涨跌幅,...,领涨股,...
        result = []
        for _, row in df.head(top_n).iterrows():
            vals = row.tolist()
            result.append({
                "name":   str(vals[1]),
                "pct":    float(vals[2]) if vals[2] else 0.0,
                "leader": str(vals[9]) if len(vals) > 9 else "",
            })
        _HOT_SECTORS_CACHE = (time.time(), result)
        return result
    except Exception:
        return []


def search_stocks(keyword: str) -> list:
    """
    按名称关键字搜索 A 股，返回 [(code, name), ...]。
    不区分大小写，支持简拼/关键词。
    """
    mapping = _load_name_map()
    kw = keyword.strip().upper()
    return [
        (code, name)
        for code, name in mapping.items()
        if kw in name.upper()
    ]


def _code_to_sina_symbol(code: str) -> str:
    """将纯数字代码转为新浪格式：600519 → sh600519，000858 → sz000858。"""
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def _fetch_from_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """使用新浪数据源拉取日线数据（东方财富源在当前环境不可用）。"""
    symbol = _code_to_sina_symbol(code)
    raw = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    raw["date"] = pd.to_datetime(raw["date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
    return df[[c for c in _COLS if c in df.columns]].reset_index(drop=True)
