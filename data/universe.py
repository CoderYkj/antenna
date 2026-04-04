import akshare as ak


def load_universe(config: dict) -> list:
    """根据 config 返回股票代码列表。"""
    pool = config["universe"].get("scan_pool", "watchlist")
    if pool == "watchlist":
        return config["universe"]["watchlist"]
    if pool == "all":
        return get_all_codes()
    raise ValueError(f"Unknown scan_pool: {pool}. Supported: watchlist / all")


def get_all_codes() -> list:
    """返回全 A 股股票代码列表（通过新浪接口，约 5500 只）。"""
    df = ak.stock_info_a_code_name()
    return df["code"].tolist()
