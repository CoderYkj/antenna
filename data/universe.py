def load_universe(config: dict) -> list:
    """根据 config 返回股票代码列表。"""
    pool = config["universe"].get("scan_pool", "watchlist")
    if pool == "watchlist":
        return config["universe"]["watchlist"]
    raise ValueError(f"Unknown scan_pool: {pool}. Supported: watchlist")
