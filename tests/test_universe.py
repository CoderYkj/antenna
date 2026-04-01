import pytest


def test_load_universe_returns_watchlist():
    from data.universe import load_universe
    config = {
        "universe": {
            "watchlist": ["600519", "000858"],
            "scan_pool": "watchlist",
        }
    }
    result = load_universe(config)
    assert result == ["600519", "000858"]


def test_load_universe_unknown_pool_raises():
    from data.universe import load_universe
    config = {
        "universe": {
            "watchlist": ["600519"],
            "scan_pool": "unknown_pool",
        }
    }
    with pytest.raises(ValueError, match="Unknown scan_pool"):
        load_universe(config)
