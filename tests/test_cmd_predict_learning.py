"""tests/test_cmd_predict_learning.py — 预测/行情指令学习上下文单元测试。"""
from unittest.mock import patch, MagicMock
import json


def test_learning_context_line_all_available():
    """_learning_context_line 应聚合市场状态、校准门槛、活跃特征数、ATR系数。"""
    def _rt(self, **kw):
        name = str(self)
        if "model_learner" in name:
            return json.dumps({"abs_threshold": 0.38, "current_state": "bull"})
        if "feature_weights" in name:
            return json.dumps({"active": list(range(22))})
        if "price_params" in name:
            return json.dumps({"bull": {"short_atr_mult": 1.3, "short_gain_mult": 1.8}})
        raise FileNotFoundError(name)

    with patch("learning.market_state.load_current_state", return_value={"current": "bull"}), \
         patch("pathlib.Path.read_text", _rt):
        from server.predict_cmd import _learning_context_line
        line = _learning_context_line()

    assert "bull" in line
    assert "0.38" in line
    assert "22" in line
    assert "1.3" in line
    assert "学习参数" in line


def test_learning_context_line_degrades_gracefully():
    """所有学习文件缺失时返回空串。"""
    with patch("learning.market_state.load_current_state", side_effect=Exception("no state")), \
         patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        from server.predict_cmd import _learning_context_line
        line = _learning_context_line()

    assert line == ""


def test_cmd_quote_shows_blacklist_warning():
    """cmd_quote 当代码在黑名单时应显示警告。"""
    mock_bl = MagicMock()
    mock_bl.is_blocked.return_value = True

    with patch("learning.blacklist.load_blacklist", return_value=mock_bl), \
         patch("learning.market_state.load_current_state", return_value={"current": "bear"}), \
         patch("data.fetcher.fetch_realtime_prices", return_value={"000001": {
             "price": 10.0, "pct": -1.0, "high": 10.5, "low": 9.8, "open": 10.2, "name": "平安银行"
         }}), \
         patch("data.fetcher.fetch_intraday_kline", return_value=None):
        from server.predict_cmd import cmd_quote
        result = cmd_quote("000001")

    card = json.loads(result["content"])
    all_text = " ".join(e.get("content", "") for e in card["elements"] if e.get("tag") == "markdown")
    assert "黑名单" in all_text
    assert "bear" in all_text


def test_cmd_quote_no_blacklist_warning_when_clean():
    """cmd_quote 不在黑名单时不显示警告。"""
    mock_bl = MagicMock()
    mock_bl.is_blocked.return_value = False

    with patch("learning.blacklist.load_blacklist", return_value=mock_bl), \
         patch("learning.market_state.load_current_state", return_value={"current": "range"}), \
         patch("data.fetcher.fetch_realtime_prices", return_value={"000001": {
             "price": 10.0, "pct": 0.5, "high": 10.5, "low": 9.8, "open": 10.0, "name": "平安银行"
         }}), \
         patch("data.fetcher.fetch_intraday_kline", return_value=None):
        from server.predict_cmd import cmd_quote
        result = cmd_quote("000001")

    card = json.loads(result["content"])
    all_text = " ".join(e.get("content", "") for e in card["elements"] if e.get("tag") == "markdown")
    assert "黑名单" not in all_text
