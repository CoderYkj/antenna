import pandas as pd
import numpy as np
from pathlib import Path


def _make_chart_df(n=120):
    np.random.seed(3)
    from features.technical import add_indicators
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        "amount": np.random.uniform(5e8, 5e9, n),
        "turnover": np.random.uniform(0.1, 2.0, n),
    })
    return add_indicators(df)


_PREDICTION = {
    "rise_prob": 0.72,
    "fall_prob": 0.28,
    "confidence": "高",
    "signal": "买入",
}


def test_render_creates_html_file(tmp_path):
    from reports.renderer import render_report
    df = _make_chart_df()
    path = render_report("600519", df, _PREDICTION, output_dir=str(tmp_path))
    assert Path(path).exists()
    assert path.endswith(".html")


def test_render_html_contains_code(tmp_path):
    from reports.renderer import render_report
    df = _make_chart_df()
    path = render_report("600519", df, _PREDICTION, output_dir=str(tmp_path))
    content = Path(path).read_text(encoding="utf-8")
    assert "600519" in content


def test_render_html_contains_signal(tmp_path):
    from reports.renderer import render_report
    df = _make_chart_df()
    path = render_report("600519", df, _PREDICTION, output_dir=str(tmp_path))
    content = Path(path).read_text(encoding="utf-8")
    # Plotly 可能将中文转义为 \uXXXX，兼容两种格式
    assert "买入" in content or "\\u4e70\\u5165" in content
