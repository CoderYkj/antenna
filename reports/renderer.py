from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_report(
    code: str,
    df: pd.DataFrame,
    prediction: dict,
    output_dir: str = "reports/output",
) -> str:
    """生成包含 K线 + 指标 + 预测结果的单文件 HTML 报告。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.18, 0.16, 0.16],
        subplot_titles=["K线 + 均线", "成交量", "MACD", "RSI6"],
        vertical_spacing=0.04,
    )

    # K线图
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线",
        increasing_line_color="red", decreasing_line_color="green",
    ), row=1, col=1)

    # 均线
    ma_colors = {5: "#1f77b4", 10: "#ff7f0e", 20: "#2ca02c", 60: "#d62728"}
    for period, color in ma_colors.items():
        col_name = f"ma{period}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col_name], name=f"MA{period}",
                line=dict(color=color, width=1),
            ), row=1, col=1)

    # 成交量
    vol_colors = [
        "red" if c >= o else "green"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="成交量",
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # MACD
    if "macd_dif" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_dif"], name="DIF",
            line=dict(color="#1f77b4", width=1),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_dea"], name="DEA",
            line=dict(color="#ff7f0e", width=1),
        ), row=3, col=1)
        hist_colors = ["red" if v >= 0 else "green" for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["macd_hist"], name="HIST",
            marker_color=hist_colors, showlegend=False,
        ), row=3, col=1)

    # RSI
    if "rsi6" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi6"], name="RSI6",
            line=dict(color="#9467bd", width=1),
        ), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    # 标题含预测结果
    signal_emoji = {"买入": "▲", "观望": "●", "回避": "▼"}
    emoji = signal_emoji.get(prediction["signal"], "")
    title = (
        f"{code}  {emoji} {prediction['signal']}  "
        f"涨概率: {prediction['rise_prob']:.1%}  "
        f"置信度: {prediction['confidence']}"
    )

    fig.update_layout(
        title=title,
        height=900,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(type="category")

    output_path = str(Path(output_dir) / f"{code}_report.html")
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
