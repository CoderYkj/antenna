"""
kline.py - 生成 K 线图并上传到飞书，返回 image_key
"""
import os
import tempfile

import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # 非交互模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import requests

# ── 设置中文字体（优先 Windows 系统字体）────────────────────
_CHINESE_FONT = "DejaVu Sans"   # 默认英文回退

def _set_chinese_font():
    global _CHINESE_FONT
    candidates = [
        ("C:/Windows/Fonts/msyh.ttc",   "Microsoft YaHei"),
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
        ("C:/Windows/Fonts/simsun.ttc", "SimSun"),
    ]
    for path, name in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            _CHINESE_FONT = name
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

_set_chinese_font()


def generate_kline(df: pd.DataFrame, code: str,
                   pred_high: float = None, pred_low: float = None,
                   days: int = 30) -> str:
    """
    生成近 N 日 K 线图（含均线、成交量、预测高低点标注），
    保存为临时 PNG 文件，返回文件路径。
    """
    recent = df.tail(days).copy()
    recent.index = pd.to_datetime(recent["date"])
    ohlcv = recent[["open", "high", "low", "close", "volume"]].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
    ohlcv = ohlcv.astype(float)

    # 均线叠加
    add_plots = [
        mpf.make_addplot(recent["ma5"].values,  color="#FF6B35", width=1.0, label="MA5"),
        mpf.make_addplot(recent["ma10"].values, color="#FFD700", width=1.0, label="MA10"),
        mpf.make_addplot(recent["ma20"].values, color="#00CED1", width=1.0, label="MA20"),
    ]

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mpf.make_marketcolors(
            up="red", down="green",
            edge="inherit", wick="inherit", volume="inherit",
        ),
        facecolor="#1a1a2e",
        figcolor="#1a1a2e",
        gridcolor="#333355",
        gridstyle="--",
        rc={"axes.labelcolor": "#cccccc", "xtick.color": "#cccccc",
            "ytick.color": "#cccccc", "font.family": _CHINESE_FONT,
            "axes.unicode_minus": False},
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()

    fig, axes = mpf.plot(
        ohlcv,
        type="candle",
        style=style,
        volume=True,
        addplot=add_plots,
        title=f"  {code}  近{days}日",
        figsize=(10, 6),
        returnfig=True,
        tight_layout=True,
    )

    # 标注预测高低点
    ax = axes[0]
    if pred_high:
        ax.axhline(pred_high, color="#FF4444", linestyle="--", linewidth=1.2,
                   label=f"预测高点 {pred_high:.2f}")
    if pred_low:
        ax.axhline(pred_low,  color="#44FF88", linestyle="--", linewidth=1.2,
                   label=f"预测低点 {pred_low:.2f}")
    if pred_high or pred_low:
        ax.legend(loc="upper left", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="#cccccc")

    fig.savefig(tmp.name, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)
    return tmp.name


def generate_intraday_kline(df_intraday: pd.DataFrame, code: str,
                            name: str = "",
                            cur_price: float = None,
                            prev_close: float = None,
                            pred_high: float = None,
                            pred_low: float = None) -> str:
    """
    生成当日分时 K 线图（5分钟棒），含成交量、昨收/预测高低线。
    保存为临时 PNG，返回路径。df_intraday 来自 fetch_intraday_kline()。
    """
    if df_intraday is None or df_intraday.empty:
        return ""

    df = df_intraday.copy()
    df.index = pd.to_datetime(df["time"])
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
    ohlcv = ohlcv.astype(float)

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mpf.make_marketcolors(
            up="red", down="green",
            edge="inherit", wick="inherit", volume="inherit",
        ),
        facecolor="#1a1a2e",
        figcolor="#1a1a2e",
        gridcolor="#333355",
        gridstyle="--",
        rc={"axes.labelcolor": "#cccccc", "xtick.color": "#cccccc",
            "ytick.color": "#cccccc", "font.family": _CHINESE_FONT,
            "axes.unicode_minus": False},
    )

    date_str = df.index[-1].strftime("%Y-%m-%d") if len(df) else ""
    title_name = f"{name}（{code}）" if name else code
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()

    fig, axes = mpf.plot(
        ohlcv,
        type="candle",
        style=style,
        volume=True,
        title=f"  {title_name}  {date_str}  5min",
        figsize=(10, 6),
        returnfig=True,
        tight_layout=True,
        xrotation=15,
        datetime_format="%H:%M",
    )

    ax = axes[0]
    legend_items = []
    if prev_close:
        ax.axhline(prev_close, color="#AAAAAA", linestyle=":", linewidth=1.0,
                   label=f"昨收 {prev_close:.2f}")
        legend_items.append(f"昨收 {prev_close:.2f}")
    if cur_price:
        ax.axhline(cur_price, color="#FFFFFF", linestyle="-", linewidth=1.0, alpha=0.8,
                   label=f"现价 {cur_price:.2f}")
        legend_items.append(f"现价 {cur_price:.2f}")
    if pred_high:
        ax.axhline(pred_high, color="#FF4444", linestyle="--", linewidth=1.2,
                   label=f"预测高 {pred_high:.2f}")
        legend_items.append(f"预测高 {pred_high:.2f}")
    if pred_low:
        ax.axhline(pred_low, color="#44FF88", linestyle="--", linewidth=1.2,
                   label=f"预测低 {pred_low:.2f}")
        legend_items.append(f"预测低 {pred_low:.2f}")
    if legend_items:
        ax.legend(loc="upper left", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="#cccccc", framealpha=0.8)

    fig.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return tmp.name

def upload_image_to_feishu(token: str, img_path: str) -> str:
    """
    将本地图片上传到飞书，返回 image_key（失败返回空串）。
    """
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    try:
        if not os.path.exists(img_path):
            print(f"[kline] 图片文件不存在: {img_path}")
            return ""
        with open(img_path, "rb") as f:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                data={"image_type": "message"},
                files={"image": f},
                timeout=30,
            ).json()
        image_key = (resp.get("data") or {}).get("image_key", "")
        if not image_key:
            print(f"[kline] 图片上传 API 错误: code={resp.get('code')} msg={resp.get('msg')} data={resp.get('data')}")
        return image_key
    except Exception as e:
        print(f"[kline] 图片上传异常: {e}")
        return ""
    finally:
        try:
            os.unlink(img_path)
        except Exception:
            pass
