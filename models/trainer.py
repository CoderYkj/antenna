import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── LightGBM 参数(集中管理,原 train/train_weighted 内联常量) ──
_LGB_PARAMS = {
    "objective":     "binary",
    "metric":        "auc",
    "learning_rate": 0.05,
    "num_leaves":    31,
    "verbose":       -1,
}
_NUM_BOOST_ROUND   = 200
_EARLY_STOP_ROUNDS = 30
_LOG_EVERY_ROUNDS  = 50


def build_labels(df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """计算未来 N 天涨幅是否超过阈值（1=涨，0=不涨）。"""
    future_return = df["close"].shift(-target_days) / df["close"] - 1
    return (future_return > threshold).astype(int)


def _train_impl(
    df: pd.DataFrame,
    feature_cols: list,
    weight_col: str | None = None,
) -> lgb.Booster:
    """train / train_weighted 的共享实现。

    weight_col=None:普通训练;否则从 df 取 weight 列透传给 lgb.Dataset(weight=)。
    划分/早停/AUC 报告统一。
    """
    required = feature_cols + ["label"]
    if weight_col is not None:
        required = required + [weight_col]
    df = df.dropna(subset=required)

    split_date = df["date"].max() - pd.DateOffset(months=3)
    train_df = df[df["date"] <= split_date]
    test_df  = df[df["date"] >  split_date]

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test,  y_test  = test_df[feature_cols],  test_df["label"]

    if weight_col is not None:
        w_train = train_df[weight_col].astype(float)
        w_test  = test_df[weight_col].astype(float)
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        valid_data = lgb.Dataset(X_test,  label=y_test,  weight=w_test, reference=train_data)
        auc_label  = "Test AUC (weighted)"
    else:
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test,  label=y_test,  reference=train_data)
        auc_label  = "Test AUC"

    model = lgb.train(
        _LGB_PARAMS,
        train_data,
        num_boost_round=_NUM_BOOST_ROUND,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(_EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(_LOG_EVERY_ROUNDS),
        ],
    )

    if not test_df.empty and y_test.nunique() > 1:
        auc = roc_auc_score(y_test, model.predict(X_test))
        print(f"  {auc_label}: {auc:.4f}")

    return model


def train(df: pd.DataFrame, feature_cols: list) -> lgb.Booster:
    """训练 LightGBM 二分类模型。df 须已含 'label' 列(由调用方按股票单独计算)。"""
    return _train_impl(df, feature_cols, weight_col=None)


def train_weighted(
    df: pd.DataFrame,
    feature_cols: list,
    weight_col: str = "sample_weight",
) -> lgb.Booster:
    """带 sample_weight 的 LightGBM 训练(P1 §4.4)。

    df 须含 'label' 列与 weight_col 列(默认 'sample_weight')。
    其他逻辑(walk-forward 划分 / 早停 / AUC 报告)与 train 一致。

    sample_weight 由调用方(model_learner.retrain_with_weights)按 spec §4.1 表
    根据历史 (signal, hit_tier) 查表预填,本函数不做权重计算,只透传给 LightGBM。
    """
    return _train_impl(df, feature_cols, weight_col=weight_col)


def save_model(model: lgb.Booster, saved_dir: str = "models/saved") -> str:
    Path(saved_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = Path(saved_dir) / f"model_{date_str}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")
    return str(path)


def load_latest_model(saved_dir: str = "models/saved") -> lgb.Booster:
    files = sorted(Path(saved_dir).glob("model_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No model found in {saved_dir}. Run: python cli.py train")
    with open(files[-1], "rb") as f:
        return pickle.load(f)
