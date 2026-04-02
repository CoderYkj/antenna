import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_labels(df: pd.DataFrame, target_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """计算未来 N 天涨幅是否超过阈值（1=涨，0=不涨）。"""
    future_return = df["close"].shift(-target_days) / df["close"] - 1
    return (future_return > threshold).astype(int)


def train(
    df: pd.DataFrame,
    feature_cols: list,
) -> lgb.Booster:
    """训练 LightGBM 二分类模型。df 须已含 'label' 列（由调用方按股票单独计算）。"""
    df = df.dropna(subset=feature_cols + ["label"])

    split_date = df["date"].max() - pd.DateOffset(months=3)
    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    if not test_df.empty and y_test.nunique() > 1:
        auc = roc_auc_score(y_test, model.predict(X_test))
        print(f"  Test AUC: {auc:.4f}")

    return model


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
