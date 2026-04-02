from models.trainer import load_latest_model


def predict(df, feature_cols: list, model_dir: str = "models/saved") -> dict:
    """
    对给定股票的最新数据行进行预测。

    Returns:
        dict with keys: rise_prob, fall_prob, confidence, signal
    """
    model = load_latest_model(model_dir)
    valid = df[feature_cols].dropna()
    if valid.empty:
        raise ValueError("No valid rows after dropping NaN — need more history data.")

    prob = float(model.predict(valid.iloc[[-1]])[0])

    if prob >= 0.65:
        confidence = "高"
    elif prob >= 0.5:
        confidence = "中"
    else:
        confidence = "低"

    if prob >= 0.6:
        signal = "买入"
    elif prob >= 0.4:
        signal = "观望"
    else:
        signal = "回避"

    return {
        "rise_prob": round(prob, 4),
        "fall_prob": round(1 - prob, 4),
        "confidence": confidence,
        "signal": signal,
    }
