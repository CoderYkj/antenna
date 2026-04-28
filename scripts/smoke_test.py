"""
smoke_test.py - 功能冒烟测试，每次代码改动后运行，验证核心指令正常响应。

用法：
    python scripts/smoke_test.py
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = []
FAIL = []


def check(name: str, fn, is_command: bool = False):
    try:
        result = fn()
        assert result is not None, f"返回 None"
        # 只对飞书指令返回值校验 msg_type
        if is_command and isinstance(result, dict):
            assert "msg_type" in result, f"缺少 msg_type: {result}"
        PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  [FAIL] {name}: {e}")


def run():
    print("=" * 50)
    print("Antenna Smoke Test")
    print("=" * 50)

    from server.commands import handle_command

    # ── 模块导入 ──────────────────────────────────────────
    print("\n[1] Module imports")

    def _import(mod):
        return lambda: (__import__(mod), True)[1]

    for mod in [
        "server.predict_cmd",
        "server.commands",
        "server.feishu_poll",
        "server.feishu_push",
        "server.watchlist",
        "features.technical",
        "features.builder",
        "features.analyser",
        "models.predictor",
        "learning.optimizer",
        "learning.tracker",
        "notify.feishu",
        "reports.kline",
    ]:
        check(f"import {mod}", _import(mod))

    # ── 核心指令 ──────────────────────────────────────────
    print("\n[2] Commands")

    check("自选",    lambda: handle_command("自选"),    is_command=True)
    check("策略",    lambda: handle_command("策略"),    is_command=True)
    check("自选表现", lambda: handle_command("自选表现"), is_command=True)
    check("帮助",    lambda: handle_command("帮助"),    is_command=True)

    # 预测：用缓存数据，避免实时网络请求（可能失败）
    def _predict():
        r = handle_command("预测 600623")
        assert isinstance(r, dict) or isinstance(r, str)
        return r or True  # 无数据时返回字符串也算通过

    check("预测 600623", _predict, is_command=True)

    # 行情
    check("行情 平安", lambda: handle_command("行情 平安"), is_command=True)

    # ── 关键模块函数 ──────────────────────────────────────
    print("\n[3] Key functions")

    def _load_strategy():
        from learning.optimizer import load_strategy
        s = load_strategy()
        assert "buy_top_pct" in s
        return s

    check("load_strategy", _load_strategy)

    def _load_predictions():
        from learning.tracker import list_prediction_dates
        dates = list_prediction_dates()
        return True  # 有没有记录都算通过

    check("list_prediction_dates", _load_predictions)

    def _kline_import():
        from reports.kline import generate_intraday_kline, upload_image_to_feishu
        from features.analyser import text_intraday_kline
        return True

    check("kline functions exist", _kline_import)

    # ── 汇总 ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Result: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print(f"FAILED: {', '.join(FAIL)}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    run()
