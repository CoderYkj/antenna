"""
orchestrator.py - 瘦编排器:按依赖顺序跑所有学习子模块,单模块失败不影响其他。

P0 仅挂载 market_state,后续阶段会追加:
  - P1: model_learner, blacklist
  - P2: tactic_learner, ai_reason
  - P3: alt_data, feature_learner
  - P4: price_learner

每个 MODULES 条目:
  {"name": str, "run": callable(date_str) → dict, "depends_on": list[str]}
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from learning import feedback_io, alerts, market_state, model_learner, tactic_learner, blacklist


# ── 模块注册表(按依赖顺序) ─────────────────────────────
MODULES: list[dict] = [
    {
        "name":       "market_state",
        "run":        market_state.run,
        "depends_on": [],
    },
    {
        "name":       "model_learner",
        "run":        model_learner.run,
        "depends_on": ["market_state"],
    },
    {
        "name":       "tactic_learner",
        "run":        tactic_learner.run,
        "depends_on": ["market_state"],
    },
    {
        "name":       "blacklist",
        "run":        blacklist.run,
        "depends_on": ["market_state"],
    },
    # P3/P4 后续阶段追加
]


def run_all(date_str: str | None = None, dry_run: bool = False) -> dict:
    """
    依次执行 MODULES 中每个模块。某模块失败:
      - 记录告警
      - 不执行依赖它的模块(标记 skipped)
      - 继续执行无依赖关系的其他模块
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    results: dict[str, dict] = {}
    failed: set[str] = set()

    for module in MODULES:
        name = module["name"]
        deps = module.get("depends_on") or []

        if any(d in failed for d in deps):
            results[name] = {
                "status":  "skipped",
                "reason":  f"dependency failed: {[d for d in deps if d in failed]}",
            }
            continue

        if dry_run:
            results[name] = {"status": "dry_run"}
            continue

        try:
            result = module["run"](date_str)
            results[name] = {"status": "ok", "result": result}
            feedback_io.write_feedback(name, date_str, {"result": result})
        except Exception as e:
            failed.add(name)
            results[name] = {"status": "failed", "error": str(e)}
            alerts.send_alert(name, f"学习失败: {e}", traceback=True)

    return results


def check() -> int:
    """
    启动自检:验证所有学习产物 JSON 可解析。
    返回 exit code(0 成功;非 0 失败)。
    """
    files_to_check = [
        Path("learning/market_state.json"),
        Path("learning/strategy.json"),
        Path("learning/model_learner.json"),  # P1 产物
        Path("learning/tactic_params.json"),  # P2 产物
        Path("learning/blacklist.json"),      # 横向黑名单产物
        # P3-P4 阶段会追加 feature_weights.json / price_params.json
    ]
    errors = []
    for p in files_to_check:
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"{p}: {e}")

    if errors:
        for err in errors:
            print(f"[check] {err}", file=sys.stderr)
        return 1
    print("[check] 所有学习产物验证通过")
    return 0
