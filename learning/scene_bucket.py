"""
scene_bucket.py - 按 scene/market_state 分桶工具,给所有 learner 复用。

scene 字段缺失的旧记录视作 "scan"。
market_state 字段缺失的旧记录视作 "range"。
"""
from collections import defaultdict
from typing import Iterable

DEFAULT_SCENE = "scan"
DEFAULT_STATE = "range"


def _scene(r: dict) -> str:
    return r.get("scene") or DEFAULT_SCENE


def _state(r: dict) -> str:
    return r.get("market_state") or DEFAULT_STATE


def bucket_by_scene(records: Iterable[dict]) -> dict[str, list[dict]]:
    """按 scene 分桶。子战法各自成桶(tactic:value / tactic:growth / ...)。"""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[_scene(r)].append(r)
    return dict(out)


def bucket_by_state(records: Iterable[dict]) -> dict[str, list[dict]]:
    """按 market_state 分桶。"""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[_state(r)].append(r)
    return dict(out)


def merge_buckets(records: Iterable[dict], by: tuple[str, ...]) -> dict[tuple, list[dict]]:
    """
    多维组合分桶。by=("scene", "market_state") 返回 {(scene, state): [...]}

    Raises:
        ValueError: 当 by 中包含未知维度时(有效维度:scene、market_state)。
    """
    extractors = {
        "scene": _scene,
        "market_state": _state,
    }
    unknown = set(by) - extractors.keys()
    if unknown:
        raise ValueError(
            f"merge_buckets: unknown dimension(s) {sorted(unknown)}. "
            f"Valid: {sorted(extractors)}"
        )
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = tuple(extractors[k](r) for k in by)
        out[key].append(r)
    return dict(out)
