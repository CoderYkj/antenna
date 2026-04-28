import pytest

from learning.scene_bucket import bucket_by_scene, bucket_by_state, merge_buckets


SAMPLE_PREDS = [
    {"code": "600519", "scene": "scan",         "market_state": "bull",  "rise_prob": 0.75},
    {"code": "000001", "scene": "scan",         "market_state": "bull",  "rise_prob": 0.55},
    {"code": "300750", "scene": "tactic:value", "market_state": "bull",  "rise_prob": 0.80},
    {"code": "002174", "scene": "tactic:growth","market_state": "range", "rise_prob": 0.62},
    {"code": "600036", "scene": "predict",      "market_state": "range", "rise_prob": 0.68},
    {"code": "000002",                           "market_state": "bear",  "rise_prob": 0.45},  # 无 scene
]


def test_bucket_by_scene_returns_expected_keys():
    buckets = bucket_by_scene(SAMPLE_PREDS)
    assert set(buckets.keys()) == {"scan", "tactic:value", "tactic:growth", "predict"}
    # 无 scene 的记录归到 "scan"
    codes_in_scan = [r["code"] for r in buckets["scan"]]
    assert "600519" in codes_in_scan
    assert "000001" in codes_in_scan
    assert "000002" in codes_in_scan  # 旧数据


def test_bucket_by_state_splits_by_market_state():
    buckets = bucket_by_state(SAMPLE_PREDS)
    assert set(buckets.keys()) == {"bull", "range", "bear"}
    assert len(buckets["bull"]) == 3
    assert len(buckets["range"]) == 2
    assert len(buckets["bear"]) == 1


def test_bucket_by_state_missing_defaults_to_range():
    preds_missing_state = [{"code": "X", "scene": "scan", "rise_prob": 0.5}]
    buckets = bucket_by_state(preds_missing_state)
    assert "range" in buckets
    assert len(buckets["range"]) == 1


def test_merge_buckets_by_scene_then_state():
    buckets = merge_buckets(SAMPLE_PREDS, by=("scene", "market_state"))
    # ("scan","bull") 应有 2 条 (600519, 000001)
    assert len(buckets[("scan", "bull")]) == 2
    assert len(buckets[("tactic:value", "bull")]) == 1


def test_bucket_by_scene_empty_input():
    assert bucket_by_scene([]) == {}


def test_bucket_by_state_empty_input():
    assert bucket_by_state([]) == {}


def test_merge_buckets_empty_input():
    assert merge_buckets([], by=("scene", "market_state")) == {}


def test_merge_buckets_unknown_dimension_raises():
    with pytest.raises(ValueError, match="unknown dimension"):
        merge_buckets(SAMPLE_PREDS, by=("scene", "bad_key"))


def test_merge_buckets_record_missing_both_fields_uses_defaults():
    """记录同时缺 scene 和 market_state 时应归入 (scan, range)。"""
    records = [{"code": "X", "rise_prob": 0.5}]
    buckets = merge_buckets(records, by=("scene", "market_state"))
    assert buckets[("scan", "range")] == records
