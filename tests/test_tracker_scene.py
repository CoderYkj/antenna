import json
from pathlib import Path
import pytest
from learning import tracker


@pytest.fixture
def fixture_date(tmp_path, monkeypatch):
    """复制 fixture 到临时 DATA_DIR。"""
    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    src = Path(__file__).parent / "fixtures" / "sample_preds.jsonl"
    (tmp_path / "pred_2026-04-01.jsonl").write_bytes(src.read_bytes())
    return "2026-04-01"


def test_load_predictions_by_scene_returns_only_matching(fixture_date):
    result = tracker.load_predictions_by_scene(fixture_date, "scan")
    # 2 explicit scan records + 1 no-scene backward-compat record (600036) = 3
    assert len(result) == 3
    assert all(r["scene"] == "scan" for r in result)


def test_load_predictions_by_scene_tactic_matches_prefix(fixture_date):
    result = tracker.load_predictions_by_scene(fixture_date, "tactic:value")
    assert len(result) == 1
    assert result[0]["code"] == "600519"


def test_load_predictions_by_scene_all_tactic_substrings(fixture_date):
    result = tracker.load_predictions_by_scene(fixture_date, "tactic")
    assert len(result) == 1


def test_load_predictions_by_scene_backward_compat_records(fixture_date):
    """无 scene 字段的旧记录默认归到 'scan' 桶。"""
    result = tracker.load_predictions_by_scene(fixture_date, "scan")
    codes = [r["code"] for r in result]
    assert "600036" in codes  # 最后一条没 scene,应视作 scan


def test_log_predictions_preserves_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    records = [
        {"code": "600519", "name": "贵州茅台", "signal": "买入", "rise_prob": 0.7, "scene": "tactic:growth"},
    ]
    tracker.log_predictions("2026-04-02", records)
    loaded = tracker.load_predictions("2026-04-02")
    assert loaded[0]["scene"] == "tactic:growth"


def test_log_predictions_same_code_different_scene_coexist(tmp_path, monkeypatch):
    """同一天同一 code 不同 scene 应各存一条,不去重。"""
    monkeypatch.setattr(tracker, "DATA_DIR", tmp_path)
    tracker.log_predictions("2026-04-03", [
        {"code": "600519", "scene": "scan", "signal": "买入", "rise_prob": 0.7},
    ])
    tracker.log_predictions("2026-04-03", [
        {"code": "600519", "scene": "tactic:value", "signal": "买入", "rise_prob": 0.72},
    ])
    loaded = tracker.load_predictions("2026-04-03")
    scenes = sorted(r["scene"] for r in loaded)
    assert scenes == ["scan", "tactic:value"]
