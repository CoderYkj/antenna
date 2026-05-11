import json
from pathlib import Path
import pytest
from learning import feedback_io


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_io, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(feedback_io, "HISTORY_DIR",  tmp_path / "history")
    return tmp_path


def test_write_feedback_creates_file_with_expected_schema(workdir):
    feedback_io.write_feedback("market_state", "2026-04-27", {
        "samples_used":    100,
        "accuracy_before": 0.32,
        "accuracy_after":  0.40,
        "detail":          "bull → range",
    })
    path = workdir / "feedback" / "market_state_2026-04-27.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["module"] == "market_state"
    assert data["date"] == "2026-04-27"
    assert data["samples_used"] == 100


def test_snapshot_rotates_to_max_7(workdir):
    target = workdir / "market_state.json"
    target.write_text('{"v":1}', encoding="utf-8")

    for i in range(10):
        date_str = f"2026-04-{str(10 + i).zfill(2)}"
        feedback_io.snapshot_file(target, date_str)

    snapshots = sorted((workdir / "history").glob("market_state_*.json"))
    assert len(snapshots) == 7
    # 最早 3 份应被删除,保留最近 7 份
    names = [p.name for p in snapshots]
    assert "market_state_2026-04-13.json" in names
    assert "market_state_2026-04-19.json" in names
    assert "market_state_2026-04-10.json" not in names


def test_snapshot_missing_source_is_no_op(workdir):
    """源文件不存在时不报错,只是不快照。"""
    missing = workdir / "nonexistent.json"
    feedback_io.snapshot_file(missing, "2026-04-27")  # 不应抛异常
    assert not (workdir / "history").exists() or \
        len(list((workdir / "history").glob("*.json"))) == 0


def test_atomic_write_does_not_leave_tmp(workdir):
    target = workdir / "subdir" / "out.json"
    feedback_io.atomic_write_json(target, {"k": "v"})
    assert target.exists()
    # 确保没有遗留的 .tmp
    assert not list(target.parent.glob("*.tmp"))


def test_envelope_keys_take_precedence_over_payload(workdir):
    """payload 中若包含同名 envelope key,envelope 应优先(module/date/written_at)。"""
    feedback_io.write_feedback("real_module", "2026-04-28", {
        "module":     "EVIL",          # 不应覆盖
        "date":       "1999-01-01",    # 不应覆盖
        "written_at": "forged",         # 不应覆盖
        "samples":    42,
    })
    path = workdir / "feedback" / "real_module_2026-04-28.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["module"] == "real_module"
    assert data["date"] == "2026-04-28"
    assert data["written_at"] != "forged"
    assert data["samples"] == 42


def test_atomic_write_on_tmp_suffix_path(workdir):
    """即使目标路径自身以 .tmp 结尾,也不会发生 temp/target 冲突。"""
    target = workdir / "weird.tmp"
    feedback_io.atomic_write_json(target, {"k": "v"})
    assert target.exists()
    # temp 命名应是 weird.tmp.tmp,不会与 target 同名
    assert not (workdir / "weird.tmp.tmp").exists()


# ── Windows PermissionError 重试机制 ──────────────────

def test_replace_retries_on_permission_error(workdir, monkeypatch):
    """Windows 偶发 PermissionError 时重试 + 退避,最终成功不抛。"""
    import os
    calls = {"count": 0}
    real_replace = os.replace

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError(13, "fake windows lock")
        return real_replace(src, dst)

    monkeypatch.setattr(feedback_io.os, "replace", flaky_replace)
    # 把 sleep 置空加速测试
    monkeypatch.setattr(feedback_io.time, "sleep", lambda _: None)

    target = workdir / "flaky.json"
    feedback_io.atomic_write_json(target, {"ok": True})
    assert target.exists()
    assert calls["count"] == 3  # 前 2 次失败,第 3 次成功


def test_replace_gives_up_after_max_retries(workdir, monkeypatch):
    """重试耗尽仍失败 → 抛出原 PermissionError,且清理 .tmp 文件。"""
    def always_fail(src, dst):
        raise PermissionError(13, "persistent lock")

    monkeypatch.setattr(feedback_io.os, "replace", always_fail)
    monkeypatch.setattr(feedback_io.time, "sleep", lambda _: None)

    target = workdir / "stuck.json"
    with pytest.raises(PermissionError):
        feedback_io.atomic_write_json(target, {"x": 1})
    # .tmp 应已清理
    assert not (workdir / "stuck.json.tmp").exists()
