"""
feedback_io.py - 学习模块的统一 I/O 工具。

- write_feedback(module, date, data)          → learning/feedback/<module>_<date>.json
- snapshot_file(source, date)                 → learning/history/<basename>_<date>.json(最多保留 7 份)
- atomic_write_json(path, data)               → 临时文件 + os.replace 原子写

注意:write_feedback 中 envelope 字段(module/date/written_at)优先于 payload 里的同名键,
不会被 payload 覆盖。
"""
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

FEEDBACK_DIR = Path("learning/feedback")
HISTORY_DIR = Path("learning/history")
MAX_SNAPSHOTS = 7

# Windows 上 os.replace 在目标文件被其他进程打开读取时会抛 PermissionError(13)。
# 典型场景:antenna-bot 在读 market_state.json,同时"学习"指令正要 os.replace 新版本。
# Linux 下 os.replace 是原子的无此问题。
# 此处的 retry 只对 Windows 场景救命;Linux 第一次就会成功。
_REPLACE_RETRY_TIMES = 5
_REPLACE_RETRY_BASE  = 0.05   # 初始 50ms,指数退避 × 2


def _os_replace_with_retry(tmp: Path, path: Path) -> None:
    """Windows 文件锁友好的 os.replace 封装,PermissionError 重试。"""
    last_err: Exception | None = None
    for attempt in range(_REPLACE_RETRY_TIMES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            if attempt == _REPLACE_RETRY_TIMES - 1:
                break
            time.sleep(_REPLACE_RETRY_BASE * (2 ** attempt))
    raise last_err if last_err else RuntimeError("replace failed without error")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 用 with_name 而非 with_suffix,避免 path 本身以 .tmp 结尾时 temp 与 target 同名
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _os_replace_with_retry(tmp, path)
    except BaseException:
        # 覆盖 KeyboardInterrupt/SystemExit 等非 Exception 子类,保证不留 .tmp
        tmp.unlink(missing_ok=True)
        raise


def write_feedback(module: str, date_str: str, payload: dict) -> Path:
    """
    写入一条 feedback 日志。
    payload 典型字段:samples_used, accuracy_before, accuracy_after, detail。

    envelope 字段 `module` / `date` / `written_at` 由本函数填充,优先级高于 payload;
    即使 payload 中包含同名 key 也不会覆盖 envelope。
    """
    path = FEEDBACK_DIR / f"{module}_{date_str}.json"
    data = {
        **payload,
        "module":     module,
        "date":       date_str,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(path, data)
    return path


def snapshot_file(source: Path, date_str: str) -> Path | None:
    """
    将 source 拷贝到 history/,命名为 <basename>_<date_str>.json。
    超过 MAX_SNAPSHOTS 时删除最早的快照(单次切片删除,避免 glob 循环中的竞态)。
    source 不存在时静默返回 None。
    """
    source = Path(source)
    if not source.exists():
        return None

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    basename = source.stem
    dest = HISTORY_DIR / f"{basename}_{date_str}{source.suffix}"
    shutil.copy2(source, dest)

    snapshots = sorted(HISTORY_DIR.glob(f"{basename}_*{source.suffix}"))
    for old in snapshots[:-MAX_SNAPSHOTS]:
        old.unlink(missing_ok=True)
    return dest
