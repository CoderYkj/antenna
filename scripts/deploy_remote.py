"""
deploy_remote.py - 一键远端同步 + 部署 + 健康检查入口。

默认面向当前生产机：
  host=115.29.240.130, port=22, user=root, remote_dir=/data/antenna

用法：
  set ANTENNA_REMOTE_PASSWORD=***
  python scripts/deploy_remote.py
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import posixpath
import tempfile
import tarfile
from pathlib import Path


DEFAULT_HOST = "115.29.240.130"
DEFAULT_PORT = 22
DEFAULT_USER = "root"
DEFAULT_REMOTE_DIR = "/data/antenna"
DEFAULT_PASSWORD_ENV = "ANTENNA_REMOTE_PASSWORD"
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "logs",
}
EXCLUDED_REL_PREFIXES = (
    "models/saved/",
    "data/cache/",
)
EXCLUDED_FILES = {
    "config.yaml",
    ".deploy.lock",
}


def _emit(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Antenna 一键远端同步 + 重启 + 健康检查",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="远端主机")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="SSH 端口")
    parser.add_argument("--user", default=DEFAULT_USER, help="SSH 用户")
    parser.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help="远端项目目录（默认 /data/antenna）",
    )
    parser.add_argument(
        "--password-env",
        default=DEFAULT_PASSWORD_ENV,
        help="读取 SSH 密码的环境变量名（未设置时将交互输入）",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="部署时传入 SKIP_SMOKE_TEST=1",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="远端存在运行时文件变更时继续部署",
    )
    return parser


def _require_paramiko():
    try:
        import paramiko
    except Exception as exc:  # pragma: no cover - 环境差异处理
        raise RuntimeError(
            "缺少 paramiko，请先安装：python -m pip install paramiko"
        ) from exc
    return paramiko


def _should_skip(rel_posix: str, is_dir: bool) -> bool:
    if rel_posix in EXCLUDED_FILES:
        return True
    for prefix in EXCLUDED_REL_PREFIXES:
        if rel_posix.startswith(prefix):
            return True
    if is_dir and Path(rel_posix).name in EXCLUDED_DIR_NAMES:
        return True
    return False


def _create_release_archive(repo_root: Path) -> Path:
    temp = tempfile.NamedTemporaryFile(prefix="antenna-release-", suffix=".tar.gz", delete=False)
    temp.close()
    archive_path = Path(temp.name)
    with tarfile.open(archive_path, mode="w:gz") as tar:
        for path in repo_root.rglob("*"):
            rel = path.relative_to(repo_root)
            rel_posix = rel.as_posix()
            if _should_skip(rel_posix, path.is_dir()):
                continue
            if path.is_file():
                tar.add(path, arcname=rel_posix, recursive=False)
    return archive_path


def _run_remote(ssh, command: str, timeout: int = 1800) -> str:
    _emit(f"\n$ {command}")
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if out.strip():
        _emit(out.rstrip())
    if err.strip():
        _emit(err.rstrip())
    if rc != 0:
        raise RuntimeError(f"远端命令失败 (exit={rc}): {command}")
    return out


def _deploy(args: argparse.Namespace) -> None:
    paramiko = _require_paramiko()
    repo_root = Path(__file__).resolve().parents[1]
    archive_path = _create_release_archive(repo_root)

    password = os.getenv(args.password_env, "")
    if not password:
        password = getpass.getpass(f"{args.user}@{args.host} password: ")
    if not password:
        raise RuntimeError("未提供远端密码，已终止。")

    remote_archive = f"/tmp/antenna-release-{os.getpid()}.tar.gz"
    deploy_flags = "SKIP_GIT_PULL=1 ROLLBACK_ON_FAILURE=1 AUTO_TRAIN_IF_MISSING=1"
    if args.skip_smoke_test:
        deploy_flags += " SKIP_SMOKE_TEST=1"
    if args.allow_dirty:
        deploy_flags += " ALLOW_DIRTY=1"

    ssh = None
    sftp = None
    try:
        _emit(f"[deploy-remote] 打包完成: {archive_path}")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=args.host,
            port=args.port,
            username=args.user,
            password=password,
            timeout=20,
        )
        sftp = ssh.open_sftp()
        _emit(f"[deploy-remote] 上传发布包到 {remote_archive}")
        sftp.put(str(archive_path), remote_archive)

        _run_remote(ssh, f"mkdir -p {args.remote_dir}")
        _run_remote(ssh, f"tar -xzf {remote_archive} -C {args.remote_dir}")
        _run_remote(
            ssh,
            (
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                f"p=Path({json.dumps(posixpath.join(args.remote_dir, 'scripts/deploy_linux.sh'))})\n"
                "txt=p.read_text(encoding='utf-8', errors='replace')\n"
                "p.write_text(txt.replace('\\r\\n','\\n'), encoding='utf-8')\n"
                "PY"
            ),
        )
        _run_remote(ssh, f"chmod +x {posixpath.join(args.remote_dir, 'scripts/deploy_linux.sh')}")
        _run_remote(
            ssh,
            f"cd {args.remote_dir} && {deploy_flags} ./scripts/deploy_linux.sh",
            timeout=3600,
        )
        _run_remote(ssh, "pm2 status")
        _run_remote(
            ssh,
            (
                "python3 - <<'PY'\n"
                "import base64\n"
                "import urllib.request\n"
                "from pathlib import Path\n"
                "import yaml\n"
                f"cfg_path=Path({json.dumps(posixpath.join(args.remote_dir, 'config.yaml'))})\n"
                "cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}\n"
                "mon=(cfg.get('pm2_monitor') or {})\n"
                "port=int(mon.get('port',9615))\n"
                "u=mon.get('username','admin')\n"
                "p=mon.get('password','')\n"
                "health=urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=8).read().decode('utf-8','replace').strip()\n"
                "if health != 'ok':\n"
                "    raise RuntimeError('healthz 返回异常: ' + health)\n"
                "req=urllib.request.Request(f'http://127.0.0.1:{port}/logs/antenna-bot')\n"
                "if p:\n"
                "    token=base64.b64encode((u+':'+p).encode()).decode()\n"
                "    req.add_header('Authorization','Basic '+token)\n"
                "page=urllib.request.urlopen(req, timeout=8).read().decode('utf-8','replace')\n"
                "if '/api/logs/' not in page:\n"
                "    raise RuntimeError('日志页缺少实时接口引用 /api/logs/')\n"
                "print('healthz=ok')\n"
                "print('logs_realtime=ok')\n"
                "PY"
            ),
        )
        _emit("\n[deploy-remote] 发布完成：同步 + 重启 + 健康检查均通过。")
    finally:
        try:
            if ssh is not None:
                _run_remote(ssh, f"rm -f {remote_archive}")
        except Exception:
            pass
        if sftp is not None:
            sftp.close()
        if ssh is not None:
            ssh.close()
        if archive_path.exists():
            archive_path.unlink()


def main() -> None:
    args = _build_parser().parse_args()
    _deploy(args)


if __name__ == "__main__":
    main()
