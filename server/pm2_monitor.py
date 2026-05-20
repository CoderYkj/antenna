"""
pm2_monitor.py - Antenna PM2 进程 Web 监控面板。

功能:
  - Basic Auth 保护(账号密码配置在 config.yaml.pm2_monitor)
  - 首页展示所有 PM2 进程状态(name / pid / uptime / cpu / mem / status)
  - 支持重启 / 停止 / 启动指定进程
  - 查看最近 N 行日志

配置(config.yaml):
  pm2_monitor:
    host:      "0.0.0.0"      # 监听地址(默认 127.0.0.1 仅本机)
    port:      9615           # 端口
    username:  "admin"        # 登录账号
    password:  "change_me"    # 登录密码
    log_lines: 200            # /logs 返回最近多少行
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import time
from functools import wraps
from pathlib import Path

import yaml
from flask import Flask, Response, abort, jsonify, redirect, request, url_for

app = Flask(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


# ── 配置 ────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    """每次读配置,支持改完 config.yaml 不用重启监控。"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _mon_cfg() -> dict:
    return _load_cfg().get("pm2_monitor") or {}


def _pm2_bin() -> str:
    """找 pm2 可执行路径。Windows 下 pm2 通常是 pm2.cmd。"""
    for candidate in ("pm2", "pm2.cmd"):
        p = shutil.which(candidate)
        if p:
            return p
    return "pm2"


# ── Basic Auth ──────────────────────────────────────────────────────────

def _require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        cfg      = _mon_cfg()
        username = cfg.get("username") or "admin"
        password = cfg.get("password") or ""

        if not password:
            abort(500, description="pm2_monitor.password 未配置,拒绝启动以避免裸奔")

        auth = request.authorization
        if not auth or auth.username != username or auth.password != password:
            return Response(
                "Authentication required",
                status=401,
                headers={"WWW-Authenticate": 'Basic realm="Antenna PM2 Monitor"'},
            )
        return fn(*args, **kwargs)
    return wrapper


# ── PM2 交互 ────────────────────────────────────────────────────────────

def _pm2(*subcmd: str, timeout: int = 15) -> tuple[int, str, str]:
    """运行 pm2 子命令,返回 (returncode, stdout, stderr)。"""
    try:
        r = subprocess.run(
            [_pm2_bin(), *subcmd],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", "pm2 可执行文件未找到"
    except subprocess.TimeoutExpired:
        return 124, "", f"pm2 {' '.join(subcmd)} 超时"


def _pm2_jlist() -> list[dict]:
    rc, out, err = _pm2("jlist")
    if rc != 0 or not out.strip():
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def _format_uptime(pm_uptime_ms: int | None) -> str:
    if not pm_uptime_ms:
        return "—"
    seconds = max(0, int((time.time() * 1000 - pm_uptime_ms) / 1000))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _format_mem(bytes_: int | None) -> str:
    if not bytes_:
        return "—"
    mb = bytes_ / 1024 / 1024
    return f"{mb:.1f} MB"


# ── 路由 ────────────────────────────────────────────────────────────────

_PAGE_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<title>Antenna PM2 Monitor</title>
<meta http-equiv="refresh" content="10">
<style>
  body { font: 14px/1.5 -apple-system, "Segoe UI", sans-serif; background: #0e1116; color: #e6edf3; margin: 0; padding: 24px; }
  h1 { margin: 0 0 8px; font-size: 20px; }
  .meta { color: #8b949e; margin-bottom: 16px; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 6px; overflow: hidden; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #30363d; }
  th { background: #21262d; font-weight: 600; }
  tr:last-child td { border-bottom: none; }
  .st-online  { color: #3fb950; font-weight: 600; }
  .st-errored { color: #f85149; font-weight: 600; }
  .st-stopped { color: #8b949e; font-weight: 600; }
  .st-launching { color: #d29922; font-weight: 600; }
  .actions { white-space: nowrap; }
  form { display: inline-block; margin: 0 2px; }
  button { background: #21262d; color: #e6edf3; border: 1px solid #30363d;
           padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  button:hover { background: #30363d; }
  .btn-danger { color: #f85149; }
  .btn-primary { color: #3fb950; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .empty { padding: 32px; text-align: center; color: #8b949e; }
</style>
</head><body>
<h1>Antenna PM2 Monitor</h1>
<div class="meta">
  每 10 秒自动刷新 · 最后更新 __NOW__ · 共 __COUNT__ 个进程
  · <a href="__REFRESH_URL__">立即刷新</a>
</div>
__BODY__
</body></html>"""


@app.route("/")
@_require_auth
def index():
    procs = _pm2_jlist()
    if not procs:
        body = '<div class="empty">没有运行的 PM2 进程,或 pm2 未安装。</div>'
    else:
        rows = []
        for p in procs:
            name      = p.get("name", "—")
            pid       = p.get("pid") or "—"
            status    = p.get("pm2_env", {}).get("status", "—")
            uptime    = _format_uptime(p.get("pm2_env", {}).get("pm_uptime"))
            restarts  = p.get("pm2_env", {}).get("restart_time", 0)
            cpu       = p.get("monit", {}).get("cpu", 0)
            mem       = _format_mem(p.get("monit", {}).get("memory"))
            st_class  = {"online": "st-online", "stopped": "st-stopped",
                         "errored": "st-errored", "launching": "st-launching"}.get(status, "")
            rows.append(
                f"<tr>"
                f"<td>{html.escape(str(name))}</td>"
                f"<td>{pid}</td>"
                f"<td class='{st_class}'>{html.escape(str(status))}</td>"
                f"<td>{html.escape(uptime)}</td>"
                f"<td>↺ {restarts}</td>"
                f"<td>{cpu}%</td>"
                f"<td>{html.escape(mem)}</td>"
                f"<td class='actions'>"
                f"  <form method='post' action='/action/restart/{html.escape(str(name))}'>"
                f"    <button class='btn-primary' type='submit'>Restart</button></form>"
                f"  <form method='post' action='/action/stop/{html.escape(str(name))}'>"
                f"    <button class='btn-danger' type='submit'>Stop</button></form>"
                f"  <form method='post' action='/action/start/{html.escape(str(name))}'>"
                f"    <button type='submit'>Start</button></form>"
                f"  <a href='/logs/{html.escape(str(name))}'>Logs</a>"
                f"</td>"
                f"</tr>"
            )
        body = (
            "<table><thead><tr>"
            "<th>Name</th><th>PID</th><th>Status</th><th>Uptime</th>"
            "<th>Restarts</th><th>CPU</th><th>Memory</th><th>Actions</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    return (
        _PAGE_HTML
        .replace("__NOW__", time.strftime("%Y-%m-%d %H:%M:%S"))
        .replace("__COUNT__", str(len(procs)))
        .replace("__REFRESH_URL__", url_for("index"))
        .replace("__BODY__", body)
    )


@app.route("/action/<action>/<name>", methods=["POST"])
@_require_auth
def do_action(action: str, name: str):
    if action not in ("restart", "stop", "start"):
        abort(400, description="invalid action")
    if not name or "/" in name or " " in name:
        abort(400, description="invalid process name")
    rc, out, err = _pm2(action, name)
    if rc != 0:
        return f"<pre>{html.escape(err or out)}</pre>", 500
    return redirect(url_for("index"))


@app.route("/logs/<name>")
@_require_auth
def logs(name: str):
    if not name or "/" in name or " " in name:
        abort(400, description="invalid process name")
    lines = _mon_cfg().get("log_lines", 200)
    rc, out, err = _pm2("logs", name, "--lines", str(lines), "--nostream")
    text = out or err or "(no logs)"
    return Response(
        f"<pre style='background:#0e1116;color:#e6edf3;padding:20px;"
        f"font:12px/1.4 monospace;white-space:pre-wrap'>"
        f"<a href='/' style='color:#58a6ff'>&larr; back</a>\n\n"
        f"{html.escape(text)}</pre>",
        mimetype="text/html",
    )


@app.route("/api/processes")
@_require_auth
def api_processes():
    return jsonify(_pm2_jlist())


@app.route("/healthz")
def healthz():
    """裸路径健康检查,无需鉴权,用于外部监控。"""
    return "ok", 200


# ── 入口 ────────────────────────────────────────────────────────────────

def main():
    cfg  = _mon_cfg()
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 9615))
    if not cfg.get("password"):
        raise RuntimeError(
            "config.yaml.pm2_monitor.password 未配置,拒绝启动以避免监控面板裸奔。"
        )
    print(f"[pm2_monitor] 启动 http://{host}:{port}  user={cfg.get('username','admin')}")
    # 生产环境用 waitress / gunicorn 更好,这里用 Flask 自带服务器足够
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
