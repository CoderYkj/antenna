#!/usr/bin/env bash
set -Eeuo pipefail

# Linux 一键更新部署脚本
# 用法:
#   DEPLOY_BRANCH=main APP_DIR=/opt/antenna ./scripts/deploy_linux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-}"
DEPLOY_REF="${DEPLOY_REF:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
PM2_BIN="${PM2_BIN:-pm2}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"
SKIP_GIT_PULL="${SKIP_GIT_PULL:-0}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-0}"
DEPLOY_LOCK_FILE="${DEPLOY_LOCK_FILE:-$APP_DIR/.deploy.lock}"
AUTO_TRAIN_IF_MISSING="${AUTO_TRAIN_IF_MISSING:-1}"
TRAIN_WEIGHTED_IF_MISSING="${TRAIN_WEIGHTED_IF_MISSING:-0}"

log() { printf '[deploy] %s\n' "$*"; }
fail() { printf '[deploy] ERROR: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "缺少命令: $1"
}

require_cmd git
require_cmd "$PYTHON_BIN"
require_cmd "$PM2_BIN"

acquire_lock() {
  if [[ -e "$DEPLOY_LOCK_FILE" ]]; then
    fail "检测到部署锁文件: $DEPLOY_LOCK_FILE（可能已有部署在进行）"
  fi
  : > "$DEPLOY_LOCK_FILE"
}

release_lock() {
  rm -f "$DEPLOY_LOCK_FILE" || true
}

PREV_REV=""
VENV_PY=""
rollback_on_error() {
  local exit_code=$?
  release_lock
  if [[ $exit_code -ne 0 && "$ROLLBACK_ON_FAILURE" == "1" && -n "$PREV_REV" ]]; then
    log "部署失败，执行自动回滚到 $PREV_REV"
    git reset --hard "$PREV_REV" || true
    if [[ -n "$VENV_PY" && -x "$VENV_PY" ]]; then
      ANTENNA_PYTHON="$VENV_PY" "$PM2_BIN" start ecosystem.config.cjs --update-env || true
      "$PM2_BIN" save || true
    fi
  fi
  exit "$exit_code"
}

trap rollback_on_error EXIT

cd "$APP_DIR"
log "工作目录: $APP_DIR"
acquire_lock

if [[ ! -f "requirements.txt" || ! -f "ecosystem.config.cjs" ]]; then
  fail "当前目录不是 Antenna 项目根目录（缺少 requirements.txt 或 ecosystem.config.cjs）"
fi

HAS_GIT_REPO=1
if [[ ! -d ".git" ]]; then
  HAS_GIT_REPO=0
fi

if [[ "$HAS_GIT_REPO" == "1" ]]; then
  if [[ "$ALLOW_DIRTY" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
    fail "检测到未提交改动。请先提交/清理，或设置 ALLOW_DIRTY=1 强制继续。"
  fi
elif [[ "$SKIP_GIT_PULL" != "1" ]]; then
  fail "当前目录不是 git 仓库，无法拉取最新代码。请改用 SKIP_GIT_PULL=1 或先以 git clone 部署。"
fi

CURRENT_BRANCH=""
if [[ "$HAS_GIT_REPO" == "1" ]]; then
  CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if [[ -z "$DEPLOY_BRANCH" ]]; then
    DEPLOY_BRANCH="$CURRENT_BRANCH"
  fi
  PREV_REV="$(git rev-parse HEAD)"
  log "部署前版本: $PREV_REV"
else
  log "检测到非 git 部署目录，进入包部署模式"
fi

if [[ "$SKIP_GIT_PULL" != "1" ]]; then
  [[ "$HAS_GIT_REPO" == "1" ]] || fail "非 git 目录不能执行 git 拉取"
  if [[ -n "$DEPLOY_REF" ]]; then
    log "按指定 ref 部署: $DEPLOY_REF"
    git fetch --prune origin
    git checkout --detach "$DEPLOY_REF"
  else
    log "拉取代码: branch=$DEPLOY_BRANCH"
    git fetch --prune origin
    if [[ "$CURRENT_BRANCH" != "$DEPLOY_BRANCH" ]]; then
      git checkout "$DEPLOY_BRANCH"
    fi
    git pull --ff-only origin "$DEPLOY_BRANCH"
  fi
else
  log "已跳过 git 拉取（SKIP_GIT_PULL=1）"
fi
if [[ "$HAS_GIT_REPO" == "1" ]]; then
  NEW_REV="$(git rev-parse HEAD)"
  log "代码版本: $PREV_REV -> $NEW_REV"
else
  NEW_REV="package-mode"
fi

log "安装 Python 依赖"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

log "检查模型文件"
mkdir -p models/saved
MODEL_COUNT="$(find models/saved -maxdepth 1 -type f -name 'model_*.pkl' | wc -l | tr -d ' ')"
if [[ "${MODEL_COUNT:-0}" == "0" ]]; then
  if [[ "$AUTO_TRAIN_IF_MISSING" == "1" ]]; then
    if [[ "$TRAIN_WEIGHTED_IF_MISSING" == "1" ]]; then
      log "未检测到模型，自动执行加权训练: python cli.py train --weighted"
      "$VENV_PY" cli.py train --weighted
    else
      log "未检测到模型，自动执行标准训练: python cli.py train"
      "$VENV_PY" cli.py train
    fi
    MODEL_COUNT="$(find models/saved -maxdepth 1 -type f -name 'model_*.pkl' | wc -l | tr -d ' ')"
    [[ "${MODEL_COUNT:-0}" != "0" ]] || fail "训练完成后仍未生成 model_*.pkl，请检查训练日志。"
  else
    fail "未检测到模型文件 models/saved/model_*.pkl，请先执行: $VENV_PY cli.py train"
  fi
fi

log "重启 PM2 服务"
mkdir -p logs
# 清理旧进程定义，避免脚本/解释器变更时仍沿用历史配置（如 start.cjs 被 python 解释）。
"$PM2_BIN" delete antenna-bot >/dev/null 2>&1 || true
"$PM2_BIN" delete pm2-monitor >/dev/null 2>&1 || true
ANTENNA_PYTHON="$VENV_PY" "$PM2_BIN" start ecosystem.config.cjs --update-env
"$PM2_BIN" save

log "检查进程状态"
"$PM2_BIN" describe antenna-bot >/dev/null
"$PM2_BIN" describe pm2-monitor >/dev/null

if [[ "$SKIP_SMOKE_TEST" != "1" ]]; then
  log "运行 smoke test"
  "$VENV_PY" scripts/smoke_test.py
fi

log "部署完成。当前版本: $NEW_REV"
log "如需回滚，可在项目目录执行:"
log "  git reset --hard $PREV_REV"
log "  ANTENNA_PYTHON=\"$VENV_PY\" $PM2_BIN start ecosystem.config.cjs --update-env"
release_lock
trap - EXIT
