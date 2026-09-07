# Linux 生产部署运行手册

> 说明：本手册只记录非敏感信息。密码/私钥请放在密码管理器或 CI Secret，禁止写入仓库。

## 目标主机（当前）

- Host: `115.29.240.130`
- Port: `22`
- User: `root`
- App Path: `/data/antenna`

## 一键部署（推荐）

```bash
cd /data/antenna
chmod +x scripts/deploy_linux.sh
SKIP_GIT_PULL=1 ROLLBACK_ON_FAILURE=1 AUTO_TRAIN_IF_MISSING=1 ./scripts/deploy_linux.sh
```

> 当前服务器目录为“非 git 包部署模式”，因此使用 `SKIP_GIT_PULL=1`。

## 固定命令入口（本地直连远端发布）

在项目根目录执行：

```bash
export ANTENNA_REMOTE_PASSWORD='***'
python scripts/deploy_remote.py
```

说明：

- 默认目标：`root@115.29.240.130:22`，部署目录 `/data/antenna`
- 自动流程：打包当前 `HEAD` → 上传到远端并覆盖代码 → 执行 `deploy_linux.sh`（包部署模式）→ 重启 PM2 → 健康检查
- 密码不写入仓库：通过环境变量 `ANTENNA_REMOTE_PASSWORD` 或交互输入

可选参数示例：

```bash
python scripts/deploy_remote.py --skip-smoke-test
python scripts/deploy_remote.py --host 1.2.3.4 --user root --remote-dir /data/antenna
```

## 常用运维命令

### 1) 检查模型是否存在

```bash
cd /data/antenna
find models/saved -maxdepth 1 -type f -name 'model_*.pkl'
```

### 2) 手动训练模型（模型缺失时）

```bash
cd /data/antenna
source .venv/bin/activate
python cli.py train
```

### 3) PM2 状态与重启

```bash
cd /data/antenna
pm2 status
pm2 restart antenna-bot
pm2 restart pm2-monitor
pm2 logs antenna-bot --lines 100
```

### 4) 健康检查

```bash
cd /data/antenna
python scripts/smoke_test.py
```

## 故障快速定位

### 报错：`No model found in models/saved`

处理步骤：

1. 先跑 `python cli.py fetch`（若无缓存数据）
2. 再跑 `python cli.py train`
3. 最后 `pm2 restart antenna-bot`

### 报错：`No objects to concatenate`

表示训练前没有可用行情缓存。先执行：

```bash
python cli.py fetch
```

然后重试训练。
