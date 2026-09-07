"""
每日 9:00 定时任务：训练模型，完成后推送飞书通知。
由 Windows 任务计划程序调用：
  python e:\\antenna\\scripts\\task_train.py
"""
import sys
import io
import os
import time

# 强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[task_train] 开始训练 ...")
    t0 = time.time()

    # 调用 CLI 内的训练函数
    import argparse
    from cli import cmd_train
    args = argparse.Namespace(workers=8)
    cmd_train(args, config)

    elapsed = time.time() - t0
    print(f"[task_train] 训练完成，耗时 {elapsed:.0f}s")

    # 推送通知
    from notify import send_train_done
    results = send_train_done(elapsed_seconds=elapsed)
    ok = any(results.values()) if results else False
    print(f"[task_train] 推送{'成功' if ok else '失败或未配置'}: {results}")


if __name__ == "__main__":
    main()
