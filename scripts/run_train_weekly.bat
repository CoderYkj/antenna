@echo off
REM ------------------------------------------------------------------
REM Antenna Weekly Train (P1 加权重训)
REM 每周日 20:00 执行:按历史 (signal, hit_tier) 给样本加权的 LightGBM 重训
REM ------------------------------------------------------------------
cd /d "e:\antenna"
if not exist logs mkdir logs
python cli.py train --weighted >> logs\train_weekly.log 2>&1
