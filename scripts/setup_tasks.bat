@echo off
:: Antenna scheduled tasks setup (ASCII-only, GBK-safe)
:: Run as Administrator

set ROOT=e:\antenna
set LOG_DIR=%ROOT%\logs

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo === Antenna Task Setup ===

schtasks /delete /tn "Antenna-Train" /f 2>nul
schtasks /delete /tn "Antenna-Scan"  /f 2>nul

echo [1/2] Registering Antenna-Train (daily 09:00)...
schtasks /create /tn "Antenna-Train" /tr "%ROOT%\scripts\run_train.bat" /sc DAILY /st 09:00 /ru "%USERNAME%" /rl HIGHEST /f
if %errorlevel% neq 0 (echo [FAIL] Antenna-Train) else (echo [OK] Antenna-Train)

echo [2/2] Registering Antenna-Scan (MON-FRI 09:15)...
schtasks /create /tn "Antenna-Scan" /tr "%ROOT%\scripts\run_scan.bat" /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 09:15 /ru "%USERNAME%" /rl HIGHEST /f
if %errorlevel% neq 0 (echo [FAIL] Antenna-Scan) else (echo [OK] Antenna-Scan)

echo === Done. Logs: %LOG_DIR% ===
pause
