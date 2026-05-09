# Antenna scheduled tasks setup
# Run as Administrator (script auto-elevates)

$ROOT    = "e:\antenna"
$LOG_DIR = "$ROOT\logs"

if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
}

Write-Host "=== Antenna Task Setup ===" -ForegroundColor Cyan

$taskNames = @("Antenna-Train","Antenna-Scan","Antenna-Predict-AM","Antenna-Predict-PM","Antenna-Noon-Review","Antenna-Daily-Review","Antenna-WeeklyTrain")
foreach ($tn in $taskNames) {
    schtasks /delete /tn $tn /f 2>$null | Out-Null
}

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

function New-BatAction([string]$BatPath) {
    return New-ScheduledTaskAction -Execute "cmd.exe" -Argument ('/c "' + $BatPath + '"') -WorkingDirectory $ROOT
}

function New-Weekday([string]$Time) {
    return New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 `
        -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Time
}

# Repeat every 15 minutes for the given duration, weekdays only
function New-WeekdayRepeat([string]$Time, [int]$DurationMinutes) {
    $t = New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 `
        -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Time
    $repClass = Get-CimClass -Namespace 'Root/Microsoft/Windows/TaskScheduler' -ClassName 'MSFT_TaskRepetitionPattern'
    $rep = New-CimInstance -CimClass $repClass -Property @{
        Interval          = 'PT15M'
        Duration          = ('PT' + $DurationMinutes + 'M')
        StopAtDurationEnd = $false
    } -ClientOnly
    $t.Repetition = $rep
    return $t
}

function Reg([string]$Name, $Action, $Trigger) {
    # Recreate settings and principal each call - CimInstance objects cannot be safely reused
    $s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) -StartWhenAvailable
    $p = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType S4U -RunLevel Highest
    Register-ScheduledTask -TaskName $Name -Action $Action -Trigger $Trigger `
        -Settings $s -Principal $p -Force | Out-Null
    Write-Host "[OK] $Name" -ForegroundColor Green
}

Reg "Antenna-Train"        (New-BatAction "$ROOT\scripts\run_train.bat")        (New-ScheduledTaskTrigger -Daily -At "09:00")
Reg "Antenna-Scan"         (New-BatAction "$ROOT\scripts\run_scan.bat")         (New-Weekday "09:15")
Reg "Antenna-Predict-AM"   (New-BatAction "$ROOT\scripts\run_predict.bat")      (New-WeekdayRepeat "09:30" 135)
Reg "Antenna-Predict-PM"   (New-BatAction "$ROOT\scripts\run_predict.bat")      (New-WeekdayRepeat "13:00" 135)
Reg "Antenna-Noon-Review"  (New-BatAction "$ROOT\scripts\run_noon_review.bat")  (New-Weekday "11:32")
Reg "Antenna-Daily-Review" (New-BatAction "$ROOT\scripts\run_daily_review.bat") (New-Weekday "15:32")
# P1 加权重训:每周日 20:00 执行(盘后且用户无感,失败不影响日级链路)
Reg "Antenna-WeeklyTrain"  (New-BatAction "$ROOT\scripts\run_train_weekly.bat") (New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek Sunday -At "20:00")

Write-Host ""
Write-Host "=== Verify ===" -ForegroundColor Cyan
foreach ($tn in $taskNames) {
    $task = Get-ScheduledTask -TaskName $tn -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "  [OK] $tn  LogonType=$($task.Principal.LogonType)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $tn" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done. Logs -> $LOG_DIR" -ForegroundColor Cyan
Write-Host "To test: schtasks /run /tn Antenna-Scan" -ForegroundColor Yellow
Read-Host "Press Enter to exit"
