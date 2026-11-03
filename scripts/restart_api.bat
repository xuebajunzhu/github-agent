@echo off
rem ============================================================
rem  GitHub Agent - restart the local API service.
rem  Kills whatever listens on 8788, then starts a fresh,
rem  minimized console running scripts\start_api.bat (the API
rem  process also drives the 24/7 scheduler loop).
rem ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8788 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('Stopping old API process (pid ' + $_ + ')'); Stop-Process -Id $_ -Force }"

rem give the socket a moment to release
timeout /t 2 /nobreak >nul

start "GitHubAgentAPI" /min scripts\start_api.bat

echo Waiting for health check
set TRIES=0
:waitloop
timeout /t 2 /nobreak >nul
curl -sf http://127.0.0.1:8788/health >nul 2>&1
if errorlevel 1 (
    set /a TRIES+=1
    if !TRIES! lss 30 goto waitloop
    echo [ERROR] API did not become healthy within 60s. Check logs\agent.log
    exit /b 1
)
echo API is healthy on http://127.0.0.1:8788
endlocal
