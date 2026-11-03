@echo off
rem ============================================================
rem  GitHub Agent - restart the local API service.
rem  Kills whatever listens on 8788, then starts a fresh,
rem  minimized console running scripts\start_api.bat (the API
rem  process also drives the 24/7 scheduler loop).
rem ============================================================
setlocal
cd /d "%~dp0.."

set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8788 ^| findstr LISTENING') do (
    echo Stopping old API process (pid %%p) ...
    taskkill /F /PID %%p >nul 2>&1
    set FOUND=1
)
if "%FOUND%"=="0" echo No running API found on 8788.

rem give the socket a moment to release
timeout /t 2 /nobreak >nul

start "GitHubAgentAPI" /min scripts\start_api.bat

echo Waiting for health check ...
set /a TRIES=0
:waitloop
timeout /t 2 /nobreak >nul
curl -sf http://127.0.0.1:8788/health >nul 2>&1
if errorlevel 1 (
    set /a TRIES+=1
    if %TRIES% lss 30 goto waitloop
    echo [ERROR] API did not become healthy within 60s. Check logs\agent.log
    exit /b 1
)
echo API is healthy on http://127.0.0.1:8788
endlocal
