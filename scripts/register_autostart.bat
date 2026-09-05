@echo off
rem ============================================================
rem  GitHub Agent - register the public API to start at logon.
rem  The API process also runs the 24/7 fetch/refresh/patrol loop.
rem  Remove later with:
rem    schtasks /delete /tn "GitHubAgentAPI" /f
rem ============================================================
setlocal
set PROJECT_DIR=%~dp0..
set LAUNCHER=%PROJECT_DIR%\scripts\start_api.bat

if not exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] venv python not found. Install dependencies first.
    pause
    exit /b 1
)

schtasks /create /tn "GitHubAgentAPI" /tr "\"%LAUNCHER%\"" /sc onlogon /rl limited /f
if %errorlevel%==0 (
    echo.
    echo [OK] Task "GitHubAgentAPI" registered. The agent (API + 24/7 loop) will
    echo      start at every logon on http://0.0.0.0:8788
    echo      Start it now with:  schtasks /run /tn "GitHubAgentAPI"
) else (
    echo [ERROR] Failed to register the scheduled task.
)
pause
