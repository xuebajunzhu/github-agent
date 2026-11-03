@echo off
rem ============================================================
rem  GitHub Agent - update an existing deployment in place.
rem
rem    scripts\deploy_local.bat
rem
rem  Pulls the latest code, syncs dependencies, runs the test
rem  suite as a gate, then restarts the API. The DB schema is
rem  migrated additively on service startup (init_db).
rem ============================================================
setlocal
cd /d "%~dp0.."

echo [1/4] Pulling latest code ...
git pull --ff-only || (echo [ERROR] git pull failed - resolve locally first & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No venv found. Run scripts\setup_windows.bat first.
    exit /b 1
)

echo [2/4] Syncing dependencies ...
.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt || exit /b 1

echo [3/4] Running test gate ...
.venv\Scripts\python.exe -m pytest -q || (echo [ERROR] tests failed - NOT restarting with a broken build & exit /b 1)

echo [4/4] Restarting API ...
call "%~dp0restart_api.bat"

echo.
echo Deploy finished. Verify: curl http://127.0.0.1:8788/health
endlocal
