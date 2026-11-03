@echo off
rem ============================================================
rem  GitHub Agent - one-command fresh install (Windows)
rem
rem    scripts\setup_windows.bat
rem
rem  Creates the venv, installs dependencies, seeds .env with
rem  RANDOM SECRET_KEY/ADMIN_TOKEN (never the documented defaults),
rem  initializes the database and runs a smoke test.
rem  Idempotent: safe to re-run over an existing install.
rem ============================================================
setlocal
cd /d "%~dp0.."

echo [1/6] Checking Python ...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.11+ first.
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo [2/6] venv already exists - reusing.
) else (
    echo [2/6] Creating venv ...
    python -m venv .venv || exit /b 1
)

echo [3/6] Installing core dependencies ...
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt -r requirements-dev.txt || exit /b 1
echo        (optional extras: pip install -r requirements-ml.txt / requirements-speech.txt)

if exist ".env" (
    echo [4/6] .env already exists - keeping it.
) else (
    echo [4/6] Seeding .env with RANDOM secrets ...
    .venv\Scripts\python.exe -c "import secrets;from pathlib import Path;t=Path('.env.example').read_text(encoding='utf-8');t=t.replace('please-change-me-to-a-random-string',secrets.token_urlsafe(48)).replace('change-me-admin-token',secrets.token_urlsafe(24));Path('.env').write_text(t,encoding='utf-8');print('       .env created (SECRET_KEY/ADMIN_TOKEN randomized)')"
    if errorlevel 1 exit /b 1
)

echo [5/6] Initializing database (additive migrations run at startup too) ...
.venv\Scripts\python.exe -c "from app.database import create_db_engine, init_db;from app.config import Settings;init_db(create_db_engine(Settings().database_url));print('       database ready')" || exit /b 1

echo [6/6] Smoke test ...
.venv\Scripts\python.exe -m pytest -q tests/test_api.py tests/test_mcp.py || exit /b 1

echo.
echo ============================================================
echo  Install complete. Next steps:
echo    start the 7x24 service:   scripts\start_api.bat
echo    register autostart:       scripts\register_autostart.bat
echo    update an existing box:   scripts\deploy_local.bat
echo    UI:  http://127.0.0.1:8788/   API docs: /docs   MCP: POST /mcp
echo ============================================================
endlocal
