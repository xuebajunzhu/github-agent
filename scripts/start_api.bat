@echo off
rem Start the GitHub Agent API + 24/7 loop (used by the scheduled task)
cd /d "%~dp0.."
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8788
