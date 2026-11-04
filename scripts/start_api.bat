@echo off
rem Direct launcher: the control plane (manage.ps1) spawns this via WMI so the
rem API process breaks away from any scheduled-task job object and survives
rem the task's own exit. Do not delegate back to manage.ps1 from here.
cd /d "%~dp0.."
set TRUST_PROXY_HEADERS=true
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8788
