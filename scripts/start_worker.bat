@echo off
rem Start the standalone 24/7 worker without HTTP API (optional companion)
cd /d "%~dp0.."
.venv\Scripts\python.exe -m app.worker
