@echo off
rem Start the GitHub Agent API + 24/7 loop (used by the scheduled task)
cd /d "%~dp0.."
rem This box exposes the service through a cloudflared tunnel (all inbound
rem connections arrive from 127.0.0.1): trust forwarded client IPs so the
rem per-IP rate limiter works per real visitor, not one shared bucket.
set TRUST_PROXY_HEADERS=true
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8788
