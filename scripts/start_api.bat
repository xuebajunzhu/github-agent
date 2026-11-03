@echo off
rem Start the GitHub Agent API + 24/7 loop (delegates to the control plane)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0manage.ps1" start
