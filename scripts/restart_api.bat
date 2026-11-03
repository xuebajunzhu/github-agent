@echo off
rem Restart the GitHub Agent API (delegates to the control plane)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0manage.ps1" restart
