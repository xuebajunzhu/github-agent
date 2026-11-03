@echo off
rem Update an existing deployment (delegates to the control plane)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0manage.ps1" deploy
