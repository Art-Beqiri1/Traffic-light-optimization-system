@echo off
cd /d "%~dp0"
title FlowMind Live - Desktop
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" desktop_app.py
) else (
    python desktop_app.py
)
if errorlevel 1 echo See LIVE_README.md for dependency installation and troubleshooting.
pause
