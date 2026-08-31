@echo off
cd /d "%~dp0"
title FlowMind Live - Browser Dashboard
echo Open http://127.0.0.1:5000 after the server starts.
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" live_app.py
) else (
    python live_app.py
)
if errorlevel 1 echo See LIVE_README.md for dependency installation and troubleshooting.
pause
