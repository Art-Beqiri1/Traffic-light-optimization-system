@echo off
title KujtimHackeri Vehicle Detection
echo ==========================================
echo   KujtimHackeri Vehicle Detection
echo ==========================================
if not exist venv (
    echo Po krijohet virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Po instalohen paketat...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Sistemi po niset...
python app.py
pause
