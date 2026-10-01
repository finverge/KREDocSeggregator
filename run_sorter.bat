@echo off
title KYC Document Sorter
echo Starting KYC Document Sorter...
python main.py --gui
if errorlevel 1 (
    echo.
    echo Application exited with error. Press any key to close.
    pause >nul
)
