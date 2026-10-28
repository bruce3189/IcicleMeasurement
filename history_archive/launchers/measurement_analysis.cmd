@echo off
cd /d "%~dp0"
if exist "D:\318\pythonw.exe" (
    start "" "D:\318\pythonw.exe" -m icicle_measure
) else (
    python -m icicle_measure
)
