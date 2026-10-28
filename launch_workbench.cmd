@echo off
setlocal
cd /d "%~dp0Measurement" || exit /b 1
set "QT_QPA_PLATFORM=windows"
if exist "D:\318\pythonw.exe" (
    start "" "D:\318\pythonw.exe" -m icicle_measure --pseudo-depth
) else (
    python -m icicle_measure --pseudo-depth
)
endlocal
