@echo off
if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual Environment .venv tidak ditemukan!
    echo Silakan jalankan run.bat terlebih dahulu.
    pause
    exit /b
)

if "%~1"=="" (
    echo [USAGE] predict.bat ^<path_ke_gambar^>
    echo Contoh: predict.bat data\raw\athlete_foot\athlete_1.png
    pause
    exit /b
)

.venv\Scripts\python.exe predict.py "%~1"
pause
