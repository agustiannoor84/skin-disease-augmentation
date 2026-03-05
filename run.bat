@echo off
echo ==============================================
echo  Skin Disease Image Augmentation Setup and Run
echo ==============================================

if not exist .venv\Scripts\python.exe (
    echo [1/3] Membuat Virtual Environment...
    python -m venv .venv
) else (
    echo [1/3] Virtual Environment sudah ada.
)

echo [2/3] Menginstal/Memastikan Library Python terinstal...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo [3/3] Menjalankan program pemrosesan gambar (Preprocess + Augment)...
if not exist data\raw (
    echo File data\raw tidak ditemukan, membuat folder...
    mkdir data\raw
    echo [INFO] Silakan letakkan gambar penyakit kulit Anda di folder data\raw lalu jalankan run.bat lagi.
) else (
    .venv\Scripts\python.exe process_images.py
)

pause
