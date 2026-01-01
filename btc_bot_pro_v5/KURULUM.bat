@echo off
title BTC Bot Pro v5.0 - Kurulum
color 0A

echo.
echo  ==============================================================
echo                 BTC BOT PRO v5.0 - KURULUM                     
echo  ==============================================================
echo.

cd /d "%~dp0"

echo  [1/4] Python kontrol ediliyor...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  HATA: Python bulunamadi!
    echo  Python 3.10+ yukleyin: https://python.org/downloads
    echo  Yuklerken "Add Python to PATH" secenegini isaretleyin!
    echo.
    pause
    exit /b 1
)
echo        [OK] Python bulundu
echo.

echo  [2/4] pip guncelleniyor...
python -m pip install --upgrade pip -q
echo        [OK] pip guncellendi
echo.

echo  [3/4] Gerekli paketler yukleniyor...
echo        (Bu islem birkac dakika surebilir)
echo.
pip install numpy pandas scikit-learn requests flask pyyaml -q
echo        [OK] Paketler yuklendi
echo.

echo  [4/4] Kurulum test ediliyor...
python -c "from core import *; print('        [OK] Tum moduller calisiyor!')"
echo.

echo  ==============================================================
echo                    KURULUM TAMAMLANDI!                         
echo  ==============================================================
echo.
echo  Simdi BASLAT.bat dosyasina cift tikla!
echo.

pause
