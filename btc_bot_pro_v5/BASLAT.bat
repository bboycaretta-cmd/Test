@echo off
title BTC Bot Pro v5.0 - Trading Panel
color 0B

echo.
echo  ==============================================================
echo           BTC BOT PRO v5.0 - CANLI TRADING PANELI              
echo  ==============================================================
echo.

cd /d "%~dp0"

echo  Dashboard baslatiliyor...
echo.
echo  Tarayici 3 saniye icinde acilacak...
echo  Dashboard: http://127.0.0.1:5000
echo.
echo  Kapatmak icin bu pencereyi kapat veya CTRL+C bas.
echo.
echo  ==============================================================
echo.

timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

python trading_panel.py

pause
