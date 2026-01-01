@echo off
title BTC Bot Pro v5.0 - Canli Sinyal
color 0C

cd /d "%~dp0"

echo.
echo  ==============================================================
echo                 BTC BOT PRO v5.0 - CANLI SINYAL                
echo  ==============================================================
echo.
echo  NOT: Bu bir simulasyondur, gercek trade yapmaz.
echo  Cikmak icin CTRL+C bas.
echo.
echo  ==============================================================
echo.

python main.py live

pause
