@echo off
title BTC Bot Pro v5.0 - Backtest
color 0D

cd /d "%~dp0"

echo.
echo  ==============================================================
echo                 BTC BOT PRO v5.0 - BACKTEST                    
echo  ==============================================================
echo.

python main.py backtest

echo.
pause
