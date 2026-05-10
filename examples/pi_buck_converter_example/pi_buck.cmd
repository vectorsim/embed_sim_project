@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. pi_buck_example.py
echo 2. exit

echo ====================================
set /p choice="Enter choice (1-2): "

if "%choice%"=="1" python pi_buck_example.py
if "%choice%"=="2" exit

echo.
pause
goto MENU