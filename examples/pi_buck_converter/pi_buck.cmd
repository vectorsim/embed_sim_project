@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. pi_buck_example.py
echo 2. pi_buck_ai_tuning.py
echo 3. Exit
echo ====================================
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" python pi_buck_example.py
if "%choice%"=="2" python pi_buck_ai_tuning.py
if "%choice%"=="3" exit

echo.
pause
goto MENU