@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. example_two_sines_gain.py
echo 2. simple_signal_addition.py
echo 3. three_phase_source.py
echo 4. Exit
echo ====================================
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" python example_two_sines_gain.py & pause & goto MENU
if "%choice%"=="2" python simple_signal_addition.py & pause & goto MENU
if "%choice%"=="3" python three_phase_source.py & pause & goto MENU
if "%choice%"=="4" exit

goto MENU