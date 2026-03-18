@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. example_algebraic_loop.py
echo 2. Exit
echo ====================================
set /p choice="Enter choice (1-2): "

if "%choice%"=="1" python example_algebraic_loop.py & pause & goto MENU
if "%choice%"=="2" exit

goto MENU