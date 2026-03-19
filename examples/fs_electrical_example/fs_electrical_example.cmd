@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. db42s02_openloop_fmu.py
echo 2. fs_clarke_park_codegen.py
echo 3. Exit
echo ====================================
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" python db42s02_openloop_fmu.py
if "%choice%"=="2" python fs_clarke_park_codegen.py
if "%choice%"=="3" exit

echo.
pause
goto MENU