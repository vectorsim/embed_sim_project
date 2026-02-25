@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. parallel_parking.py
echo 2. simple_circular_motion.py
echo 3. vehicle_fmu_animation.py
echo 4. Exit
echo ====================================
set /p choice="Enter choice (1-4): "

REM Run selected script
if "%choice%"=="1" python parallel_parking.py
if "%choice%"=="2" python simple_circular_motion.py
if "%choice%"=="3" python vehicle_fmu_animation.py
if "%choice%"=="4" goto END

REM Pause to see script output
echo.
pause

REM Return to menu
goto MENU

:END
echo Exiting...
pause
