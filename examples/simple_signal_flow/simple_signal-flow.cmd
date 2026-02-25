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

REM Run selected script
if "%choice%"=="1" python example_two_sines_gain.py
if "%choice%"=="2" python simple_signal_addition.py
if "%choice%"=="3" python three_phase_source.py
if "%choice%"=="4" goto END

REM Pause to see script output
echo.
pause

REM Return to menu
goto MENU

:END
echo Exiting...
pause
