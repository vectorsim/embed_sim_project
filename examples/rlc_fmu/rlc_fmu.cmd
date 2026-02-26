@echo off
title Python Script Runner

:MENU
cls
echo ====================================
echo        Python Script Runner
echo ====================================
echo 1. rlc_fmu.py
echo 2. rlc_fmu_pi_tuning.py
echo 3. rlc_fmu_control_training.py
echo 4. rlc_fmu_nn_training_animation.py
echo 5. rlc_fmu_nn_es.py
echo 6. Exit
echo ====================================
set /p choice="Enter choice (1-6): "

REM Run selected script
if "%choice%"=="1" python rlc_fmu.py
if "%choice%"=="2" python rlc_fmu_pi_tuning.py
if "%choice%"=="3" python rlc_fmu_control_training.py
if "%choice%"=="4" python rlc_fmu_nn_training_animation.py
if "%choice%"=="5" python rlc_fmu_nn_es.py
if "%choice%"=="6" goto END

REM Pause to see script output
echo.
pause

REM Return to menu
goto MENU

:END
echo Exiting...
pause
