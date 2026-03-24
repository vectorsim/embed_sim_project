@echo off
setlocal enabledelayedexpansion

:: ============================================================================
::  SMC FOC Workflow Script  -  NANOTEC DB42S02  /  EmbedSim
::  Paul Abraham 2025
:: ============================================================================

:: Save current codepage and force CP437 so box characters render correctly.
for /f "tokens=2 delims=:" %%C in ('chcp') do set _ORIG_CP=%%C
set _ORIG_CP=!_ORIG_CP: =!
chcp 437 >nul 2>&1

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: ---- Locate Python ----------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

:: ============================================================================
::  HEADER + MENU
:: ============================================================================
call :header

echo  Python  : !PYTHON!
echo.
echo  +----------------------------------------------------------------+
echo  ^|                       SELECT MODE                             ^|
echo  +----------------------------------------------------------------+
echo  ^|  T.  SMOKE TEST  (2 min: DE=5 GP=3 t_sim=0.5s 400 RPM)       ^|  ^<-- test first
echo  ^|  0.  SIMULATE ^& CODEGEN only  (load existing gains JSON)     ^|  ^<-- small PC
echo  ^|  1.  Complete Workflow         (Tune -^> Simulate -^> CodeGen)  ^|
echo  ^|  2.  Quick Test                (20 DE / 15 GP  ~3-5 min)     ^|
echo  ^|  3.  Production Tuning         (200 DE / 100 GP ~45-60 min)  ^|
echo  ^|  4.  Standalone Tuner only     (no code generation)          ^|
echo  ^|  5.  Standalone Tuner + Plot                                  ^|
echo  ^|  6.  Custom  (interactive)                                    ^|
echo  ^|  H.  Help                                                     ^|
echo  ^|  Q.  Quit                                                     ^|
echo  +----------------------------------------------------------------+
echo.
set /p CHOICE="  Enter choice: "

if /i "!CHOICE!"=="t" goto :smoke_test
if /i "!CHOICE!"=="q" goto :end
if /i "!CHOICE!"=="h" goto :help
if    "!CHOICE!"=="0" goto :simulate_only
if    "!CHOICE!"=="1" goto :complete_workflow
if    "!CHOICE!"=="2" goto :quick_test
if    "!CHOICE!"=="3" goto :production
if    "!CHOICE!"=="4" goto :tuner_only
if    "!CHOICE!"=="5" goto :tuner_plot
if    "!CHOICE!"=="6" goto :custom

echo  [ERROR] Invalid choice.
goto :end


:: ============================================================================
::  OPTION T  -  SMOKE TEST  (fast end-to-end, ~2 minutes)
:: ============================================================================
:smoke_test
call :section_banner "SMOKE TEST  (end-to-end in ~2 minutes)"
echo  DE=5 / GP=3 / 400 RPM / t_sim=0.5s / dt=100us
echo  Verifies the full pipeline: tuner -> simulate -> codegen -> plot.
echo  Not for production gains -- use to confirm the install works.
echo.
call :confirm_start "Start smoke test?" || goto :end

call :step_banner 1 3 "DE tuning  (5 iters, fast)"
call :run_timed !PYTHON! db42s02_tune_simulate_codegen.py ^
    --rpm 400 --de_iters 5 --gp_iters 3 --t_sim 0.5 --dt 100e-6
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Smoke test"
goto :end


:: ============================================================================
::  OPTION 0  -  SIMULATE + CODEGEN only  (skip tuning, light on CPU)
:: ============================================================================
:simulate_only
call :section_banner "SIMULATE + CODEGEN  (no tuning)"
echo  Loads gains from  smc_best_gains.json  (or header defaults).
echo  Skips the DE/GP optimiser -- fast on any PC.
echo.
set /p SIM_RPM="  Target RPM [default 2000]: "
if "!SIM_RPM!"=="" set SIM_RPM=2000

set /p SIM_T="  Simulation time [s, default 2.0]: "
if "!SIM_T!"=="" set SIM_T=2.0

set /p SIM_DT_US="  Time step [us, default 50]: "
if "!SIM_DT_US!"=="" set SIM_DT_US=50

echo.
echo  +------------------------------------------------------------------+
echo  ^|  RPM   : !SIM_RPM!
echo  ^|  t_sim : !SIM_T! s
echo  ^|  dt    : !SIM_DT_US! us  (20 kHz PWM)
echo  +------------------------------------------------------------------+
echo.
call :confirm_start "Start simulate + codegen?" || goto :end

call :step_banner 1 2 "Loading gains / preparing simulation"
call :run_timed !PYTHON! db42s02_tune_simulate_codegen.py ^
    --no-tune ^
    --rpm !SIM_RPM! ^
    --t_sim !SIM_T! ^
    --dt !SIM_DT_US!e-6
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Simulate + CodeGen"
goto :end


:: ============================================================================
::  OPTION 1  -  COMPLETE WORKFLOW
:: ============================================================================
:complete_workflow
call :section_banner "COMPLETE WORKFLOW  (Tune -> Simulate -> CodeGen)"
echo  DE=100 / GP=50 / 2000 RPM / 2.0 s
echo  Expected: 15-20 min on a modern PC
echo.
call :confirm_start "Start complete workflow?" || goto :end

call :step_banner 1 3 "DE + GP Bayesian tuning  (100 DE / 50 GP)"
call :run_timed !PYTHON! db42s02_tune_simulate_codegen.py ^
    --rpm 2000 --de_iters 100 --gp_iters 50 --t_sim 2.0
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Complete workflow"
goto :end


:: ============================================================================
::  OPTION 2  -  QUICK TEST
:: ============================================================================
:quick_test
call :section_banner "QUICK TEST  (fast tuning)"
echo  DE=20 / GP=15 / 1000 RPM / 1.5 s  (~3-5 min)
echo.
call :confirm_start "Start quick test?" || goto :end

call :step_banner 1 3 "DE + GP tuning  (20 DE / 15 GP)"
call :run_timed !PYTHON! db42s02_tune_simulate_codegen.py ^
    --rpm 1000 --de_iters 20 --gp_iters 15 --t_sim 1.5
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Quick test"
goto :end


:: ============================================================================
::  OPTION 3  -  PRODUCTION TUNING
:: ============================================================================
:production
call :section_banner "PRODUCTION TUNING  (maximum quality)"
echo  DE=200 / GP=100 / 3000 RPM / 2.0 s
echo  WARNING: ~45-60 minutes -- do not use on a slow PC
echo.
call :confirm_start "Start production tuning?" || goto :end

call :step_banner 1 3 "DE global search  (200 iters)"
call :run_timed !PYTHON! db42s02_tune_simulate_codegen.py ^
    --rpm 3000 --de_iters 200 --gp_iters 100 --t_sim 2.0
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Production tuning"
goto :end


:: ============================================================================
::  OPTION 4  -  STANDALONE TUNER
:: ============================================================================
:tuner_only
call :section_banner "STANDALONE TUNER  (gains only, no codegen)"
echo.
call :confirm_start "Start tuner (DE=80, GP=40, 400 RPM)?" || goto :end

call :step_banner 1 1 "Running DE + GP optimiser"
call :run_timed !PYTHON! smc_fmu_tuner.py ^
    --rpm 400 --de_iters 80 --gp_iters 40 --out smc_tuned_gains.json
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Standalone tuner"
goto :end


:: ============================================================================
::  OPTION 5  -  STANDALONE TUNER WITH PLOT
:: ============================================================================
:tuner_plot
call :section_banner "STANDALONE TUNER + VERIFICATION PLOT"
echo.
set /p RPM5="  Target RPM [default 400]: "
if "!RPM5!"=="" set RPM5=400
set /p DE5="  DE iters [default 60]: "
if "!DE5!"=="" set DE5=60
set /p GP5="  GP iters [default 30]: "
if "!GP5!"=="" set GP5=30

call :confirm_start "Start tuner + plot (RPM=!RPM5!, DE=!DE5!, GP=!GP5!)?" || goto :end

call :step_banner 1 2 "Running DE + GP optimiser + verification plot"
call :run_timed !PYTHON! smc_fmu_tuner.py ^
    --rpm !RPM5! --de_iters !DE5! --gp_iters !GP5! ^
    --verify --out smc_tuned_gains.json
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Tuner + verification plot"
goto :end


:: ============================================================================
::  OPTION 6  -  CUSTOM (INTERACTIVE)
:: ============================================================================
:custom
call :section_banner "CUSTOM PARAMETERS"
echo.
set /p CRPM="  Target RPM [default 2000]: "
if "!CRPM!"=="" set CRPM=2000
set /p CDE="  DE iters [default 50]: "
if "!CDE!"=="" set CDE=50
set /p CGP="  GP iters [default 30]: "
if "!CGP!"=="" set CGP=30
set /p CTSIM="  Simulation time [s, default 2.0]: "
if "!CTSIM!"=="" set CTSIM=2.0
set /p CDT_US="  Time step [us, default 50]: "
if "!CDT_US!"=="" set CDT_US=50

echo.
echo  +------------------------------------------------------------------+
echo  ^|  RPM=!CRPM!  DE=!CDE!  GP=!CGP!  t_sim=!CTSIM!s  dt=!CDT_US!us
echo  +------------------------------------------------------------------+
echo.
call :confirm_start "Start with these settings?" || goto :end

call :step_banner 1 3 "Custom tuning run"
call :run_timed !PYTHON! db42s02_tune_simulate_codegen.py ^
    --rpm !CRPM! --de_iters !CDE! --gp_iters !CGP! ^
    --t_sim !CTSIM! --dt !CDT_US!e-6
set RESULT=!ERRORLEVEL!
call :step_done !RESULT! "Custom run"
goto :end


:: ============================================================================
::  HELP
:: ============================================================================
:help
cls
call :header
echo  +--------------------------------------------------------------------+
echo  ^|                   EmbedSim SMC FOC  HELP                          ^|
echo  +--------------------------------------------------------------------+
echo  ^|  OPTION T  --  Smoke Test  (~2 minutes, verifies install)        ^|
echo  ^|    DE=5 GP=3 t_sim=0.5s dt=100us 400 RPM.                       ^|
echo  ^|    Confirms tuner, simulate, codegen and plot all work.          ^|
echo  ^|    Gains are NOT production quality -- use Option 1+ for that.  ^|
echo  ^|                                                                    ^|
echo  ^|  OPTION 0  --  Simulate + CodeGen  (RECOMMENDED for slow PCs)    ^|
echo  ^|    Loads smc_best_gains.json (tuned previously).                  ^|
echo  ^|    Runs closed-loop simulation at 20 kHz.                         ^|
echo  ^|    Emits embedsim_gen/embedsim_step.c/.h for AURIX TC3xx.         ^|
echo  ^|    Generates db42s02_smc_foc_results.png.                         ^|
echo  ^|    Typical time: 30-120 seconds depending on t_sim.               ^|
echo  ^|                                                                    ^|
echo  ^|  OPTION 1  --  Complete Workflow  (Tune -^> Simulate -^> CodeGen)  ^|
echo  ^|    Runs DE global search then GP Bayesian refinement.             ^|
echo  ^|    Best for first-time commissioning at a new operating point.    ^|
echo  ^|                                                                    ^|
echo  ^|  ITERATION GUIDE:                                                  ^|
echo  ^|    Quick Test   : DE=20,  GP=15   ~3-5 min                        ^|
echo  ^|    Balanced     : DE=50,  GP=30   ~10-15 min                      ^|
echo  ^|    Production   : DE=100, GP=50   ~20-30 min                      ^|
echo  ^|    Maximum      : DE=200, GP=100  ~45-60 min                      ^|
echo  ^|                                                                    ^|
echo  ^|  COST FUNCTION:                                                    ^|
echo  ^|    J = ITAE + 2*overshoot^2 + 0.05*chattering + 0.1*^|iq_ss^|    ^|
echo  ^|                                                                    ^|
echo  ^|  OUTPUT FILES:                                                     ^|
echo  ^|    smc_best_gains.json           -- optimal SMC gains              ^|
echo  ^|    db42s02_smc_foc_results.png   -- 8-panel performance plot       ^|
echo  ^|    embedsim_gen/embedsim_step.c  -- AURIX TC3xx C code             ^|
echo  ^|    embedsim_gen/embedsim_step.h  -- generated header               ^|
echo  +--------------------------------------------------------------------+
echo.
pause
goto :end


:: ============================================================================
::  SUB-ROUTINES
:: ============================================================================

:: ---- :header ----------------------------------------------------------------
:header
cls
echo.
echo  ==================================================================
echo    EmbedSim  *  NANOTEC DB42S02  *  SMC FOC Workflow
echo    AURIX TC3xx  *  ISO 26262  *  Paul Abraham  2025
echo  ==================================================================
echo.
exit /b 0


:: ---- :section_banner <title> -----------------------------------------------
:section_banner
cls
call :header
echo  ==================================================================
echo    %~1
echo  ==================================================================
echo.
exit /b 0


:: ---- :step_banner <step_num> <total_steps> <description> -------------------
:step_banner
echo.
echo  +------------------------------------------------------------------+
echo  ^|  STEP %~1 / %~2  --  %~3
echo  +------------------------------------------------------------------+
exit /b 0


:: ---- :confirm_start <message>   returns errorlevel 1 if user says n --------
:confirm_start
set /p _CFM="  %~1 (y/n): "
if /i "!_CFM!"=="y" exit /b 0
echo  Cancelled.
exit /b 1


:: ---- :run_timed <command + args>   runs command, prints wall-clock time -----
:run_timed
echo.
echo  >> Started  : !TIME!
echo  >> Running  : %*
echo.
echo  ------------------------------------------------------------------
set _T0=%TIME%
%*
set _ERR=!ERRORLEVEL!
set _T1=%TIME%
echo  ------------------------------------------------------------------
echo.
echo  >> Finished : !TIME!
call :elapsed "!_T0!" "!_T1!"
exit /b !_ERR!


:: ---- :elapsed <start_time> <end_time>   prints elapsed time ----------------
:elapsed
setlocal
set _S=%~1
set _E=%~2
for /f "tokens=1-3 delims=:." %%A in ("!_S!") do (
    set /a _SH=1%%A-100, _SM=1%%B-100, _SS=1%%C-100
)
for /f "tokens=1-3 delims=:." %%A in ("!_E!") do (
    set /a _EH=1%%A-100, _EM=1%%B-100, _ES=1%%C-100
)
set /a _ELAPSED=(_EH*3600+_EM*60+_ES)-(_SH*3600+_SM*60+_SS)
if !_ELAPSED! LSS 0 set /a _ELAPSED+=86400
set /a _DH=_ELAPSED/3600
set /a _DM=(_ELAPSED%%3600)/60
set /a _DS=_ELAPSED%%60
if !_DH! GTR 0 (
    echo  >> Elapsed  : !_DH!h !_DM!m !_DS!s
) else if !_DM! GTR 0 (
    echo  >> Elapsed  : !_DM!m !_DS!s
) else (
    echo  >> Elapsed  : !_DS!s
)
endlocal
exit /b 0


:: ---- :step_done <errorlevel> <label> ----------------------------------------
:step_done
if NOT "%~1"=="0" goto :_step_error

echo.
echo  ==================================================================
echo    [OK]  %~2 completed successfully.
echo  ==================================================================
echo.
echo  Output files written:
if exist "db42s02_smc_foc_results.png"       echo    [+] db42s02_smc_foc_results.png
if exist "smc_best_gains.json"               echo    [+] smc_best_gains.json
if exist "smc_tuned_gains.json"              echo    [+] smc_tuned_gains.json
if exist "embedsim_gen\embedsim_step.c"     echo    [+] embedsim_gen\embedsim_step.c
if exist "embedsim_gen\embedsim_step.h"     echo    [+] embedsim_gen\embedsim_step.h
if exist "smc_tuner_verify.png"              echo    [+] smc_tuner_verify.png
exit /b 0

:_step_error
echo.
echo  ==================================================================
echo    [ERROR]  %~2 failed  (exit code %~1)
echo  ------------------------------------------------------------------
echo    Troubleshooting:
echo      * Python version:  python --version   (need 3.8+)
echo      * Install deps:    pip install -r requirements.txt
echo      * Check fs_electrical_machines/ folder is present
echo      * Option 0: ensure smc_best_gains.json exists first
echo  ==================================================================
exit /b 0


:: ============================================================================
:end
echo.
echo  ==================================================================
echo    Script finished.
echo  ==================================================================
echo.
chcp !_ORIG_CP! >nul 2>&1
pause
endlocal
