@echo off
rem =============================================================================
rem run_db42s02_smc_foc_20k.bat
rem EmbedSim -- DB42S02 SMC FOC closed-loop simulation
rem NANOTEC DB42S02 PMSM  |  AURIX TC3xx 20 kHz
rem
rem Usage:
rem   run_db42s02_smc_foc_20k.bat [--no-tune] [--no-anim]
rem
rem Options:
rem   --no-tune   Skip the interactive gain-tuner prompt (auto-answers 'n')
rem   --no-anim   Skip the interactive animation prompt  (auto-answers 'n')
rem
rem Both flags together give a fully non-interactive run:
rem   run_db42s02_smc_foc_20k.bat --no-tune --no-anim
rem =============================================================================

setlocal enabledelayedexpansion

rem ---------------------------------------------------------------------------
rem Parse flags
rem ---------------------------------------------------------------------------
set NO_TUNE=0
set NO_ANIM=0

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--no-tune" ( set NO_TUNE=1 & shift & goto :parse_args )
if /i "%~1"=="--no-anim" ( set NO_ANIM=1 & shift & goto :parse_args )
echo Unknown option: %~1
echo Usage: %~nx0 [--no-tune] [--no-anim]
exit /b 1
:args_done

rem ---------------------------------------------------------------------------
rem Locate project root via .project_root_marker (walk up from script location)
rem ---------------------------------------------------------------------------
set SCRIPT_DIR=%~dp0
rem Strip trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

set SEARCH_DIR=%SCRIPT_DIR%
set PROJECT_ROOT=

:find_root
if exist "%SEARCH_DIR%\.project_root_marker" (
    set PROJECT_ROOT=%SEARCH_DIR%
    goto :found_root
)
rem Move one level up
for %%I in ("%SEARCH_DIR%\..") do set PARENT=%%~fI
if "%PARENT%"=="%SEARCH_DIR%" (
    echo ERROR: .project_root_marker not found in any parent of %SCRIPT_DIR%
    exit /b 1
)
set SEARCH_DIR=%PARENT%
goto :find_root

:found_root
set EXAMPLE_DIR=%PROJECT_ROOT%\pmsm_smc_smo_example
set SCRIPT=%EXAMPLE_DIR%\db42s02_closed_loop_smc_foc_20k.py

if not exist "%SCRIPT%" (
    echo ERROR: Script not found: %SCRIPT%
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem Build answers for the two interactive prompts
rem ---------------------------------------------------------------------------
set TUNE_ANSWER=y
set ANIM_ANSWER=y
if %NO_TUNE%==1 set TUNE_ANSWER=n
if %NO_ANIM%==1 set ANIM_ANSWER=n

rem ---------------------------------------------------------------------------
rem Summary
rem ---------------------------------------------------------------------------
echo ============================================================
echo   EmbedSim -- DB42S02 SMC FOC 20 kHz
echo   Project root : %PROJECT_ROOT%
echo   Script       : %SCRIPT%
if %NO_TUNE%==1 (echo   Gain tuner   : SKIP) else (echo   Gain tuner   : interactive)
if %NO_ANIM%==1 (echo   Animation    : SKIP) else (echo   Animation    : interactive)
echo ============================================================

rem ---------------------------------------------------------------------------
rem Run  (pipe the two answers through stdin)
rem ---------------------------------------------------------------------------
cd /d "%EXAMPLE_DIR%"

(
    echo %TUNE_ANSWER%
    echo %ANIM_ANSWER%
) | python db42s02_closed_loop_smc_foc_20k.py

echo.
echo ============================================================
echo   Done.
echo   Results in : %EXAMPLE_DIR%
echo   AURIX code : %PROJECT_ROOT%\embedsim_gen\
echo ============================================================

endlocal
