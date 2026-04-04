@echo off
REM =============================================================================
REM build_all.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Build all EmbedSim Cython extensions in correct dependency order:
REM
REM   1. coordinate_transform  (embed_sim_matrix.c + embed_sim_coordinate_transform.c)
REM   2. motor_utility_blocks  (embed_sim_motor_utility_blocks.c)
REM   3. smc_controller        (embed_sim_smc_controller.c + coord transform + matrix)
REM   4. svpwm                 (embed_sim_sv_pwm.c + matrix)
REM   5. dfc_controller        (dfc_controller_wrapper)
REM
REM Usage:
REM   cd fs_electrical_machines\c_src
REM   build_all.bat
REM
REM Pass --clean-only to just wipe all build artefacts without compiling.
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Parse arguments
set "CLEAN_ONLY=0"
if /i "%~1"=="--clean-only" set "CLEAN_ONLY=1"

REM Colour helpers (using ANSI escape codes if available)
set "RED="
set "GREEN="
set "YELLOW="
set "CYAN="
set "BOLD="
set "RESET="
REM Check if running in a terminal that supports ANSI (Windows 10+)
for /f "tokens=2 delims=:" %%a in ('chcp') do set "CODE_PAGE=%%a"
if "%CODE_PAGE%"=="65001" (
    set "RED=[31m"
    set "GREEN=[32m"
    set "YELLOW=[33m"
    set "CYAN=[36m"
    set "BOLD=[1m"
    set "RESET=[0m"
)

echo.
echo %BOLD%%CYAN%══════════════════════════════════════════════════%RESET%
echo %BOLD%%CYAN%  EmbedSim — build all Cython extensions (Windows)%RESET%
echo %BOLD%%CYAN%══════════════════════════════════════════════════%RESET%
echo.
echo   Script dir : %SCRIPT_DIR%
if "%CLEAN_ONLY%"=="1" (echo   Mode       : clean only) else (echo   Mode       : full build)
echo.

if "%CLEAN_ONLY%"=="1" goto :clean_only

REM ---------------------------------------------------------------------------
REM Verify each sub-script exists before starting
REM ---------------------------------------------------------------------------
set "TARGETS=coordinate_transform motor_utility_blocks smc_controller svpwm dfc_controller"
set "MISSING=0"
for %%t in (%TARGETS%) do (
    if not exist "%SCRIPT_DIR%\build_%%t.bat" (
        echo %RED%✘ Missing build script: build_%%t.bat%RESET%
        set "MISSING=1"
    )
)

if "%MISSING%"=="1" (
    echo.
    echo %RED%ERROR: One or more build scripts missing.%RESET%
    echo Ensure all build_*.bat scripts are present in c_src\ before running build_all.bat.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Build each target in order
REM ---------------------------------------------------------------------------
set "PASSED="
set "FAILED="

for %%t in (%TARGETS%) do (
    echo.
    echo %BOLD%%CYAN%══════════════════════════════════════════════════%RESET%
    echo %BOLD%%CYAN%  Building: %%t%RESET%
    echo %BOLD%%CYAN%══════════════════════════════════════════════════%RESET%
    echo.

    call "%SCRIPT_DIR%\build_%%t.bat"

    if errorlevel 1 (
        echo.
        echo %RED%✘ %%t — build FAILED%RESET%
        set "FAILED=!FAILED! %%t"
    ) else (
        echo.
        echo %GREEN%✔ %%t — built successfully%RESET%
        set "PASSED=!PASSED! %%t"
    )
)

goto :summary

:clean_only
echo Cleaning all build artefacts...
if exist build rmdir /s /q build
for /r "%SCRIPT_DIR%" %%f in (*.pyd *.so *.c *.html) do (
    if /i "%%~nxf" neq "setup_coordinate_transform.py" (
        if /i "%%~nxf" neq "setup_motor_utility_blocks.py" (
            if /i "%%~nxf" neq "setup_smc_controller.py" (
                if /i "%%~nxf" neq "setup_svpwm.py" (
                    if /i "%%~nxf" neq "setup_dfc_controller.py" (
                        del /f /q "%%f" 2>nul
                    )
                )
            )
        )
    )
)

REM Remove promoted copies from parent directory
set "PARENT_DIR=%SCRIPT_DIR%\.."
if exist "%PARENT_DIR%\*.pyd" del /f /q "%PARENT_DIR%\*.pyd" 2>nul
if exist "%PARENT_DIR%\*.so" del /f /q "%PARENT_DIR%\*.so" 2>nul

echo %GREEN%✔ All artefacts removed.%RESET%
goto :eof

:summary
echo.
echo %BOLD%%CYAN%══════════════════════════════════════════════════%RESET%
echo %BOLD%%CYAN%  Build summary%RESET%
echo %BOLD%%CYAN%══════════════════════════════════════════════════%RESET%
echo.

if defined PASSED (
    for %%p in (%PASSED%) do (
        echo   %GREEN%✔%%p%RESET%
    )
)

if defined FAILED (
    for %%f in (%FAILED%) do (
        echo   %RED%✘%%f%RESET%
    )
)

echo.

if defined FAILED (
    echo %RED%%BOLD%ERROR: Some targets failed to build.%RESET%
    echo Re-run the individual build_*.bat script to see the full error.
    echo.
    pause
    exit /b 1
) else (
    echo %GREEN%%BOLD%All targets built successfully!%RESET%
    echo.
    echo Quick import check:
    echo   python -c "from fs_electrical_machines.coordinate_transform_wrapper import EmbedSimCoordinateTransform; from fs_electrical_machines.smc_controller_wrapper import SMCControllerWrapper; from fs_electrical_machines.svpwm_wrapper import EmbedSimSVPWM; from motor_utility_blocks import SpeedRampBlock; from fs_electrical_machines.dfc_controller_wrapper import DFCControllerWrapper; print('All imports OK')"
    echo.
    pause
    exit /b 0
)