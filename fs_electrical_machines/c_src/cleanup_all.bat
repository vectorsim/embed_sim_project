@echo off
REM =============================================================================
REM cleanup_all.bat  —  Master cleanup script
REM =============================================================================
REM Calls all individual cleanup scripts for:
REM   - DFC Controller
REM   - SVPWM
REM   - SMC Controller
REM   - Motor Utility Blocks
REM   - Coordinate Transform
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  MASTER CLEANUP - Removing all build artifacts
echo ============================================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Counter for tracking failures
set "FAILED=0"
set "TOTAL=5"

REM -----------------------------------------------------------------------------
REM 1. Clean DFC Controller
REM -----------------------------------------------------------------------------
echo [1/%TOTAL%] Cleaning DFC Controller...
echo ----------------------------------------
if exist "%~dp0cleanup_dfc_controller.bat" (
    call "%~dp0cleanup_dfc_controller.bat"
    if errorlevel 1 (
        echo [WARNING] DFC Controller cleanup reported errors
        set /a FAILED+=1
    )
) else (
    echo [ERROR] cleanup_dfc_controller.bat not found!
    set /a FAILED+=1
)
echo.

REM -----------------------------------------------------------------------------
REM 2. Clean SVPWM
REM -----------------------------------------------------------------------------
echo [2/%TOTAL%] Cleaning SVPWM...
echo ----------------------------------------
if exist "%~dp0cleanup_svpwm.bat" (
    call "%~dp0cleanup_svpwm.bat"
    if errorlevel 1 (
        echo [WARNING] SVPWM cleanup reported errors
        set /a FAILED+=1
    )
) else (
    echo [ERROR] cleanup_svpwm.bat not found!
    set /a FAILED+=1
)
echo.

REM -----------------------------------------------------------------------------
REM 3. Clean SMC Controller
REM -----------------------------------------------------------------------------
echo [3/%TOTAL%] Cleaning SMC Controller...
echo ----------------------------------------
if exist "%~dp0cleanup_smc_controller.bat" (
    call "%~dp0cleanup_smc_controller.bat"
    if errorlevel 1 (
        echo [WARNING] SMC Controller cleanup reported errors
        set /a FAILED+=1
    )
) else (
    echo [ERROR] cleanup_smc_controller.bat not found!
    set /a FAILED+=1
)
echo.

REM -----------------------------------------------------------------------------
REM 4. Clean Motor Utility Blocks
REM -----------------------------------------------------------------------------
echo [4/%TOTAL%] Cleaning Motor Utility Blocks...
echo ----------------------------------------
if exist "%~dp0cleanup_motor_utility_blocks.bat" (
    call "%~dp0cleanup_motor_utility_blocks.bat"
    if errorlevel 1 (
        echo [WARNING] Motor Utility Blocks cleanup reported errors
        set /a FAILED+=1
    )
) else (
    echo [ERROR] cleanup_motor_utility_blocks.bat not found!
    set /a FAILED+=1
)
echo.

REM -----------------------------------------------------------------------------
REM 5. Clean Coordinate Transform
REM -----------------------------------------------------------------------------
echo [5/%TOTAL%] Cleaning Coordinate Transform...
echo ----------------------------------------
if exist "%~dp0cleanup_coordinate_transform.bat" (
    call "%~dp0cleanup_coordinate_transform.bat"
    if errorlevel 1 (
        echo [WARNING] Coordinate Transform cleanup reported errors
        set /a FAILED+=1
    )
) else (
    echo [ERROR] cleanup_coordinate_transform.bat not found!
    set /a FAILED+=1
)
echo.

REM -----------------------------------------------------------------------------
REM Summary
REM -----------------------------------------------------------------------------
echo ============================================================
echo  MASTER CLEANUP COMPLETE
echo ============================================================
echo.
if %FAILED% EQU 0 (
    echo Status: All %TOTAL% cleanups completed successfully!
    echo.
    echo All build artifacts have been removed from:
    echo   - DFC Controller
    echo   - SVPWM
    echo   - SMC Controller
    echo   - Motor Utility Blocks
    echo   - Coordinate Transform
) else (
    echo Status: %FAILED% of %TOTAL% cleanups had issues.
    echo Please check the output above for details.
)
echo.
echo ============================================================
pause