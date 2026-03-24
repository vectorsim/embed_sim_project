@echo off
REM =============================================================================
REM cleanup_smc_controller.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Clean up build artifacts from smc_controller_wrapper compilation.
REM =============================================================================

echo ============================================================
echo  Cleaning SMC Controller build artifacts
echo ============================================================

cd /d "%~dp0"

REM Get parent directory (fs_electrical_machines/)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

echo.
echo Removing build directory...
if exist build rmdir /s /q build
echo Removing .pyd files...
if exist smc_controller_wrapper*.pyd del /f /q smc_controller_wrapper*.pyd
if exist "%PARENT_DIR%\smc_controller_wrapper*.pyd" del /f /q "%PARENT_DIR%\smc_controller_wrapper*.pyd"
echo Removing .c files from Cython...
if exist smc_controller_wrapper.c del /f /q smc_controller_wrapper.c
if exist smc_controller_wrapper.html del /f /q smc_controller_wrapper.html

echo.
echo ============================================================
echo  Clean complete!
echo ============================================================