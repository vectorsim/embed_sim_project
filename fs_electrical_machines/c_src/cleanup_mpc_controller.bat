@echo off
REM =============================================================================
REM cleanup_mpc_controller.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Clean up build artifacts from mpc_controller_wrapper compilation.
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  Cleaning MPC Controller build artifacts
echo ============================================================

cd /d "%~dp0"

REM Get parent directory (fs_electrical_machines/)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

echo.
echo Removing build directory...
if exist build rmdir /s /q build > nul 2>&1

echo Removing .pyd files...
if exist mpc_controller_wrapper*.pyd del /f /q mpc_controller_wrapper*.pyd > nul 2>&1

echo Removing from parent directory...
if exist "%PARENT_DIR%\mpc_controller_wrapper*.pyd" del /f /q "%PARENT_DIR%\mpc_controller_wrapper*.pyd" > nul 2>&1
if exist "%PARENT_DIR%\mpc_controller_wrapper.so" del /f /q "%PARENT_DIR%\mpc_controller_wrapper.so" > nul 2>&1

echo Removing .c files from Cython...
if exist mpc_controller_wrapper.c del /f /q mpc_controller_wrapper.c > nul 2>&1
if exist mpc_controller_wrapper.html del /f /q mpc_controller_wrapper.html > nul 2>&1

echo.
echo ============================================================
echo  Clean complete!
echo ============================================================