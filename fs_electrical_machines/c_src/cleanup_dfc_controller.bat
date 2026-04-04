@echo off
REM =============================================================================
REM cleanup_dfc_controller.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Clean DFC controller build artifacts.
REM =============================================================================

echo ============================================================
echo  Cleaning DFC Controller build artifacts
echo ============================================================

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
cd /d "%SCRIPT_DIR%"
set PARENT_DIR=%SCRIPT_DIR%\..

echo.
echo Removing build directory...
if exist build rmdir /s /q build

echo Removing compiled extensions...
if exist dfc_controller_wrapper*.so del /q dfc_controller_wrapper*.so
if exist dfc_controller_wrapper*.pyd del /q dfc_controller_wrapper*.pyd
if exist dfc_controller_wrapper*.c del /q dfc_controller_wrapper*.c

echo Removing from parent directory...
if exist "%PARENT_DIR%\dfc_controller_wrapper.so" del /q "%PARENT_DIR%\dfc_controller_wrapper.so"
if exist "%PARENT_DIR%\dfc_controller_wrapper.pyd" del /q "%PARENT_DIR%\dfc_controller_wrapper.pyd"

echo Removing temp directories...
for /d %%d in (dfc_controller_wrapper_*) do if exist %%d rmdir /s /q %%d

echo.
echo ============================================================
echo  Cleanup complete!
echo ============================================================