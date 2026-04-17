@echo off
REM =============================================================================
REM build_mpc_controller.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Compile the MPC FOC controller Cython extension.
REM
REM Source files compiled into mpc_controller_wrapper.cp312-win_amd64.pyd:
REM   mpc_controller_wrapper.pyx          (Cython wrapper)
REM   embed_sim_mpc_controller.c          (3-state MPC, analytical solver)
REM   embed_sim_coordinate_transform.c    (Clarke / Park / InvPark)
REM   embed_sim_matrix.c                  (MatrixFloat helpers)
REM
REM Output locations (mirrors dfc / smc pattern):
REM   c_src\                  mpc_controller_wrapper.cp312-win_amd64.pyd  (ABI-tagged)
REM   fs_electrical_machines\ mpc_controller_wrapper.pyd                  (plain name)
REM
REM Usage:
REM   cd fs_electrical_machines\c_src
REM   build_mpc_controller.bat
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

set "PARENT_DIR=%SCRIPT_DIR%\.."
set "EXT_NAME=mpc_controller_wrapper"
set "SETUP_SCRIPT=setup_mpc_controller.py"
set "VERSION=2.0.0"

echo.
echo ============================================================
echo  MPC Controller Cython extension build
echo  Version: %VERSION%
echo ============================================================
echo.

REM ---------------------------------------------------------------------------
REM [1/4] Check required source files exist
REM ---------------------------------------------------------------------------
echo [1/4] Checking source files...

set "MISSING=0"
for %%f in (
    "%SCRIPT_DIR%\mpc_controller_wrapper.pyx"
    "%SCRIPT_DIR%\embed_sim_mpc_controller.c"
    "%SCRIPT_DIR%\embed_sim_mpc_controller.h"
    "%SCRIPT_DIR%\embed_sim_coordinate_transform.c"
    "%SCRIPT_DIR%\embed_sim_matrix.c"
    "%SCRIPT_DIR%\%SETUP_SCRIPT%"
) do (
    if not exist %%f (
        echo        MISSING: %%f
        set "MISSING=1"
    ) else (
        echo        OK: %%~nxf
    )
)

if "%MISSING%"=="1" (
    echo.
    echo ERROR: One or more required files are missing.
    echo        Ensure all source files are present in c_src\ before building.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM [2/4] Remove stale .pyd binaries so the linker picks up the new build
REM ---------------------------------------------------------------------------
echo.
echo [2/4] Removing stale binaries...

for %%f in ("%SCRIPT_DIR%\%EXT_NAME%*.pyd" "%PARENT_DIR%\%EXT_NAME%*.pyd") do (
    if exist %%f (
        del /f /q %%f
        echo        Deleted: %%f
    )
)

REM ---------------------------------------------------------------------------
REM [3/4] Transpile .pyx → .c and compile the extension
REM ---------------------------------------------------------------------------
echo.
echo [3/4] Building Cython extension...
echo.

echo ============================================================
echo  Build configuration complete
echo  Extension version: %VERSION%
echo ============================================================

python "%SETUP_SCRIPT%" build_ext --inplace
if errorlevel 1 (
    echo.
    echo ERROR: Build failed.  Re-run with full output:
    echo   python %SETUP_SCRIPT% build_ext --inplace
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM [4/4] Promote the ABI-tagged .pyd to the parent directory as plain name
REM ---------------------------------------------------------------------------
echo.
echo [4/4] Copying .pyd to parent directory...

set "FOUND=0"
for %%f in ("%SCRIPT_DIR%\%EXT_NAME%*.pyd") do (
    if exist "%%f" (
        echo        Found: %%~nxf
        copy /y "%%f" "%PARENT_DIR%\%EXT_NAME%.pyd" >nul
        echo        Copied to %PARENT_DIR%\%EXT_NAME%.pyd
        set "FOUND=1"
    )
)

if "%FOUND%"=="0" (
    echo.
    echo ERROR: No .pyd found after build — check compiler output above.
    exit /b 1
)

echo.
echo ============================================================
echo  MPC Controller built successfully
echo ============================================================
echo.
echo    c_src\                    : %EXT_NAME%*.pyd  (ABI-tagged)
echo    fs_electrical_machines\   : %EXT_NAME%.pyd   (plain name)
echo.
echo Import with:
echo    from fs_electrical_machines.mpc_controller_wrapper import MPCControllerWrapper
echo.

exit /b 0
