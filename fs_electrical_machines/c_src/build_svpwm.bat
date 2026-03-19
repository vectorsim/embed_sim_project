@echo off
REM =============================================================================
REM build_svpwm.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Compile svpwm_wrapper Cython extension on Windows.
REM Output .pyd is copied to fs_electrical_machines\ for easy importing.
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  Building SVPWM C extension  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Get parent directory (fs_electrical_machines/)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

REM Get project root (embed_sim_project/)
for %%i in ("%PARENT_DIR%") do set "PROJECT_ROOT=%%~dpi"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo.
echo [1/4] Cleaning previous builds...
if exist build rmdir /s /q build > nul 2>&1
if exist svpwm_wrapper*.pyd del /f /q svpwm_wrapper*.pyd > nul 2>&1
if exist "%PARENT_DIR%\svpwm_wrapper*.pyd" del /f /q "%PARENT_DIR%\svpwm_wrapper*.pyd" > nul 2>&1
echo        Clean complete

echo.
echo [2/4] Setting up Python environment...

set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
    echo        Using .venv Python: %VENV_PYTHON%
) else (
    set "PYTHON=python"
    echo        .venv not found — using system Python
)

%PYTHON% --version

echo        Checking dependencies...
%PYTHON% -c "import Cython" > nul 2>&1
if errorlevel 1 (
    echo        Cython not found. Installing...
    %PYTHON% -m pip install --upgrade pip cython
    if errorlevel 1 ( echo        Failed to install Cython & goto :error )
) else ( echo        Cython OK )

%PYTHON% -c "import numpy" > nul 2>&1
if errorlevel 1 (
    echo        NumPy not found. Installing...
    %PYTHON% -m pip install numpy
    if errorlevel 1 ( echo        Failed to install NumPy & goto :error )
) else ( echo        NumPy OK )

echo.
echo [3/4] Building svpwm_wrapper...
%PYTHON% setup_svpwm.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK - svpwm_wrapper compiled

echo.
echo [4/4] Copying .pyd to parent directory...
set "PYD_FOUND=0"
for %%f in ("%CD%\svpwm_wrapper*.pyd") do (
    set "PYD_FOUND=1"
    echo        Found: %%~nxf
    copy /y "%%f" "%PARENT_DIR%\svpwm_wrapper.pyd" > nul
    if errorlevel 1 (
        echo        ERROR: copy failed — check permissions on %PARENT_DIR%
        goto :error
    )
    echo        Copied to %PARENT_DIR%\svpwm_wrapper.pyd
)
if "!PYD_FOUND!"=="0" goto :warning

echo.
echo ============================================================
echo  SVPWM built successfully!
echo ============================================================
echo.
echo   c_src\                   : svpwm_wrapper*.pyd  (ABI-tagged)
echo   fs_electrical_machines\  : svpwm_wrapper.pyd   (plain name)
echo.
echo Import with:
echo   from fs_electrical_machines.svpwm_wrapper import EmbedSimSVPWM
echo.
goto :eof

:error
echo.
echo ============================================================
echo  ERROR: Build failed!
echo ============================================================
echo.
echo Common causes:
echo   1. MSVC / Build Tools not on PATH  (run from a VS Developer prompt)
echo   2. Missing source files: svpwm.c, svpwm.h, Matrix.c, Matrix.h
echo   3. C compiler errors — check output above
echo.
pause
exit /b 1

:warning
echo.
echo ============================================================
echo  WARNING: Build succeeded but no .pyd found in %CD%
echo ============================================================
echo.
echo Run:  dir /s /b svpwm_wrapper*.pyd
echo to locate the output and copy manually.
echo.
pause
exit /b 1
