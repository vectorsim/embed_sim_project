@echo off
REM =============================================================================
REM build_dfc_controller.bat  —  pmsm\c_src
REM =============================================================================
REM Compile dfc_controller_wrapper Cython extension on Windows.
REM Output .pyd is copied to pmsm\ for easy importing.
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  Building DFC (Differential Flatness) Controller C extension (Windows)
echo ============================================================

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
cd /d "%SCRIPT_DIR%"
set PARENT_DIR=%SCRIPT_DIR%\..
set PROJECT_ROOT=%PARENT_DIR%\..

echo.
echo [1/5] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dfc_controller_wrapper*.pyd del /q dfc_controller_wrapper*.pyd
if exist dfc_controller_wrapper*.c del /q dfc_controller_wrapper*.c
if exist "%PARENT_DIR%\dfc_controller_wrapper*.pyd" del /q "%PARENT_DIR%\dfc_controller_wrapper*.pyd"
echo        Clean complete

echo.
echo [2/5] Setting up Python environment...

set PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe
if exist "%PYTHON%" (
    echo        Using .venv Python: %PYTHON%
) else (
    set PYTHON=python
    echo        .venv not found — using system Python
)

%PYTHON% --version

echo        Checking dependencies...
%PYTHON% -c "import setuptools" 2>nul || (
    echo        setuptools not found. Installing...
    %PYTHON% -m pip install --upgrade pip setuptools || exit /b 1
)
echo        setuptools OK

%PYTHON% -c "import Cython" 2>nul || (
    echo        Cython not found. Installing...
    %PYTHON% -m pip install --upgrade pip cython || exit /b 1
)
echo        Cython OK

%PYTHON% -c "import numpy" 2>nul || (
    echo        NumPy not found. Installing...
    %PYTHON% -m pip install numpy || exit /b 1
)
echo        NumPy OK

echo.
echo [3/5] Checking source files...
set MISSING=0
for %%s in (dfc_controller_wrapper.pyx embed_sim_dfc_controller.c embed_sim_coordinate_transform.c embed_sim_sv_pwm.c embed_sim_matrix.c) do (
    if exist "%%s" (
        echo        Found: %%s
    ) else (
        echo        ERROR: Missing %%s
        set MISSING=1
    )
)
if !MISSING! NEQ 0 (
    echo        ERROR: Missing source files. Aborting.
    exit /b 1
)

echo.
echo [4/5] Building dfc_controller_wrapper...
%PYTHON% setup_dfc_controller.py build_ext --inplace
if errorlevel 1 (
    echo        ERROR: Build failed
    exit /b 1
)
echo        OK - dfc_controller_wrapper compiled

echo.
echo [5/5] Copying .pyd to parent directory...
set PYD_FOUND=0
for %%f in (dfc_controller_wrapper*.pyd) do (
    set PYD_FOUND=1
    echo        Found: %%f
    copy /y "%%f" "%PARENT_DIR%\dfc_controller_wrapper.pyd" >nul
    if errorlevel 1 (
        echo        ERROR: copy failed — check permissions on %PARENT_DIR%
        exit /b 1
    )
    echo        Copied to %PARENT_DIR%\dfc_controller_wrapper.pyd
)

if "%PYD_FOUND%"=="0" (
    echo.
    echo ============================================================
    echo  WARNING: Build succeeded but no .pyd found in %SCRIPT_DIR%
    echo ============================================================
    echo.
    echo Run:  dir /s dfc_controller_wrapper*.pyd
    echo to locate the output and copy manually.
    exit /b 1
)

echo.
echo ============================================================
echo  DFC Controller built successfully!
echo ============================================================
echo.
echo    c_src\                    : dfc_controller_wrapper*.pyd (ABI-tagged)
echo    pmsm\   : dfc_controller_wrapper.pyd  (plain name)
echo.
echo Import with:
echo    from dfc_controller_wrapper import DFCControllerWrapper   (pmsm/ on sys.path)
echo.