@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Building EmbedSim Control Wrapper (Sensor-Based) for Windows
echo ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
pushd "%SCRIPT_DIR%"
set "PARENT_DIR=%SCRIPT_DIR%\.."
set "PROJECT_ROOT=%PARENT_DIR%\.."

echo.
echo [1/5] Cleaning previous builds...
if exist build rmdir /s /q build
del /f /q embedsim_control_wrapper*.pyd 2>nul
del /f /q embedsim_control_wrapper*.c 2>nul
if exist "%PARENT_DIR%\embedsim_control_wrapper.pyd" del /f /q "%PARENT_DIR%\embedsim_control_wrapper.pyd"
echo        Clean complete

echo.
echo [2/5] Setting up Python environment...

set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
    echo        Using .venv Python: %VENV_PYTHON%
) else (
    set "PYTHON=python"
    echo        .venv not found -- using system Python
)

%PYTHON% --version

echo        Checking dependencies...
%PYTHON% -c "import setuptools" >nul 2>&1
if errorlevel 1 (
    echo        setuptools not found. Installing...
    %PYTHON% -m pip install --upgrade pip setuptools
    if errorlevel 1 (
        echo        ERROR: Failed to install setuptools
        exit /b 1
    )
) else (
    echo        setuptools OK
)

%PYTHON% -c "import Cython" >nul 2>&1
if errorlevel 1 (
    echo        Cython not found. Installing...
    %PYTHON% -m pip install --upgrade pip cython
    if errorlevel 1 (
        echo        ERROR: Failed to install Cython
        exit /b 1
    )
) else (
    echo        Cython OK
)

%PYTHON% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo        NumPy not found. Installing...
    %PYTHON% -m pip install numpy
    if errorlevel 1 (
        echo        ERROR: Failed to install NumPy
        exit /b 1
    )
) else (
    echo        NumPy OK
)

echo.
echo [3/5] Checking source files...
set MISSING=0
for %%f in (
    embedsim_control_wrapper.pyx
    embed_sim_control.c
    embed_sim_dfc_controller.c
    embed_sim_coordinate_transform.c
    embed_sim_sv_pwm.c
    embed_sim_matrix.c
) do (
    if not exist "%%f" (
        echo        ERROR: Missing %%f
        set MISSING=1
    ) else (
        echo        Found: %%f
    )
)

if %MISSING% neq 0 (
    echo        ERROR: Missing source files. Aborting.
    exit /b 1
)

echo.
echo [4/5] Building embedsim_control_wrapper...
%PYTHON% embedsim_control_wrapper.py build_ext --inplace
if errorlevel 1 (
    echo        ERROR: Build failed
    exit /b 1
)
echo        OK - embedsim_control_wrapper compiled

echo.
echo [5/5] Copying .pyd to parent directory...
set PYD_FOUND=0
for %%f in (embedsim_control_wrapper*.pyd) do (
    set PYD_FOUND=1
    echo        Found: %%f
    copy /y "%%f" "%PARENT_DIR%\embedsim_control_wrapper.pyd"
    if errorlevel 1 (
        echo        ERROR: copy failed -- check permissions on %PARENT_DIR%
        exit /b 1
    )
    echo        Copied to %PARENT_DIR%\embedsim_control_wrapper.pyd
)

if %PYD_FOUND% equ 0 (
    echo.
    echo ============================================================
    echo  WARNING: Build succeeded but no .pyd found in %SCRIPT_DIR%
    echo ============================================================
    echo.
    echo Run:  dir /s embedsim_control_wrapper*.pyd
    echo to locate the output and copy manually.
    exit /b 1
)

echo.
echo ============================================================
echo  EmbedSim Control Wrapper built successfully!
echo ============================================================
echo.
echo   c_src\                    : embedsim_control_wrapper*.pyd (ABI-tagged)
echo   pmsm\  : embedsim_control_wrapper.pyd   (plain name)
echo.
echo Import with:
echo   from embedsim_control_wrapper import control_init, control_step
echo.

popd

echo.
echo Press any key to exit...
pause >nul
exit /b 0