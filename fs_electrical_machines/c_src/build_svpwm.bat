@echo off
REM =============================================================================
REM build_svpwm.bat  —  foc_generator\c_src
REM =============================================================================
REM Compile svpwm_wrapper Cython extension on Windows.
REM Output .pyd is copied to foc_generator/ for easy importing.
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  Building SVPWM C extension  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Get parent directory (foc_generator/)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

REM Build output dir — CPython 3.12 x64
set BUILD_LIB=build\lib.win-amd64-cpython-312

echo.
echo [1/4] Cleaning previous builds...
if exist build rmdir /s /q build > nul 2>&1
if exist svpwm_wrapper.pyd del /f /q svpwm_wrapper.pyd > nul 2>&1
if exist "%PARENT_DIR%\svpwm_wrapper.pyd" del /f /q "%PARENT_DIR%\svpwm_wrapper.pyd" > nul 2>&1
echo        Clean complete

echo.
echo [2/4] Building svpwm_wrapper...

REM Auto-detect Python: prefer .venv if present, otherwise use PATH python
set "VENV_PYTHON=C:\EmbedSimProject\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
    echo        Using .venv Python: %VENV_PYTHON%
) else (
    set "PYTHON=python"
    echo        .venv not found — using system/Anaconda Python
)

%PYTHON% setup_svpwm.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK - svpwm_wrapper compiled

echo.
echo [3/4] Copying .pyd files...
set "PYD_FOUND=0"

if exist "%BUILD_LIB%\svpwm_wrapper*.pyd" (
    echo        Found in build directory:
    for %%f in ("%BUILD_LIB%\svpwm_wrapper*.pyd") do (
        set "PYD_FOUND=1"
        echo          - %%~nxf
        copy "%%f" "." > nul
        echo          Copied to current directory
        copy "%%f" "%PARENT_DIR%\svpwm_wrapper.pyd" > nul
        echo          Copied to parent as svpwm_wrapper.pyd
    )
) else (
    echo        No .pyd files found in %BUILD_LIB%
)

if !PYD_FOUND!==0 (
    if exist svpwm_wrapper*.pyd (
        for %%f in (svpwm_wrapper*.pyd) do (
            set "PYD_FOUND=1"
            echo          - %%~nxf
            copy "%%f" "%PARENT_DIR%\svpwm_wrapper.pyd" > nul
            echo          Copied to parent as svpwm_wrapper.pyd
        )
    )
)

if !PYD_FOUND!==0 (
    echo        WARNING: No .pyd files found anywhere!
    goto :warning
)

echo.
echo [4/4] Creating simple module name in current directory...
if exist svpwm_wrapper*.pyd (
    copy /y svpwm_wrapper*.pyd svpwm_wrapper.pyd > nul
    echo        Created svpwm_wrapper.pyd
)

echo.
echo ============================================================
echo  SVPWM built successfully!
echo ============================================================
echo.
echo Files created:
echo   Current directory (%CD%):
for %%f in (svpwm_wrapper*.pyd) do echo     %%~nxf
echo.
echo   Parent directory (%PARENT_DIR%):
if exist "%PARENT_DIR%\svpwm_wrapper.pyd" echo     svpwm_wrapper.pyd
echo.
echo You can now import in Python:
echo   from foc_generator.svpwm_block import SVPWMBlock
echo.
goto :eof

:error
echo.
echo ============================================================
echo  ERROR: Build failed
echo ============================================================
echo.
echo Possible issues:
echo   1. Cython not installed  (pip install cython)
echo   2. Missing svpwm.c / svpwm.h
echo   3. Compiler error — check output above
echo.
pause
exit /b 1

:warning
echo.
echo ============================================================
echo  WARNING: Build completed but no .pyd found
echo ============================================================
echo.
pause
exit /b 1
