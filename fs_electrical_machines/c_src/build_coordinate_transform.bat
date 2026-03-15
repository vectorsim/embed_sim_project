@echo@echo off
REM =============================================================================
REM build_coordinate_transform.bat  —  fs_electrical_machines\c_src
REM =============================================================================
REM Compile coordinate_transform_wrapper Cython extension on Windows.
REM Output files are copied to fs_electrical_machines/ for easy importing.
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  Building Coordinate Transform C extension  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Get parent directory (fs_electrical_machines/)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

REM Build output dir — CPython 3.12 x64
set BUILD_LIB=build\lib.win-amd64-cpython-312

echo.
echo [1/4] Cleaning previous builds...
if exist build rmdir /s /q build > nul 2>&1
if exist coordinate_transform_wrapper.pyd del /f /q coordinate_transform_wrapper.pyd > nul 2>&1
if exist "%PARENT_DIR%\coordinate_transform_wrapper.pyd" del /f /q "%PARENT_DIR%\coordinate_transform_wrapper.pyd" > nul 2>&1
echo        Clean complete

echo.
echo [2/4] Building coordinate_transform_wrapper...

REM Auto-detect Python: prefer .venv if present, otherwise use PATH python
set "VENV_PYTHON=C:\EmbedSimProject\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
    echo        Using .venv Python: %VENV_PYTHON%
) else (
    set "PYTHON=python"
    echo        .venv not found — using system/Anaconda Python
)

%PYTHON% setup_coordinate_transform.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK - coordinate_transform_wrapper compiled

echo.
echo [3/4] Copying .pyd files...
set "PYD_FOUND=0"

REM Check build directory first
if exist "%BUILD_LIB%\coordinate_transform_wrapper*.pyd" (
    echo        Found in build directory:
    for %%f in ("%BUILD_LIB%\coordinate_transform_wrapper*.pyd") do (
        set "PYD_FOUND=1"
        echo          - %%~nxf

        REM Copy to current directory
        copy "%%f" "." > nul
        echo          Copied to current directory

        REM Copy to parent directory with simple name
        copy "%%f" "%PARENT_DIR%\coordinate_transform_wrapper.pyd" > nul
        echo          Copied to parent as coordinate_transform_wrapper.pyd
    )
) else (
    echo        No .pyd files found in %BUILD_LIB%
)

REM If not found in build, check current directory
if !PYD_FOUND!==0 (
    if exist coordinate_transform_wrapper*.pyd (
        echo        Found in current directory:
        for %%f in (coordinate_transform_wrapper*.pyd) do (
            set "PYD_FOUND=1"
            echo          - %%~nxf

            REM Copy to parent with simple name
            copy "%%f" "%PARENT_DIR%\coordinate_transform_wrapper.pyd" > nul
            echo          Copied to parent as coordinate_transform_wrapper.pyd
        )
    )
)

if !PYD_FOUND!==0 (
    echo        WARNING: No .pyd files found anywhere!
    goto :warning
)

echo.
echo [4/4] Creating simple module name in current directory...
if exist coordinate_transform_wrapper*.pyd (
    copy /y coordinate_transform_wrapper*.pyd coordinate_transform_wrapper.pyd > nul
    echo        Created coordinate_transform_wrapper.pyd in current directory
) else (
    echo        WARNING: No source .pyd file found
)

echo.
echo ============================================================
echo  Coordinate Transform built successfully!
echo ============================================================
echo.
echo Files created:
echo   Current directory (%CD%):
for %%f in (coordinate_transform_wrapper*.pyd) do (
    echo     %%~nxf
)
echo.
echo   Parent directory (%PARENT_DIR%):
if exist "%PARENT_DIR%\coordinate_transform_wrapper.pyd" (
    echo     coordinate_transform_wrapper.pyd
)
echo.
echo You can now import in Python:
echo   from coordinate_transform_blocks import ClarkeTransformBlock
echo.
goto :eof

:error
echo.
echo ============================================================
echo  ERROR: Build failed!
echo ============================================================
echo.
echo Possible issues:
echo   1. Cython not installed (run: pip install cython)
echo   2. Missing dependencies
echo   3. Compiler errors in C code
echo.
pause
exit /b 1

:warning
echo.
echo ============================================================
echo  WARNING: Build completed but no .pyd files found
echo ============================================================
echo.
echo The compilation may have succeeded but the .pyd file
echo was not found in expected locations.
echo Check the build output above for errors.
echo.
pause
exit /b 1