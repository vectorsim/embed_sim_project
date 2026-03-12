@echo off
REM =============================================================================
REM build_pi_buck.bat  —  buck_converter\c_src
REM =============================================================================
REM Compile PI Buck Controller Cython extension on Windows.
REM Output files are copied to buck_converter/ for easy importing.
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  Building PI Buck Controller C extension  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Get parent directory (buck_converter/)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

REM Build output dir — CPython 3.12 x64
set BUILD_LIB=build\lib.win-amd64-cpython-312

echo.
echo [1/4] Cleaning previous builds...
if exist build rmdir /s /q build > nul 2>&1
if exist pi_buck_wrapper.pyd del /f /q pi_buck_wrapper.pyd > nul 2>&1
if exist "%PARENT_DIR%\pi_buck_wrapper.pyd" del /f /q "%PARENT_DIR%\pi_buck_wrapper.pyd" > nul 2>&1
echo        Clean complete

echo.
echo [2/4] Building pi_buck_wrapper (PI Buck Controller)...
C:\EmbedSimProject\.venv\Scripts\python.exe setup_pi_buck.py build_ext --inplace
if errorlevel 1 goto :error
echo        ✅ OK - pi_buck_wrapper compiled

echo.
echo [3/4] Copying .pyd files...
set "PYD_FOUND=0"

REM Check build directory first
if exist "%BUILD_LIB%\pi_buck_wrapper*.pyd" (
    echo        Found in build directory:
    for %%f in ("%BUILD_LIB%\pi_buck_wrapper*.pyd") do (
        set "PYD_FOUND=1"
        echo          - %%~nxf

        REM Copy to current directory
        copy "%%f" "." > nul
        echo          ✅ Copied to current directory

        REM Copy to parent directory with simple name
        copy "%%f" "%PARENT_DIR%\pi_buck_wrapper.pyd" > nul
        echo          ✅ Copied to parent as pi_buck_wrapper.pyd
    )
) else (
    echo        No .pyd files found in %BUILD_LIB%
)

REM If not found in build, check current directory
if !PYD_FOUND!==0 (
    if exist pi_buck_wrapper*.pyd (
        echo        Found in current directory:
        for %%f in (pi_buck_wrapper*.pyd) do (
            set "PYD_FOUND=1"
            echo          - %%~nxf

            REM Copy to parent with simple name
            copy "%%f" "%PARENT_DIR%\pi_buck_wrapper.pyd" > nul
            echo          ✅ Copied to parent as pi_buck_wrapper.pyd
        )
    )
)

if !PYD_FOUND!==0 (
    echo        ⚠️  No .pyd files found anywhere!
    goto :warning
)

echo.
echo [4/4] Creating simple module name in current directory...
if exist pi_buck_wrapper*.pyd (
    copy /y pi_buck_wrapper*.pyd pi_buck_wrapper.pyd > nul
    echo        ✅ Created pi_buck_wrapper.pyd in current directory
) else (
    echo        ⚠️  No source .pyd file found
)

echo.
echo ============================================================
echo  🎉 PI Buck Controller built successfully!
echo ============================================================
echo.
echo Files created:
echo   📁 Current directory (%CD%):
for %%f in (pi_buck_wrapper*.pyd) do (
    echo     📄 %%~nxf
)
echo.
echo   📁 Parent directory (%PARENT_DIR%):
if exist "%PARENT_DIR%\pi_buck_wrapper.pyd" (
    echo     📄 pi_buck_wrapper.pyd
)
echo.
echo You can now import in Python:
echo   import pi_buck_wrapper
echo   ctrl = pi_buck_wrapper.PI_BuckWrapper()
echo.
goto :eof

:error
echo.
echo ============================================================
echo  ❌ ERROR: Build failed!
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
echo  ⚠️  WARNING: Build completed but no .pyd files found
echo ============================================================
echo.
echo The compilation may have succeeded but the .pyd file
echo was not found in expected locations.
echo.
echo Check the build output above for errors.
echo.
pause
exit /b 1