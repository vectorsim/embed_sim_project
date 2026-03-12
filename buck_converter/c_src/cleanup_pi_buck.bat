@echo off
REM =============================================================================
REM cleanup_pi_buck.bat  —  buck_converter\c_src\
REM =============================================================================
REM Removes all build artefacts produced by build_pi_buck.bat.
REM
REM DELETED:
REM   build\                        Cython compile output
REM   pi_buck_wrapper.c             Cython-generated C transpile
REM   pi_buck_wrapper.html          Cython annotation file
REM   pi_buck_wrapper*.pyd          Versioned .pyd in c_src\
REM   pi_buck_wrapper.pyd           Plain copy in c_src\
REM   ..\pi_buck_wrapper.pyd        Promoted copy in buck_converter\
REM
REM KEPT (never touched):
REM   pi_buck_controller.c / .h     Hand-written controller source
REM   pi_buck_wrapper.pyx           Cython wrapper source
REM   setup_pi_buck.py              Build script
REM   build_pi_buck.bat             This script's sibling
REM   Sys_Types.h                   Shared type header
REM   pi_buck_wrapper_test.py       Unit test
REM   pi_buck_test.png              Test output image
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  PI Buck Controller — Clean-up
echo ============================================================

REM Move to c_src\ regardless of where the script is invoked from
cd /d "%~dp0"

REM Resolve parent (buck_converter\)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

echo  c_src : %CD%
echo  parent: %PARENT_DIR%
echo.

REM ── 1. build\ directory ──────────────────────────────────────────────────────
echo [1/5] Removing build\ directory...
if exist build (
    rmdir /s /q build
    echo        ✅ build\ removed
) else (
    echo        (nothing to do)
)

REM ── 2. Cython-generated wrapper .c ───────────────────────────────────────────
echo.
echo [2/5] Removing Cython-generated pi_buck_wrapper.c...
if exist pi_buck_wrapper.c (
    del /f /q pi_buck_wrapper.c
    echo        ✅ pi_buck_wrapper.c removed
) else (
    echo        (nothing to do)
)

REM ── 3. Cython HTML annotation ─────────────────────────────────────────────────
echo.
echo [3/5] Removing Cython HTML annotation file...
if exist pi_buck_wrapper.html (
    del /f /q pi_buck_wrapper.html
    echo        ✅ pi_buck_wrapper.html removed
) else (
    echo        (nothing to do)
)

REM ── 4. .pyd files in c_src\ ──────────────────────────────────────────────────
echo.
echo [4/5] Removing .pyd files from c_src\...
set "FOUND=0"
for %%f in (pi_buck_wrapper*.pyd) do (
    del /f /q "%%f"
    echo        ✅ %%~nxf removed
    set "FOUND=1"
)
if !FOUND!==0 echo        (nothing to do)

REM ── 5. Promoted copy in buck_converter\ ──────────────────────────────────────
echo.
echo [5/5] Removing promoted copy from parent directory...
if exist "%PARENT_DIR%\pi_buck_wrapper.pyd" (
    del /f /q "%PARENT_DIR%\pi_buck_wrapper.pyd"
    echo        ✅ %PARENT_DIR%\pi_buck_wrapper.pyd removed
) else (
    echo        (nothing to do)
)

echo.
echo ============================================================
echo  ✅ PI Buck Controller clean complete
echo ============================================================
echo.
