@echo off
REM =============================================================================
REM cleanup_svpwm.bat  —  foc_generator\c_src\
REM =============================================================================
REM Removes all build artefacts produced by build_svpwm.bat.
REM
REM DELETED:
REM   build\                      Cython compile output
REM   svpwm_wrapper.c             Cython-generated C transpile
REM   svpwm_wrapper.html          Cython annotation file
REM   svpwm_wrapper*.pyd          Versioned .pyd in c_src\
REM   svpwm_wrapper.pyd           Plain copy in c_src\
REM   ..\svpwm_wrapper.pyd        Promoted copy in foc_generator\
REM
REM KEPT (never touched):
REM   svpwm.c / svpwm.h           Hand-written SVPWM source
REM   Sys_Types.h                 Shared type header
REM   svpwm_wrapper.pyx           Cython wrapper source
REM   setup_svpwm.py              Build script
REM   build_svpwm.bat             Build script
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  SVPWM -- Clean-up
echo ============================================================

cd /d "%~dp0"

REM Resolve parent (foc_generator\)
for %%i in ("%CD%") do set "PARENT_DIR=%%~dpi"
set "PARENT_DIR=%PARENT_DIR:~0,-1%"

echo  c_src : %CD%
echo  parent: %PARENT_DIR%
echo.

REM ── 1. build\ directory ──────────────────────────────────────────────────────
echo [1/5] Removing build\ directory...
if exist build (
    rmdir /s /q build
    echo        build\ removed
) else (
    echo        (nothing to do)
)

REM ── 2. Cython-generated wrapper .c ───────────────────────────────────────────
echo.
echo [2/5] Removing Cython-generated svpwm_wrapper.c...
if exist svpwm_wrapper.c (
    del /f /q svpwm_wrapper.c
    echo        svpwm_wrapper.c removed
) else (
    echo        (nothing to do)
)

REM ── 3. Cython HTML annotation ─────────────────────────────────────────────────
echo.
echo [3/5] Removing Cython HTML annotation file...
if exist svpwm_wrapper.html (
    del /f /q svpwm_wrapper.html
    echo        svpwm_wrapper.html removed
) else (
    echo        (nothing to do)
)

REM ── 4. .pyd files in c_src\ ──────────────────────────────────────────────────
echo.
echo [4/5] Removing .pyd files from c_src\...
set "FOUND=0"
for %%f in (svpwm_wrapper*.pyd) do (
    del /f /q "%%f"
    echo        %%~nxf removed
    set "FOUND=1"
)
if !FOUND!==0 echo        (nothing to do)

REM ── 5. Promoted copy in foc_generator\ ───────────────────────────────────────
echo.
echo [5/5] Removing promoted copy from parent directory...
if exist "%PARENT_DIR%\svpwm_wrapper.pyd" (
    del /f /q "%PARENT_DIR%\svpwm_wrapper.pyd"
    echo        %PARENT_DIR%\svpwm_wrapper.pyd removed
) else (
    echo        (nothing to do)
)

echo.
echo ============================================================
echo  SVPWM clean complete
echo ============================================================
echo.
