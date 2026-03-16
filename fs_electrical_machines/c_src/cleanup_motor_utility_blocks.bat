@echo off
REM =============================================================================
REM cleanup_motor_utility_blocks.bat  —  fs_electrical_machines\c_src\
REM =============================================================================
REM Removes all build artefacts produced by build_motor_utility_blocks.bat.
REM
REM DELETED:
REM   build\                                  Cython compile output
REM   motor_utility_blocks_wrapper.c          Cython-generated C transpile
REM   motor_utility_blocks_wrapper.html       Cython annotation file
REM   motor_utility_blocks_wrapper*.pyd       Versioned .pyd in c_src\
REM   motor_utility_blocks_wrapper.pyd        Plain copy in c_src\
REM   ..\motor_utility_blocks_wrapper.pyd     Promoted copy in fs_electrical_machines\
REM
REM KEPT (never touched):
REM   motor_utility_blocks.c / .h             Hand-written C source
REM   Sys_Types.h                             Shared type header
REM   motor_utility_blocks_wrapper.pyx        Cython wrapper source
REM   setup_motor_utility_blocks.py           Build script
REM   build_motor_utility_blocks.bat          This build script
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  motor_utility_blocks -- Clean-up
echo ============================================================

cd /d "%~dp0"

REM Resolve parent (fs_electrical_machines\)
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
echo [2/5] Removing Cython-generated motor_utility_blocks_wrapper.c...
if exist motor_utility_blocks_wrapper.c (
    del /f /q motor_utility_blocks_wrapper.c
    echo        motor_utility_blocks_wrapper.c removed
) else (
    echo        (nothing to do)
)

REM ── 3. Cython HTML annotation ─────────────────────────────────────────────────
echo.
echo [3/5] Removing Cython HTML annotation file...
if exist motor_utility_blocks_wrapper.html (
    del /f /q motor_utility_blocks_wrapper.html
    echo        motor_utility_blocks_wrapper.html removed
) else (
    echo        (nothing to do)
)

REM ── 4. .pyd files in c_src\ ──────────────────────────────────────────────────
echo.
echo [4/5] Removing .pyd files from c_src\...
set "FOUND=0"
for %%f in (motor_utility_blocks_wrapper*.pyd) do (
    del /f /q "%%f"
    echo        %%~nxf removed
    set "FOUND=1"
)
if !FOUND!==0 echo        (nothing to do)

REM ── 5. Promoted copy in fs_electrical_machines\ ───────────────────────────────
echo.
echo [5/5] Removing promoted copy from parent directory...
if exist "%PARENT_DIR%\motor_utility_blocks_wrapper.pyd" (
    del /f /q "%PARENT_DIR%\motor_utility_blocks_wrapper.pyd"
    echo        %PARENT_DIR%\motor_utility_blocks_wrapper.pyd removed
) else (
    echo        (nothing to do)
)

echo.
echo ============================================================
echo  motor_utility_blocks clean complete
echo ============================================================
echo.
