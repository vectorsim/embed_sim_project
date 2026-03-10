@echo off
REM =============================================================================
REM cleanup.bat  --  C:\EmbedSimProject\  (PROJECT ROOT)
REM =============================================================================
REM Removes all Cython build artefacts from electrical_blocks\c_src\:
REM   - build\  directory (compiled .obj, .pyd, .lib, .exp, .html)
REM   - generated .c files from .pyx (svpwm_wrapper.c, smo_wrapper.c, etc.)
REM   - generated .html annotation files
REM
REM Does NOT delete:
REM   - hand-written .c / .h source files
REM   - .pyx wrapper sources
REM   - .pyd files already copied to electrical_blocks\
REM   - setup_*.py files
REM =============================================================================

echo.
echo ============================================================
echo   EmbedSim -- Clean Build Artefacts
echo ============================================================

pushd "%~dp0electrical_blocks\c_src"
if errorlevel 1 (
    echo ERROR: Could not enter electrical_blocks\c_src
    exit /b 1
)
echo   Working dir: %CD%

echo.
echo   Removing build\ directory...
if exist build\ (
    rmdir /s /q build\
    echo        Removed: build\
) else (
    echo        Nothing to remove: build\ not found
)

echo.
echo   Removing generated Cython .c files...
for %%f in (coordinate_transform_wrapper.c smc_wrapper.c speed_pi_wrapper.c svpwm_wrapper.c smo_wrapper.c) do (
    if exist "%%f" (del "%%f" & echo        Removed: %%f) else (echo        Not found: %%f)
)

echo.
echo   Removing generated .html annotation files...
for %%f in (*.html) do (
    del "%%f" & echo        Removed: %%f
)

popd
echo.
echo ============================================================
echo   Cleanup complete.
echo ============================================================
