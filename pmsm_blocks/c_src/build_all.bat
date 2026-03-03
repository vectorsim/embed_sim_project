@echo off
REM =============================================================================
REM build_all.bat
REM =============================================================================
REM Compile all pmsm_blocks Cython extensions on Windows.
REM
REM Run from the pmsm_blocks\c_src directory:
REM   cd pmsm_blocks\c_src
REM   build_all.bat
REM =============================================================================

echo ============================================================
echo  Building pmsm_blocks C extensions (Windows)
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/3]  pmsm_motor_wrapper
python setup_pmsm_motor.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  pmsm_motor_wrapper compiled

echo.
echo [2/3]  transforms_wrapper
python setup_transforms.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  transforms_wrapper compiled

echo.
echo [3/3]  pi_controller_wrapper
python setup_pi_controller.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  pi_controller_wrapper compiled

echo.
echo Copying .pyd files to parent package directory...
for /r . %%f in (*.pyd) do (
    copy "%%f" ".." > nul
)

echo.
echo ============================================================
echo  All extensions built successfully.
echo ============================================================
goto :eof

:error
echo.
echo ERROR: Build failed. Check output above.
exit /b 1
