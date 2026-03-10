@echo off
REM =============================================================================
REM build_all.bat  —  pmsm_blocks\c_src
REM =============================================================================
REM Compile all pmsm_blocks Cython extensions on Windows.
REM
REM Run from the pmsm_blocks\c_src directory:
REM   cd pmsm_blocks\c_src
REM   build_all.bat
REM
REM Output (.pyd files copied to parent package dir):
REM   pi_controller_wrapper.pyd
REM   pmsm_motor_wrapper.pyd
REM   transforms_wrapper.pyd
REM   svpwm_wrapper.pyd
REM   smo_wrapper.pyd
REM =============================================================================

echo ============================================================
echo  Building pmsm_blocks C extensions  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Build output dir — CPython 3.12 x64
set BUILD_LIB=build\lib.win-amd64-cpython-312

echo.
echo [1/5]  pi_controller_wrapper  (PI Controller)
python setup_pi_controller.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  pi_controller_wrapper compiled

echo.
echo [2/5]  pmsm_motor_wrapper  (PMSM Motor model)
python setup_pmsm_motor.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  pmsm_motor_wrapper compiled

echo.
echo [3/5]  transforms_wrapper  (Clarke / Park / Inverse transforms)
python setup_transforms.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  transforms_wrapper compiled

echo.
echo [4/5]  svpwm_wrapper  (Space Vector PWM Modulator)
python setup_svpwm.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  svpwm_wrapper compiled

echo.
echo [5/5]  smo_wrapper  (Sliding Mode Observer)
python setup_smo.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  smo_wrapper compiled

echo.
echo Copying .pyd files to parent package directory...
for %%f in ("%BUILD_LIB%\pi_controller_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)
for %%f in ("%BUILD_LIB%\pmsm_motor_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)
for %%f in ("%BUILD_LIB%\transforms_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)
for %%f in ("%BUILD_LIB%\svpwm_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)
for %%f in ("%BUILD_LIB%\smo_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)

echo.
echo ============================================================
echo  All extensions built successfully.
echo ============================================================
goto :eof

:error
echo.
echo ERROR: Build failed at step above. See output above.
exit /b 1
