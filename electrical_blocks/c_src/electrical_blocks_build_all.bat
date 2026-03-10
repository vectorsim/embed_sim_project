@echo off
REM =============================================================================
REM build_all.bat  —  electrical_blocks\c_src
REM =============================================================================
REM Compile all electrical_blocks Cython extensions on Windows.
REM
REM Run from the electrical_blocks\c_src directory:
REM   cd electrical_blocks\c_src
REM   build_all.bat
REM
REM Output (.pyd files copied to parent package dir):
REM   coordinate_transform_wrapper.pyd
REM   smc_wrapper.pyd
REM   speed_pi_wrapper.pyd
REM   svpwm_wrapper.pyd
REM   smo_wrapper.pyd
REM =============================================================================

echo ============================================================
echo  Building electrical_blocks C extensions  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Build output dir — CPython 3.12 x64
set BUILD_LIB=build\lib.win-amd64-cpython-312

echo.
echo [1/5]  coordinate_transform_wrapper
python setup_coordinate_transform.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  coordinate_transform_wrapper compiled

echo.
echo [2/5]  smc_wrapper  (Sliding Mode Controller)
python setup_smc.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  smc_wrapper compiled

echo.
echo [3/5]  speed_pi_wrapper  (Speed PI Controller)
python setup_speed_pi.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  speed_pi_wrapper compiled

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
for %%f in ("%BUILD_LIB%\coordinate_transform_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)
for %%f in ("%BUILD_LIB%\smc_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf
)
for %%f in ("%BUILD_LIB%\speed_pi_wrapper*.pyd") do (
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
