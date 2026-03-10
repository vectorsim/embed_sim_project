@echo off
REM =============================================================================
REM build_all.bat  --  C:\EmbedSimProject\  (PROJECT ROOT)
REM =============================================================================
REM Compiles all Cython extensions for electrical_blocks.
REM
REM Place at:  C:\EmbedSimProject\build_all.bat
REM Run from:  C:\EmbedSimProject\
REM
REM Output (.pyd files copied to electrical_blocks\):
REM   coordinate_transform_wrapper
REM   smc_wrapper
REM   speed_pi_wrapper
REM   svpwm_wrapper
REM   smo_wrapper
REM =============================================================================

echo.
echo ============================================================
echo   EmbedSim -- C Extension Build
echo ============================================================

set BUILD_LIB=build\lib.win-amd64-cpython-312

pushd "%~dp0electrical_blocks\c_src"
if errorlevel 1 (
    echo ERROR: Could not enter electrical_blocks\c_src
    echo        Is build_all.bat at the project root?
    exit /b 1
)
echo   Working dir: %CD%

echo.
echo [1/5]  coordinate_transform_wrapper
python setup_coordinate_transform.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  coordinate_transform_wrapper

echo.
echo [2/5]  smc_wrapper
python setup_smc.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  smc_wrapper

echo.
echo [3/5]  speed_pi_wrapper
python setup_speed_pi.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  speed_pi_wrapper

echo.
echo [4/5]  svpwm_wrapper
python setup_svpwm.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  svpwm_wrapper

echo.
echo [5/5]  smo_wrapper
python setup_smo.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  smo_wrapper

echo.
echo   Copying to electrical_blocks\...
for %%f in ("%BUILD_LIB%\coordinate_transform_wrapper*.pyd") do (copy "%%f" ".." > nul & echo        Copied: %%~nxf)
for %%f in ("%BUILD_LIB%\smc_wrapper*.pyd")                  do (copy "%%f" ".." > nul & echo        Copied: %%~nxf)
for %%f in ("%BUILD_LIB%\speed_pi_wrapper*.pyd")             do (copy "%%f" ".." > nul & echo        Copied: %%~nxf)
for %%f in ("%BUILD_LIB%\svpwm_wrapper*.pyd")                do (copy "%%f" ".." > nul & echo        Copied: %%~nxf)
for %%f in ("%BUILD_LIB%\smo_wrapper*.pyd")                  do (copy "%%f" ".." > nul & echo        Copied: %%~nxf)

popd
echo.
echo ============================================================
echo   All 5 extensions built and copied successfully.
echo ============================================================
goto :eof

:error
echo.
echo ERROR: Build failed. See output above.
popd
exit /b 1
