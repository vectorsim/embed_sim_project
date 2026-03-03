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
echo  Building electricalblocks C extensions (Windows)
echo ============================================================

cd /d "%~dp0"



echo.
echo [1/1]  transform_wrapper
python setup_coordinate_transform.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  transforms_wrapper compiled



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
