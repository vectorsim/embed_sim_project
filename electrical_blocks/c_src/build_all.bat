@echo off
REM =============================================================================
REM build_all.bat
REM =============================================================================
REM Compile all electrical_blocks Cython extensions on Windows.
REM
REM Run from the electrical_blocks\c_src directory:
REM   cd electrical_blocks\c_src
REM   build_all.bat
REM
REM Output: coordinate_transform_wrapper.pyd  (copied to parent package dir)
REM =============================================================================

echo ============================================================
echo  Building electrical_blocks C extensions  (Windows)
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/1]  coordinate_transform_wrapper
python setup_coordinate_transform.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  coordinate_transform_wrapper compiled

echo.
echo Copying .pyd files to parent package directory...
for /r . %%f in (coordinate_transform_wrapper*.pyd) do (
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
echo ERROR: Build failed. See output above.
exit /b 1
