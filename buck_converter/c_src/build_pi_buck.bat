@echo off
REM =============================================================================
REM build_pi_buck.bat  —  electrical_blocks\c_src
REM =============================================================================
REM Compile PI Buck Controller Cython extension on Windows.
REM
REM Run from the electrical_blocks\c_src directory:
REM   cd electrical_blocks\c_src
REM   build_pi_buck.bat
REM
REM Output (.pyd files copied to parent package dir):
REM   pi_buck_wrapper.pyd
REM =============================================================================

echo ============================================================
echo  Building PI Buck Controller C extension  (Windows)
echo ============================================================

cd /d "%~dp0"

REM Build output dir — CPython 3.12 x64
set BUILD_LIB=build\lib.win-amd64-cpython-312

echo.
echo [1/1]  pi_buck_wrapper  (PI Buck Controller)
python setup_pi_buck.py build_ext --inplace
if errorlevel 1 goto :error
echo        OK  pi_buck_wrapper compiled

echo.
echo Copying .pyd files to parent package directory...
for %%f in ("%BUILD_LIB%\pi_buck_wrapper*.pyd") do (
    copy "%%f" ".." > nul
    echo        Copied: %%~nxf to ..\
)

echo.
echo Copying to buck_converter/c_src/ for integration...
if not exist "..\..\buck_converter\c_src" (
    mkdir "..\..\buck_converter\c_src" 2>nul
)
for %%f in ("%BUILD_LIB%\pi_buck_wrapper*.pyd") do (
    copy "%%f" "..\..\buck_converter\c_src\" > nul
    echo        Copied: %%~nxf to ..\..\buck_converter\c_src\
)

echo.
echo Copying C source files to buck_converter/c_src/...
copy "pi_buck_controller.c" "..\..\buck_converter\c_src\" > nul
echo        Copied: pi_buck_controller.c to ..\..\buck_converter\c_src\
copy "pi_buck_controller.h" "..\..\buck_converter\c_src\" > nul
echo        Copied: pi_buck_controller.h to ..\..\buck_converter\c_src\
copy "pi_buck_wrapper.pyx" "..\..\buck_converter\c_src\" > nul
echo        Copied: pi_buck_wrapper.pyx to ..\..\buck_converter\c_src\

echo.
echo Copying Python block to buck_converter/...
copy "pi_buck_block.py" "..\..\buck_converter\" > nul
echo        Copied: pi_buck_block.py to ..\..\buck_converter\

echo.
echo ============================================================
echo  PI Buck Controller built and copied successfully.
echo ============================================================
echo.
echo Files copied to:
echo   - electrical_blocks\                (pi_buck_wrapper.pyd)
echo   - buck_converter\c_src\             (C sources + wrapper)
echo   - buck_converter\                   (pi_buck_block.py)
echo.
goto :eof

:error
echo.
echo ERROR: Build failed. See output above.
exit /b 1