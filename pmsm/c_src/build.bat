@echo off
setlocal EnableExtensions EnableDelayedExpansion

title EmbedSim Control Wrapper Build

echo.
echo ============================================================
echo  EmbedSim Control Wrapper Build - Windows
echo  Sensor-Based PMSM / DFC + Motor State Reporting
echo ============================================================
echo.

rem ============================================================
rem  Resolve directories
rem ============================================================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

pushd "%SCRIPT_DIR%"
if errorlevel 1 goto :FAIL_DIR

set "PARENT_DIR=%SCRIPT_DIR%\.."
set "PROJECT_ROOT=%PARENT_DIR%\.."
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

echo ------------------------------------------------------------
echo  Build directories
echo ------------------------------------------------------------
echo Source directory :
echo   %SCRIPT_DIR%
echo.
echo Parent directory :
echo   %PARENT_DIR%
echo.
echo Project root :
echo   %PROJECT_ROOT%
echo.

rem ============================================================
rem  [1/6] CLEAN
rem ============================================================

echo ------------------------------------------------------------
echo  [1/6] Cleaning previous build
echo ------------------------------------------------------------
echo.

if exist "build" (
    echo Removing build directory...
    rmdir /s /q "build"

    if exist "build" (
        echo ERROR: Could not remove build directory.
        goto :FAIL
    )
)

echo Removing old generated files...

del /f /q "embedsim_control_wrapper*.pyd" 2>nul
del /f /q "embedsim_control_wrapper*.c" 2>nul
del /f /q "embedsim_control_wrapper*.html" 2>nul

rem Remove previous parent artifact.
if exist "%PARENT_DIR%\embedsim_control_wrapper.pyd" (
    echo Removing old parent artifact...
    del /f /q "%PARENT_DIR%\embedsim_control_wrapper.pyd"

    if exist "%PARENT_DIR%\embedsim_control_wrapper.pyd" (
        echo ERROR: Could not remove old parent artifact.
        goto :FAIL
    )
)

echo.
echo Clean complete.
echo.

rem ============================================================
rem  [2/6] PYTHON
rem ============================================================

echo ------------------------------------------------------------
echo  [2/6] Selecting Python environment
echo ------------------------------------------------------------
echo.

if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
    echo Using project virtual environment:
    echo   %VENV_PYTHON%
) else (
    set "PYTHON=python"
    echo Project .venv not found.
    echo Using system Python.
)

echo.
echo Python version:
%PYTHON% --version

if errorlevel 1 (
    echo.
    echo ERROR: Python was not found or could not start.
    goto :FAIL
)

echo.

rem ============================================================
rem  [3/6] DEPENDENCIES
rem ============================================================

echo ------------------------------------------------------------
echo  [3/6] Checking Python build dependencies
echo ------------------------------------------------------------
echo.

echo Checking setuptools...
%PYTHON% -c "import setuptools; print('  setuptools:', setuptools.__version__)" >nul 2>&1

if errorlevel 1 (
    echo setuptools not found.
    echo Installing setuptools...
    %PYTHON% -m pip install --upgrade setuptools

    if errorlevel 1 (
        echo ERROR: Failed to install setuptools.
        goto :FAIL
    )
) else (
    %PYTHON% -c "import setuptools; print('  setuptools:', setuptools.__version__)"
)

echo.
echo Checking Cython...
%PYTHON% -c "import Cython; print('  Cython:', Cython.__version__)" >nul 2>&1

if errorlevel 1 (
    echo Cython not found.
    echo Installing Cython...
    %PYTHON% -m pip install --upgrade cython

    if errorlevel 1 (
        echo ERROR: Failed to install Cython.
        goto :FAIL
    )
) else (
    %PYTHON% -c "import Cython; print('  Cython:', Cython.__version__)"
)

echo.
echo Checking NumPy...
%PYTHON% -c "import numpy; print('  NumPy:', numpy.__version__)" >nul 2>&1

if errorlevel 1 (
    echo NumPy not found.
    echo Installing NumPy...
    %PYTHON% -m pip install --upgrade numpy

    if errorlevel 1 (
        echo ERROR: Failed to install NumPy.
        goto :FAIL
    )
) else (
    %PYTHON% -c "import numpy; print('  NumPy:', numpy.__version__)"
)

echo.

rem ============================================================
rem  [4/6] SOURCE FILE CHECK
rem ============================================================

echo ------------------------------------------------------------
echo  [4/6] Checking required source/header files
echo ------------------------------------------------------------
echo.

set "MISSING=0"

for %%f in (
    embedsim_control_wrapper.pyx
    embed_sim_control.c
    embed_sim_control.h
    embed_sim_dfc_controller.c
    embed_sim_dfc_controller.h
    embed_sim_coordinate_transform.c
    embed_sim_coordinate_transform.h
    embed_sim_sv_pwm.c
    embed_sim_sv_pwm.h
    embed_sim_matrix.c
    embed_sim_matrix.h
    embed_sim_cython_interface.c
    embed_sim_cython_interface.h
) do (
    if not exist "%%f" (
        echo ERROR: Missing:
        echo   %%f
        set "MISSING=1"
    ) else (
        echo Found:
        echo   %%f
    )
)

if "!MISSING!"=="1" (
    echo.
    echo ERROR: Required source/header files are missing.
    goto :FAIL
)

echo.
echo All required source/header files found.
echo.

rem ============================================================
rem  [5/6] BUILD
rem ============================================================

echo ------------------------------------------------------------
echo  [5/6] Building embedsim_control_wrapper
echo ------------------------------------------------------------
echo.

echo Command:
echo.
echo   %PYTHON% embedsim_control_wrapper.py build_ext --inplace
echo.

%PYTHON% embedsim_control_wrapper.py build_ext --inplace

set "BUILD_ERROR=!errorlevel!"

echo.
echo Build process returned error level:
echo   !BUILD_ERROR!
echo.

if not "!BUILD_ERROR!"=="0" (
    echo ============================================================
    echo ERROR: BUILD FAILED
    echo ============================================================
    echo.
    echo The compiler/Cython output above contains the error.
    goto :FAIL
)

echo ============================================================
echo BUILD COMMAND COMPLETED SUCCESSFULLY
echo ============================================================
echo.

rem ============================================================
rem  Locate generated .pyd
rem ============================================================

echo Searching for generated .pyd...
echo.

set "PYD_FOUND=0"
set "PYD_FILE="

for %%f in (embedsim_control_wrapper*.pyd) do (
    set "PYD_FOUND=1"
    set "PYD_FILE=%%f"
)

if "!PYD_FOUND!"=="0" (
    echo ERROR: Build returned success but no .pyd was found.
    echo.
    echo Current directory:
    echo   %SCRIPT_DIR%
    echo.
    echo Searching recursively...
    echo.

    dir /s /b "embedsim_control_wrapper*.pyd" 2>nul

    echo.
    echo ERROR: Cannot locate generated extension.
    goto :FAIL
)

echo Generated .pyd found:
echo   %SCRIPT_DIR%\!PYD_FILE!
echo.

rem ============================================================
rem  [6/6] COPY ARTIFACT TO PARENT
rem ============================================================

echo ------------------------------------------------------------
echo  [6/6] Copying artifact to parent directory
echo ------------------------------------------------------------
echo.

echo Source:
echo   %SCRIPT_DIR%\!PYD_FILE!
echo.
echo Destination:
echo   %PARENT_DIR%\embedsim_control_wrapper.pyd
echo.

copy /y "%SCRIPT_DIR%\!PYD_FILE!" "%PARENT_DIR%\embedsim_control_wrapper.pyd"

set "COPY_ERROR=!errorlevel!"

echo.
echo Copy command returned error level:
echo   !COPY_ERROR!
echo.

if not "!COPY_ERROR!"=="0" (
    echo ============================================================
    echo ERROR: ARTIFACT COPY FAILED
    echo ============================================================
    goto :FAIL
)

rem ============================================================
rem  Verify parent artifact
rem ============================================================

echo ------------------------------------------------------------
echo  Verifying parent artifact
echo ------------------------------------------------------------
echo.

if not exist "%PARENT_DIR%\embedsim_control_wrapper.pyd" (
    echo ============================================================
    echo ERROR: PARENT ARTIFACT DOES NOT EXIST
    echo ============================================================
    echo.
    echo Expected:
    echo   %PARENT_DIR%\embedsim_control_wrapper.pyd
    echo.
    goto :FAIL
)

echo SUCCESS: Parent artifact exists.
echo.

rem ============================================================
rem  Verify file size
rem ============================================================

for %%A in ("%PARENT_DIR%\embedsim_control_wrapper.pyd") do (
    set "PYD_SIZE=%%~zA"
)

echo Parent artifact:
echo   %PARENT_DIR%\embedsim_control_wrapper.pyd
echo.
echo Artifact size:
echo   !PYD_SIZE! bytes
echo.

if "!PYD_SIZE!"=="0" (
    echo ============================================================
    echo ERROR: Parent .pyd exists but is ZERO BYTES.
    echo ============================================================
    goto :FAIL
)

goto :SUCCESS


rem ################################################################
rem  SUCCESS
rem ################################################################

:SUCCESS

echo.
echo ============================================================
echo.
echo  BUILD SUCCESSFUL
echo.
echo ============================================================
echo.
echo Source artifact:
echo   %SCRIPT_DIR%\!PYD_FILE!
echo.
echo Parent artifact:
echo   %PARENT_DIR%\embedsim_control_wrapper.pyd
echo.
echo ============================================================
echo  Artifact transfer verified.
echo ============================================================
echo.
echo Features:
echo   - Sensor-based PMSM control
echo   - DFC controller
echo   - Clarke transform
echo   - Park transform
echo   - Inverse Park transform
echo   - Inverse Clarke transform
echo   - Motor state reporting
echo   - PI state reporting
echo   - Torque reporting
echo   - Voltage reporting
echo   - Trajectory reporting
echo   - Spinning/stopped detection
echo.
echo Python import:
echo.
echo   from embedsim_control_wrapper import control_init
echo   from embedsim_control_wrapper import control_step
echo   from embedsim_control_wrapper import get_motor_state
echo.
echo   control_init()
echo   state = get_motor_state()
echo.
echo ============================================================
echo  NOTHING ELSE WILL BE DONE.
echo  Press any key to close this window.
echo ============================================================
echo.

pause

popd
exit /b 0


rem ################################################################
rem  FAILURE
rem ################################################################

:FAIL

echo.
echo.
echo ============================================================
echo.
echo  BUILD FAILED
echo.
echo ============================================================
echo.
echo The build script encountered an error.
echo.
echo IMPORTANT:
echo   The window will remain open so you can inspect the error.
echo.
echo Check the messages above for:
echo   - C compiler errors
echo   - Cython errors
echo   - Missing source/header files
echo   - Linker errors
echo   - Copy errors
echo   - Missing parent artifact
echo.
echo ============================================================
echo  Current source directory:
echo    %SCRIPT_DIR%
echo.
echo  Expected parent artifact:
echo    %PARENT_DIR%\embedsim_control_wrapper.pyd
echo ============================================================
echo.
echo Press any key to close this window.
echo.

pause

popd
exit /b 1


rem ################################################################
rem  DIRECTORY FAILURE
rem ################################################################

:FAIL_DIR

echo.
echo ============================================================
echo ERROR: Could not enter source directory.
echo ============================================================
echo.
echo Expected directory:
echo   %SCRIPT_DIR%
echo.
echo Press any key to close.
echo.

pause

exit /b 1