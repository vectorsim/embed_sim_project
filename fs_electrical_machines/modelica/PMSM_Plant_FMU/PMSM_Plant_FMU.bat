@echo off
SET PATH=;C:/Program Files/OpenModelica1.26.3-64bit/bin/;%PATH%;
SET ERRORLEVEL=
CALL "%CD%/PMSM_Plant_FMU.exe" %*
SET RESULT=%ERRORLEVEL%

EXIT /b %RESULT%
