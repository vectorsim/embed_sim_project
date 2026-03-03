@echo off
SET PATH=;C:/Program Files/OpenModelica1.25.7-64bit/bin/;%PATH%;
SET ERRORLEVEL=
CALL "%CD%/ThreePhaseMotor.exe" %*
SET RESULT=%ERRORLEVEL%

EXIT /b %RESULT%
