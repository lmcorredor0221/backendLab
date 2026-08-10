@echo off
call "%~dp0test-suite.cmd" unit
exit /b %ERRORLEVEL%
