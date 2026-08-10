@echo off
call "%~dp0test-suite.cmd" smoke
exit /b %ERRORLEVEL%
