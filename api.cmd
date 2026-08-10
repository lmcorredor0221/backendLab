@echo off
call "%~dp0test-suite.cmd" api
exit /b %ERRORLEVEL%
