@echo off
setlocal

set "SUITE=%~1"
if "%SUITE%"=="" set "SUITE=smoke"

set "MARKER=smoke"
if /I "%SUITE%"=="unit" set "MARKER=unit and not slow"
if /I "%SUITE%"=="api" set "MARKER=api and not slow"
if /I "%SUITE%"=="api-full" set "MARKER=api"
if /I "%SUITE%"=="full" set "MARKER="

set "PYTHON_PATH=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_PATH%" (
  echo No se encontro %PYTHON_PATH%. Crea primero el entorno virtual con 'python -m venv .venv' e instala requirements.
  exit /b 1
)

pushd "%~dp0"
if defined MARKER (
  "%PYTHON_PATH%" -m pytest tests -q -m "%MARKER%"
) else (
  "%PYTHON_PATH%" -m pytest tests -q
)
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
