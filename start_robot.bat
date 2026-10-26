@echo off
rem ==============================================================================
rem SS_PROJECT_VCR_FRESH — Windows Development Launcher
rem ==============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ======================================================================
echo   SS_PROJECT_VCR_FRESH — Autonomous Voice-Controlled Robot Launcher
echo   Project Root: %SCRIPT_DIR%
echo ======================================================================

set "PYTHON_BIN=python"
if exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%SCRIPT_DIR%\.venv\Scripts\python.exe"
    echo [INFO] Using virtual environment: .venv
) else if exist "%SCRIPT_DIR%\venv\Scripts\python.exe" (
    set "PYTHON_BIN=%SCRIPT_DIR%\venv\Scripts\python.exe"
    echo [INFO] Using virtual environment: venv
) else (
    echo [INFO] Using system python
)

echo [INFO] Running static configuration and model asset preflight check...
"%PYTHON_BIN%" -m app.main --check-config
if errorlevel 1 goto :preflight_failed

echo [INFO] Configuration and asset preflight check PASSED.
echo [INFO] Launching authoritative runtime server...
echo        (Motors default stopped; zero physical actuation on startup)
echo ----------------------------------------------------------------------

"%PYTHON_BIN%" -m app.main --serve %*
endlocal

exit /b 0

:preflight_failed
set "PREFLIGHT_CODE=%errorlevel%"
echo [FATAL] Preflight check FAILED with errorlevel %PREFLIGHT_CODE%.
echo         Missing model files, invalid pin mapping, or malformed YAML configuration.
exit /b %PREFLIGHT_CODE%
