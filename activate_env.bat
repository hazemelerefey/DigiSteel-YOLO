@echo off
REM ============================================================================
REM  DigiSteel-YOLO - Quick Environment Activation
REM ============================================================================
REM  Use this to activate the environment in a new terminal window.
REM  Run setup_env.bat FIRST if you haven't set up the environment yet.
REM ============================================================================

set "VENV_DIR=%~dp0venv"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo    Run setup_env.bat first to create the environment.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo ============================================================
echo   DigiSteel-YOLO Environment Activated
echo ============================================================
echo   Python: %VENV_DIR%\Scripts\python.exe
echo   Project: %~dp0
echo ============================================================
echo.

cmd /k "cd /d %~dp0"
