@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM  DigiSteel-YOLO - Complete Environment Setup Script
REM ============================================================================
REM  This script sets up the full development environment for the
REM  DigiSteel-YOLO project. Run this after cloning the repository.
REM
REM  Requirements:
REM    - Windows 10/11
REM    - Python 3.10+ installed and added to PATH
REM    - NVIDIA GPU with CUDA 12.x support
REM    - Git installed
REM
REM  Usage:
REM    Double-click this file OR run from cmd: setup_env.bat
REM ============================================================================

title DigiSteel-YOLO Environment Setup

echo.
echo ============================================================
echo   DigiSteel-YOLO - Environment Setup
echo   Steel Surface Defect Detection with YOLO
echo ============================================================
echo.

REM --- Configuration ---
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "REQUIREMENTS=%PROJECT_DIR%requirements.txt"
set "PYTHON_MIN_MAJOR=3"
set "PYTHON_MIN_MINOR=10"
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121"

REM --- Change to project directory ---
cd /d "%PROJECT_DIR%"

REM ============================================================================
REM  STEP 1: Check Python Installation
REM ============================================================================
echo [1/8] Checking Python installation...
echo -----------------------------------------------------------

REM Try python first, then python3
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    where python3 >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python is not installed or not in PATH!
        echo.
        echo Please install Python 3.10+ from:
        echo   https://www.python.org/downloads/
        echo.
        echo IMPORTANT: Check "Add Python to PATH" during installation!
        echo.
        pause
        exit /b 1
    ) else (
        set "PYTHON_CMD=python3"
    )
) else (
    set "PYTHON_CMD=python"
)

REM Get Python version
for /f "tokens=2 delims= " %%i in ('%PYTHON_CMD% --version 2^>^&1') do set "PY_VER=%%i"
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

echo    Found Python %PY_VER%

REM Check minimum version
if %PY_MAJOR% lss %PYTHON_MIN_MAJOR% (
    echo [ERROR] Python %PYTHON_MIN_MAJOR%.%PYTHON_MIN_MINOR%+ required, found %PY_VER%
    pause
    exit /b 1
)
if %PY_MAJOR% equ %PYTHON_MIN_MAJOR% if %PY_MINOR% lss %PYTHON_MIN_MINOR% (
    echo [ERROR] Python %PYTHON_MIN_MAJOR%.%PYTHON_MIN_MINOR%+ required, found %PY_VER%
    pause
    exit /b 1
)
echo    [OK] Python version check passed.
echo.

REM ============================================================================
REM  STEP 2: Check NVIDIA GPU / CUDA
REM ============================================================================
echo [2/8] Checking NVIDIA GPU and CUDA...
echo -----------------------------------------------------------

where nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] nvidia-smi not found. Make sure NVIDIA drivers are installed.
    echo    Download from: https://www.nvidia.com/Download/index.aspx
    echo.
) else (
    echo    NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul
    echo.
)

REM Check CUDA toolkit
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo    [INFO] CUDA toolkit (nvcc) not found in PATH.
    echo    PyTorch will use its bundled CUDA runtime (this is OK).
) else (
    for /f "tokens=6 delims= " %%i in ('nvcc --version 2^>^&1 ^| findstr "release"') do set "CUDA_VER=%%i"
    echo    [OK] CUDA Toolkit: !CUDA_VER!
)
echo.

REM ============================================================================
REM  STEP 3: Create Virtual Environment
REM ============================================================================
echo [3/8] Creating Python virtual environment...
echo -----------------------------------------------------------

if exist "%VENV_DIR%" (
    echo    [INFO] Virtual environment already exists at: %VENV_DIR%
    echo    [INFO] Skipping creation. Delete venv\ folder to recreate.
) else (
    echo    Creating venv at: %VENV_DIR%
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        echo    Try: %PYTHON_CMD% -m pip install --upgrade pip
        pause
        exit /b 1
    )
    echo    [OK] Virtual environment created.
)
echo.

REM ============================================================================
REM  STEP 4: Activate Virtual Environment
REM ============================================================================
echo [4/8] Activating virtual environment...
echo -----------------------------------------------------------

call "%VENV_DIR%\Scripts\activate.bat"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo    [OK] Virtual environment activated.
echo    Python: %VENV_DIR%\Scripts\python.exe
echo.

REM ============================================================================
REM  STEP 5: Upgrade pip, setuptools, wheel
REM ============================================================================
echo [5/8] Upgrading pip, setuptools, and wheel...
echo -----------------------------------------------------------

python -m pip install --upgrade pip setuptools wheel --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Failed to upgrade pip. Continuing anyway...
) else (
    echo    [OK] pip upgraded to latest version.
)
echo.

REM ============================================================================
REM  STEP 6: Install PyTorch with CUDA 12.1
REM ============================================================================
echo [6/8] Installing PyTorch with CUDA 12.1 support...
echo -----------------------------------------------------------
echo    This may take several minutes (~2.5 GB download)...
echo.

REM Check if torch is already installed with correct CUDA
python -c "import torch; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')" 2>nul
if %ERRORLEVEL% equ 0 (
    echo    [INFO] PyTorch already installed. Skipping.
) else (
    echo    Installing PyTorch + torchvision from: %TORCH_INDEX_URL%
    pip install torch torchvision --index-url %TORCH_INDEX_URL%
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] PyTorch installation failed!
        echo    Try downloading manually or check your internet connection.
        pause
        exit /b 1
    )
    echo    [OK] PyTorch installed successfully.
)
echo.

REM ============================================================================
REM  STEP 7: Install Project Dependencies
REM ============================================================================
echo [7/8] Installing project dependencies from requirements.txt...
echo -----------------------------------------------------------

if not exist "%REQUIREMENTS%" (
    echo [ERROR] requirements.txt not found at: %REQUIREMENTS%
    pause
    exit /b 1
)

pip install -r "%REQUIREMENTS%" --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some packages may have failed. Retrying with verbose output...
    pip install -r "%REQUIREMENTS%"
)

echo    [OK] Dependencies installed.
echo.

REM Install project in editable mode (makes 'digisteel' package importable)
echo    Installing DigiSteel package in editable mode...
pip install -e "%PROJECT_DIR%" --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Editable install failed. Trying without -e flag...
    pip install "%PROJECT_DIR%" --quiet
)
echo    [OK] DigiSteel package installed.
echo.

REM ============================================================================
REM  STEP 8: Create Directories and Verify Installation
REM ============================================================================
echo [8/8] Finalizing setup...
echo -----------------------------------------------------------

REM Create required directories
if not exist "%PROJECT_DIR%datasets" mkdir "%PROJECT_DIR%datasets"
if not exist "%PROJECT_DIR%weights" mkdir "%PROJECT_DIR%weights"
if not exist "%PROJECT_DIR%runs" mkdir "%PROJECT_DIR%runs"
if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"
echo    [OK] Project directories created.

REM Verify installation
echo.
echo    Verifying installation...
echo.

python -c "
import sys
print(f'    Python:       {sys.version}')
try:
    import torch
    print(f'    PyTorch:      {torch.__version__} (CUDA: {torch.version.cuda})')
    print(f'    GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')
except: print('    [WARN] PyTorch not available')
try:
    import ultralytics; print(f'    Ultralytics:  {ultralytics.__version__}')
except: print('    [WARN] Ultralytics not available')
try:
    import cv2; print(f'    OpenCV:       {cv2.__version__}')
except: print('    [WARN] OpenCV not available')
try:
    import numpy; print(f'    NumPy:        {numpy.__version__}')
except: print('    [WARN] NumPy not available')
try:
    import pandas; print(f'    Pandas:       {pandas.__version__}')
except: print('    [WARN] Pandas not available')
try:
    import albumentations; print(f'    Albu:         {albumentations.__version__}')
except: print('    [WARN] Albumentations not available')
try:
    import onnx; print(f'    ONNX:         {onnx.__version__}')
except: print('    [WARN] ONNX not available')
try:
    import digisteel; print(f'    DigiSteel:    OK (installed)')
except: print('    [WARN] DigiSteel package not importable')
"

echo.

REM ============================================================================
REM  DONE
REM ============================================================================
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To activate the environment in a NEW terminal:
echo     cd %PROJECT_DIR%
echo     venv\Scripts\activate
echo.
echo   To start training:
echo     python scripts\train.py
echo.
echo   To run tests:
echo     pytest tests/
echo.
echo   To launch Jupyter Lab:
echo     jupyter lab
echo.
echo   Project structure:
echo     configs/       - Model and data configurations
echo     digisteel/     - Main Python package
echo     scripts/       - Training and evaluation scripts
echo     tests/         - Unit tests
echo     datasets/      - Dataset storage (add your data here)
echo     weights/       - Model weights (downloaded automatically)
echo     runs/          - Training runs output
echo ============================================================
echo.

REM Mark setup as done
echo %date% %time% > "%PROJECT_DIR%setup_done.flag"

pause
