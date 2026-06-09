@echo off
setlocal enabledelayedexpansion
title DigiSteel-YOLO - Team Environment Setup
color 0A

echo ============================================================
echo    DigiSteel-YOLO - Team Environment Setup
echo    Steel Surface Defect Detection with YOLO11
echo ============================================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found
echo.

:: Check NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] nvidia-smi not found. GPU may not be available.
    echo           Install NVIDIA drivers from https://www.nvidia.com/drivers
    echo.
    set GPU_AVAILABLE=0
) else (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
    echo.
    set GPU_AVAILABLE=1
)

:: Create virtual environment
if exist "venv" (
    echo [INFO] venv folder already exists. Skipping creation.
) else (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        color 0C
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo       Done.
)
echo.

:: Activate venv
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo       Done.
echo.

:: Install PyTorch with CUDA
echo [3/4] Installing PyTorch with CUDA 12.4 support...
echo       This is a large download (~2.5 GB), please be patient.
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    color 0C
    echo [ERROR] Failed to install PyTorch. Check your internet connection.
    pause
    exit /b 1
)
echo.
echo       Done.
echo.

:: Install remaining dependencies
echo [4/4] Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    color 0C
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo.
echo       Done.
echo.

:: Verify installation
echo ============================================================
echo    Verifying installation...
echo ============================================================
echo.

python -c "import torch; print(f'PyTorch:       {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version:   {torch.version.cuda}'); print(f'Device:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}'); x=torch.randn(100,100,device='cuda' if torch.cuda.is_available() else 'cpu'); y=torch.mm(x,x.T); print(f'Compute test:   PASSED ({y.device})')"
echo.

if errorlevel 1 (
    color 0C
    echo [ERROR] Verification failed. Check the errors above.
    pause
    exit /b 1
)

:: Install project in editable mode
pip install -e . >nul 2>&1

echo ============================================================
color 0A
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo    To use in the future, open a terminal here and run:
echo        venv\Scripts\activate
echo.
echo    To run training:
echo        python -m digisteel.train
echo.
echo    To run inference:
echo        python -m digisteel.predict --source image.jpg
echo.
pause
