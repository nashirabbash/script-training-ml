@echo off
REM ============================================================================
REM GPU Setup Script - Install cuML untuk RTX 3050
REM ============================================================================

echo ╔════════════════════════════════════════════════════════════════════╗
echo ║         GPU Acceleration Setup - RAPIDS cuML                       ║
echo ║              untuk RTX 3050 (4GB VRAM)                             ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda tidak ditemukan!
    echo.
    echo Silakan install Miniconda terlebih dahulu:
    echo https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo [Step 1] Checking CUDA version...
echo.
nvidia-smi
echo.
echo Catat CUDA Version di atas (contoh: CUDA Version: 12.6)
echo.

:choose_cuda
echo [Step 2] Pilih CUDA version yang sesuai:
echo.
echo   1. CUDA 11.x (untuk driver lama)
echo   2. CUDA 12.x (untuk driver baru - recommended)
echo   3. Cancel
echo.
set /p cuda_choice="Pilihan (1-3): "

if "%cuda_choice%"=="1" (
    set cuda_ver=11.8
    set cuml_pkg=cuml-cu11
    goto :install
)
if "%cuda_choice%"=="2" (
    set cuda_ver=12.0
    set cuml_pkg=cuml-cu12
    goto :install
)
if "%cuda_choice%"=="3" (
    echo Installation dibatalkan.
    pause
    exit /b 0
)
echo Invalid choice!
goto :choose_cuda

:install
echo.
echo [Step 3] Pilih mode instalasi:
echo.
echo   1. Create NEW environment (ml_gpu) - RECOMMENDED
echo   2. Install to CURRENT environment
echo   3. Cancel
echo.
set /p install_choice="Pilihan (1-3): "

if "%install_choice%"=="1" goto :new_env
if "%install_choice%"=="2" goto :current_env
if "%install_choice%"=="3" (
    echo Installation dibatalkan.
    pause
    exit /b 0
)
echo Invalid choice!
goto :install

:new_env
echo.
echo ════════════════════════════════════════════════════════════════════
echo Creating new environment: ml_gpu
echo ════════════════════════════════════════════════════════════════════
echo.

conda create -n ml_gpu python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create environment!
    pause
    exit /b 1
)

call conda activate ml_gpu
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate environment!
    pause
    exit /b 1
)

goto :install_packages

:current_env
echo.
echo ════════════════════════════════════════════════════════════════════
echo Installing to current environment...
echo ════════════════════════════════════════════════════════════════════
echo.

:install_packages
echo [Step 4] Installing cuML (this may take 5-10 minutes)...
echo.

conda install -c rapidsai -c conda-forge -c nvidia cuml=%cuda_ver% python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] cuML installation failed!
    echo.
    echo Try manual installation:
    echo   conda install -c rapidsai -c conda-forge -c nvidia cuml
    pause
    exit /b 1
)

echo.
echo [Step 5] Installing Python dependencies...
echo.

pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages failed to install
    echo Continue anyway...
)

echo.
echo ════════════════════════════════════════════════════════════════════
echo Installation Complete!
echo ════════════════════════════════════════════════════════════════════
echo.

echo [Step 6] Verifying GPU setup...
echo.
python test_gpu.py

echo.
echo ════════════════════════════════════════════════════════════════════
echo NEXT STEPS:
echo ════════════════════════════════════════════════════════════════════
echo.

if "%install_choice%"=="1" (
    echo 1. Activate environment:
    echo    conda activate ml_gpu
    echo.
)

echo 2. Run training scripts:
echo    python tuning_anova.py
echo    python tuning_chi2.py
echo    python tuning_relieff.py
echo.
echo 3. Check for "GPU (RTX 3050)" in output
echo.
echo ════════════════════════════════════════════════════════════════════
echo.

pause
