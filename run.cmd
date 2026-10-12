@echo off
rem ===========================================================================
rem  Run a Python script with the project venv, with UTF-8 output fixed.
rem
rem  Usage (from the deeplearn folder):
rem    run.cmd lessons\01_tensors_autograd.py
rem    run.cmd lessons\03_data_pipeline.py
rem    run.cmd project_cv\scripts\audit_data.py --data-dir data\flowers
rem    run.cmd project_cv\src\train.py --data-dir data\flowers_split --run-name 01_baseline --arch smallcnn --no-pretrained --epochs 40
rem
rem  Save output to a file (UTF-8, safe to send):
rem    run.cmd project_cv\src\train.py --data-dir data\flowers_split --run-name 01_baseline > outputs\01.log 2>&1
rem
rem  NOTE: keep this file ASCII-only. cmd.exe reads .cmd files using the OEM
rem        code page, so non-ASCII comments here would break parsing.
rem ===========================================================================

setlocal

rem --- console + Python I/O in UTF-8 ---
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"

rem --- locate project root (this script's folder) and venv python ---
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

rem --- keep model caches inside the project ---
rem torchvision / huggingface / matplotlib default to C:\Users\<you>\.cache etc,
rem which is outside the workspace and therefore not writable here.
set "TORCH_HOME=%ROOT%.cache\torch"
set "HF_HOME=%ROOT%.cache\huggingface"
set "MPLCONFIGDIR=%ROOT%.cache\matplotlib"
rem If HF downloads are slow, uncomment this mirror:
rem set "HF_ENDPOINT=https://hf-mirror.com"

if not exist "%PY%" (
    echo [ERROR] venv python not found:
    echo         %PY%
    echo         Check that the .venv folder is intact.
    exit /b 1
)

if "%~1"=="" (
    echo Usage:   run.cmd ^<script^> [args...]
    echo Example: run.cmd lessons\01_tensors_autograd.py
    echo Example: run.cmd project_cv\src\train.py --data-dir data\flowers_split --run-name 01_baseline --arch smallcnn --no-pretrained --epochs 40
    exit /b 1
)

rem --- run from project root so relative paths like data\flowers are stable ---
pushd "%ROOT%"
echo Running: python %*
"%PY%" %*
set "RC=%ERRORLEVEL%"
popd
endlocal & exit /b %RC%
