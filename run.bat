@echo off
setlocal

REM ============================================
REM LocalVoiceAgent - D: drive launcher
REM All app data, models, caches and logs stay on D:
REM ============================================

REM Project directory
cd /d D:\Projects\fully-offline-windows-desktop-voice-agent

REM Python virtual environment
set "VENV_DIR=D:\Projects\fully-offline-windows-desktop-voice-agent\venv"

REM LocalVoiceAgent data
set "LOCALAPPDATA=D:\LocalAppData"
set "APPDATA=D:\AppData"

REM Ollama models
set "OLLAMA_MODELS=D:\Ollama\Models"

REM Hugging Face caches
set "HF_HOME=D:\HF-Cache"
set "HUGGINGFACE_HUB_CACHE=D:\HF-Cache\hub"
set "TRANSFORMERS_CACHE=D:\HF-Cache\transformers"

REM Pip cache
set "PIP_CACHE_DIR=D:\pip-cache"

REM Create required directories if they do not exist
if not exist "D:\LocalAppData\LocalVoiceAgent\models" mkdir "D:\LocalAppData\LocalVoiceAgent\models"
if not exist "D:\AppData\LocalVoiceAgent\logs" mkdir "D:\AppData\LocalVoiceAgent\logs"
if not exist "D:\Ollama\Models" mkdir "D:\Ollama\Models"
if not exist "D:\HF-Cache" mkdir "D:\HF-Cache"
if not exist "D:\pip-cache" mkdir "D:\pip-cache"

echo.
echo ============================================
echo  LocalVoiceAgent
echo ============================================
echo  Project:       D:\Projects\fully-offline-windows-desktop-voice-agent
echo  LOCALAPPDATA:  %LOCALAPPDATA%
echo  APPDATA:       %APPDATA%
echo  OLLAMA_MODELS: %OLLAMA_MODELS%
echo  HF_HOME:       %HF_HOME%
echo  PIP_CACHE_DIR: %PIP_CACHE_DIR%
echo ============================================
echo.

REM Check virtual environment
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo ERROR: Python virtual environment was not found:
    echo %VENV_DIR%
    echo.
    echo Create it with:
    echo py -3.12 -m venv venv
    pause
    exit /b 1
)

REM Launch the application with live logs
echo Starting LocalVoiceAgent...
echo.

"%VENV_DIR%\Scripts\python.exe" main.py --show-logs

echo.
echo LocalVoiceAgent has stopped.
pause
