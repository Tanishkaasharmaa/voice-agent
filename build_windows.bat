@echo off
REM ============================================================
REM  LocalVoiceAgent - Windows build script (PyInstaller)
REM
REM  Builds a one-dir distribution (recommended): AI models stay
REM  OUTSIDE the exe in %LOCALAPPDATA%\LocalVoiceAgent\models.
REM
REM  Prerequisites:
REM    python -m venv .venv
REM    .venv\Scripts\pip install -r requirements.txt pyinstaller
REM ============================================================
setlocal

set VENV=.venv\Scripts\python.exe
if not exist "%VENV%" (
    echo [ERROR] Virtual environment not found. Run:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt pyinstaller
    exit /b 1
)

echo Building LocalVoiceAgent (one-dir)...
"%VENV%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --noconsole ^
    --name LocalVoiceAgent ^
    --add-data "config;config" ^
    --hidden-import faster_whisper ^
    --hidden-import onnxruntime ^
    --hidden-import kokoro_onnx ^
    --hidden-import sounddevice ^
    --hidden-import pynput.keyboard._win32 ^
    --hidden-import pynput.mouse._win32 ^
    main.py

if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo Build complete: dist\LocalVoiceAgent\LocalVoiceAgent.exe
echo Models are stored in %%LOCALAPPDATA%%\LocalVoiceAgent\models at first run.
endlocal
