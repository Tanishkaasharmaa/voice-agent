"""Windows-specific helpers.

All paths are environment-aware (APPDATA / LOCALAPPDATA); nothing is
hardcoded to a username or absolute path.
"""

from __future__ import annotations

import os
import sys
import winreg
from pathlib import Path

APP_NAME = "LocalVoiceAgent"
_AUTHORS_KEY = r"Software\LocalVoiceAgent"


def is_windows() -> bool:
    return sys.platform == "win32"


def app_data_dir() -> Path:
    """Root data directory: %APPDATA%/LocalVoiceAgent (roaming)."""
    base = os.environ.get("APPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Roaming")
    return Path(base) / APP_NAME


def local_data_dir() -> Path:
    """%LOCALAPPDATA%/LocalVoiceAgent — large data (models, cache)."""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME


def ensure_directories() -> dict[str, Path]:
    """Create and return the standard application directories."""
    dirs = {
        "config": app_data_dir() / "config",
        "data": app_data_dir() / "data",
        "logs": app_data_dir() / "logs",
        "models": local_data_dir() / "models",
        "cache": local_data_dir() / "cache",
        "temp": local_data_dir() / "temp",
        "recordings": local_data_dir() / "recordings",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_registry_value(name: str, default: str | None = None) -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _AUTHORS_KEY) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
    except OSError:
        return default


def set_registry_value(name: str, value: str) -> None:
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _AUTHORS_KEY) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    except OSError:
        pass


def start_with_windows(enabled: bool) -> bool:
    """Register/remove the HKCU Run entry. Never requires admin rights."""
    if not is_windows():
        return False
    run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                exe = sys.executable
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, f'"{exe}" --background')
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
        return True
    except OSError:
        return False


def is_start_with_windows_enabled() -> bool:
    run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key) as key:
            winreg.QueryValueEx(key, APP_NAME)
        return True
    except OSError:
        return False
