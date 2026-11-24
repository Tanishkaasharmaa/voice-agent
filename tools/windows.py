"""Windows automation tools.

Uses pywin32 / SendInput — never arbitrary shell commands for these
operations. All pywin32 imports are lazy so the module imports cleanly in
tests without the dependency installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from tools.registry import ToolContext, Tool

# Friendly-name -> launch target. Only well-known, stable Windows entries.
_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "command prompt": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "control panel": "control.exe",
    "this pc": "explorer.exe",
}

_COMMON_DIRS = [
    Path(os.environ.get("ProgramFiles", "C:/Program Files")),
    Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
    Path(os.environ.get("LOCALAPPDATA", "")),
]


def _resolve_target(name: str) -> str | None:
    """Resolve a friendly name / exe / path to something we can launch."""
    name = name.strip()
    if not name:
        return None
    lowered = name.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    if name.lower().endswith(".exe"):
        found = shutil.which(name)
        if found:
            return found
        for base in _COMMON_DIRS:
            if not base:
                continue
            for candidate in base.rglob(name):
                return str(candidate)
        return None
    if "\\" in name or "/" in name:
        path = Path(name)
        if path.exists():
            return str(path)
        return None
    # Try a bare executable name on PATH.
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    return found


def _open_application(name: str, _ctx: ToolContext) -> dict:
    target = _resolve_target(name)
    if not target:
        return {"success": False, "output": f"Could not find application '{name}'"}
    try:
        if target.startswith("ms-settings:"):
            subprocess.Popen(["explorer.exe", target])  # noqa: S603 - fixed prefix
        else:
            subprocess.Popen([target], shell=False)  # noqa: S603 - no shell, safe path
        return {"success": True, "output": f"Opened {name}", "data": {"target": target}}
    except OSError as exc:
        return {"success": False, "output": f"Failed to open {name}: {exc}"}


def _close_application(name: str, _ctx: ToolContext) -> dict:
    import win32api  # type: ignore
    import win32con  # type: ignore
    import win32gui  # type: ignore
    import win32process  # type: ignore

    wanted = name.lower().replace(".exe", "")
    closed = []

    def _enum(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            exe = win32process.GetModuleFileNameEx(handle) if handle else ""
            win32api.CloseHandle(handle) if handle else None
            base = os.path.basename(exe).lower().replace(".exe", "")
            if base == wanted:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                closed.append(win32gui.GetWindowText(hwnd))
        except Exception:
            pass

    try:
        win32gui.EnumWindows(_enum, None)
    except Exception as exc:
        return {"success": False, "output": f"Failed to enumerate windows: {exc}"}
    if closed:
        return {"success": True, "output": f"Closed {len(closed)} window(s)"}
    return {"success": False, "output": f"No open window found for '{name}'"}


def _get_active_window(_ctx: ToolContext) -> dict:
    import win32gui  # type: ignore

    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd) or "Unknown"
        return {
            "success": True,
            "output": f"Active window: {title}",
            "data": {"title": title},
        }
    except Exception as exc:
        return {"success": False, "output": f"Could not read active window: {exc}"}


def build_tools() -> list:
    return [
        Tool(
            name="open_application",
            description=(
                "Open or launch an application on Windows. Accepts an app name "
                "(e.g. 'notepad', 'calculator'), an .exe name, or a full path."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Application name or path"}},
                "required": ["name"],
            },
            handler=_open_application,
        ),
        Tool(
            name="launch_application",
            description="Alias for open_application: launch an application.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_open_application,
        ),
        Tool(
            name="close_application",
            description="Close an application by name (e.g. 'notepad'). Sends a close request to its windows.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_close_application,
        ),
        Tool(
            name="get_active_window",
            description="Return the title of the currently active window.",
            parameters={"type": "object", "properties": {}},
            handler=_get_active_window,
        ),
    ]
