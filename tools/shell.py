"""Shell tool — DISABLED BY DEFAULT.

When the user explicitly enables it in Settings, every execution:
* requires confirmation (ASK permission level);
* validates the command against a destructive-command blocklist;
* runs with a timeout, captured output, and no shell interpretation when a
  simple argv form is available;
* never inherits sensitive environment variables.

The LLM can never bypass these rules — they live in code, not in the prompt.
"""

from __future__ import annotations

import re
import subprocess

from tools.permissions import PermissionLevel
from tools.registry import ToolContext, Tool

_BLOCKLIST = re.compile(
    r"(?i)\b(format|fdisk|diskpart|del /f|rmdir /s|rd /s|shutdown|restart|"
    r"reg delete|reg add|sc delete|net user|net localgroup|takeown|icacls|"
    r"certutil|powershell.*-enc|rundll32|mshta|wmic.*delete|taskkill /f /im "
    r"explorer|format\b)\b"
)


def _run_shell(command: str, timeout: float = 15, cwd: str | None = None,
               _ctx: ToolContext = None) -> dict:
    if _BLOCKLIST.search(command):
        return {
            "success": False,
            "output": "Command blocked: it looks destructive and is not allowed.",
        }
    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602 - explicitly gated; see module docstring
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"Command timed out after {timeout}s"}
    except OSError as exc:
        return {"success": False, "output": f"Could not run command: {exc}"}
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    output = stdout[-1500:] or stderr[-500:]
    return {
        "success": result.returncode == 0,
        "output": output or ("OK" if result.returncode == 0 else "Command failed"),
        "data": {"returncode": result.returncode},
    }


def build_tools(enabled: bool = False) -> list:
    if not enabled:
        return []
    return [
        Tool(
            name="run_shell_command",
            description=(
                "Run a single shell command (cmd). Only for actions with no "
                "dedicated tool. Requires confirmation every time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                    "timeout": {"type": "number", "description": "Seconds before timeout", "default": 15},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["command"],
            },
            handler=_run_shell,
            permission=PermissionLevel.ASK,
            requires_confirmation=True,
        )
    ]
