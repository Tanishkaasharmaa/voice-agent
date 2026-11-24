"""Clipboard tools — read/write text via win32clipboard."""

from __future__ import annotations

from tools.registry import ToolContext, Tool


def _read_clipboard(_ctx: ToolContext) -> dict:
    import win32clipboard  # type: ignore
    import win32con  # type: ignore

    try:
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return {"success": True, "output": "Clipboard read.", "data": {"text": data}}
            return {"success": True, "output": "Clipboard is empty.", "data": {"text": ""}}
        finally:
            win32clipboard.CloseClipboard()
    except Exception as exc:
        return {"success": False, "output": f"Could not read clipboard: {exc}"}


def _write_clipboard(text: str, _ctx: ToolContext) -> dict:
    import win32clipboard  # type: ignore
    import win32con  # type: ignore

    if not isinstance(text, str):
        return {"success": False, "output": "Clipboard content must be text"}
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
        finally:
            win32clipboard.CloseClipboard()
        return {"success": True, "output": "Text copied to clipboard."}
    except Exception as exc:
        return {"success": False, "output": f"Could not write clipboard: {exc}"}


def build_tools() -> list:
    return [
        Tool(
            name="read_clipboard",
            description="Read the current text from the Windows clipboard.",
            parameters={"type": "object", "properties": {}},
            handler=_read_clipboard,
        ),
        Tool(
            name="write_clipboard",
            description="Write text to the Windows clipboard.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=_write_clipboard,
        ),
    ]
