"""System / keyboard / memory tools."""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import time

from tools.registry import ToolContext, Tool


# --------------------------------------------------------------- SendInput

_VK = {
    "enter": 0x0D, "tab": 0x09, "space": 0x20, "esc": 0x1B, "escape": 0x1B,
    "backspace": 0x08, "delete": 0x2E, "home": 0x24, "end": 0x23,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "pageup": 0x21, "pagedown": 0x22,
}


def _send_key(vk: int) -> None:
    user32 = ctypes.windll.user32
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("ki",)
        _fields_ = [("type", ctypes.c_ulong), ("ki", KEYBDINPUT)]

    down = INPUT()
    down.type = INPUT_KEYBOARD
    down.wVk = vk
    up = INPUT()
    up.type = INPUT_KEYBOARD
    up.wVk = vk
    up.dwFlags = KEYEVENTF_KEYUP
    user32.SendInput(2, ctypes.byref(down), ctypes.sizeof(INPUT))
    time.sleep(0.03)
    user32.SendInput(2, ctypes.byref(up), ctypes.sizeof(INPUT))


def _type_text(text: str, _ctx: ToolContext = None) -> dict:
    if len(text) > 2000:
        return {"success": False, "output": "Text too long to type"}
    try:
        for char in text:
            ctypes.windll.user32.keybd_event(0, ord(char), 0, 0)
        return {"success": True, "output": "Typed."}
    except Exception as exc:
        return {"success": False, "output": f"Could not type: {exc}"}


def _press_key(key: str, _ctx: ToolContext = None) -> dict:
    vk = _VK.get(key.lower())
    if vk is None and len(key) == 1:
        vk = ord(key.upper())
    if vk is None:
        return {"success": False, "output": f"Unknown key: {key}"}
    try:
        _send_key(vk)
        return {"success": True, "output": f"Pressed {key}."}
    except Exception as exc:
        return {"success": False, "output": f"Could not press key: {exc}"}


# ---------------------------------------------------------------- time/info


def _get_current_time(_ctx: ToolContext = None) -> dict:
    now = dt.datetime.now()
    return {
        "success": True,
        "output": now.strftime("%I:%M %p").lstrip("0"),
        "data": {"time": now.isoformat()},
    }


def _get_current_date(_ctx: ToolContext = None) -> dict:
    now = dt.date.today()
    return {"success": True, "output": now.strftime("%A, %B %d, %Y"), "data": {"date": now.isoformat()}}


def _get_system_info(_ctx: ToolContext = None) -> dict:
    try:
        from utils.system import detect_hardware

        hw = detect_hardware()
        info = {
            "os": hw.os_name,
            "cpu": hw.cpu_name,
            "cpu_threads": hw.cpu_threads,
            "ram_gb": round(hw.ram_total_gb, 1),
            "gpu": hw.gpu,
        }
        return {"success": True, "output": json.dumps(info), "data": info}
    except Exception as exc:
        return {"success": False, "output": f"Could not read system info: {exc}"}


# ------------------------------------------------------------------ memory


def _save_memory(content: str, _ctx: ToolContext = None) -> dict:
    ctx: ToolContext = _ctx
    if ctx.memory is None:
        return {"success": False, "output": "Memory system unavailable"}
    memory_id = ctx.memory.save_memory(content, kind="memory")
    return {"success": True, "output": "I'll remember that.", "data": {"memory_id": memory_id}}


def _search_memory(query: str, _ctx: ToolContext = None) -> dict:
    ctx: ToolContext = _ctx
    if ctx.memory is None:
        return {"success": False, "output": "Memory system unavailable"}
    rows = ctx.memory.search_memory(query, limit=10)
    if not rows:
        return {"success": True, "output": "No matching memories.", "data": {"memories": []}}
    previews = [{"id": r["id"], "text": r["content"][:200]} for r in rows]
    return {"success": True, "output": f"Found {len(rows)} memory item(s).", "data": {"memories": previews}}


def build_tools() -> list:
    return [
        Tool(name="get_current_time", description="Return the current local time.",
             parameters={"type": "object", "properties": {}}, handler=_get_current_time),
        Tool(name="get_current_date", description="Return today's date.",
             parameters={"type": "object", "properties": {}}, handler=_get_current_date),
        Tool(name="get_system_info", description="Return OS, CPU, and RAM information.",
             parameters={"type": "object", "properties": {}}, handler=_get_system_info),
        Tool(name="type_text", description="Type text into the focused window.",
             parameters={"type": "object", "properties": {"text": {"type": "string"}},
                         "required": ["text"]}, handler=_type_text),
        Tool(name="press_key",
             description="Press a single key (e.g. enter, tab, space, esc, up, down).",
             parameters={"type": "object", "properties": {"key": {"type": "string"}},
                         "required": ["key"]}, handler=_press_key),
        Tool(name="save_memory",
             description="Save a useful fact the user asked to remember.",
             parameters={"type": "object", "properties": {"content": {"type": "string"}},
                         "required": ["content"]}, handler=_save_memory),
        Tool(name="search_memory",
             description="Search previously saved memories.",
             parameters={"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"]}, handler=_search_memory),
    ]
