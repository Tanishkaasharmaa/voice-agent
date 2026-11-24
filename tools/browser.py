"""Browser tool — open a URL with the system default browser.

The URL itself is user-provided; this only hands it to the default handler.
No browser automation, no remote services.
"""

from __future__ import annotations

import webbrowser

from tools.registry import ToolContext, Tool


def _open_url(url: str, _ctx: ToolContext) -> dict:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        # Allow bare domains and mailto links; reject anything exotic.
        if "." not in url or " " in url:
            return {"success": False, "output": "Invalid URL"}
        url = f"https://{url}"
    try:
        webbrowser.open(url)
        return {"success": True, "output": "Opening the link in your browser."}
    except Exception as exc:
        return {"success": False, "output": f"Could not open the link: {exc}"}


def build_tools() -> list:
    return [
        Tool(
            name="open_url",
            description="Open a URL in the user's default web browser.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "http(s) URL or domain"}},
                "required": ["url"],
            },
            handler=_open_url,
        )
    ]
