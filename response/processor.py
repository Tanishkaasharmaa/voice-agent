"""ResponseProcessor — turns raw LLM output into speech-friendly sentences.

Removes markdown, code blocks, JSON, tool metadata and internal reasoning;
normalizes punctuation; and splits the text into sentence-level chunks so
TTS can start speaking before the model finishes.
"""

from __future__ import annotations

import json
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9“\"'])")
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MARKDOWN = re.compile(r"[*_~#>`|]+")
_JSON_OBJECT = re.compile(r"\{[^{}]*\}")
_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]")
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_LIST_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MULTISPACE = re.compile(r"[ \t]{2,}")
_URL = re.compile(r"https?://\S+")

# Phrases the agent may accidentally emit that should never be spoken.
_SKIP_PHRASES = (
    "tool call", "tool_calls", "function call", "<tool_call>", "</tool_call>",
    "action taken", "I'll execute", "let me run the tool",
)


class ResponseProcessor:
    def __init__(self, max_chunk_chars: int = 300) -> None:
        self.max_chunk_chars = max_chunk_chars

    def clean(self, text: str) -> str:
        """Strip everything that shouldn't be spoken."""
        if not text:
            return ""
        out = text
        out = _CODE_FENCE.sub(" ", out)          # ``` ... ```
        out = _INLINE_CODE.sub(r"\1", out)        # keep inline `text`, drop ticks
        out = _HEADING.sub("", out)
        out = _LIST_BULLET.sub("", out)
        out = _MARKDOWN.sub("", out)             # *, _, ~, #, >, |
        out = _JSON_OBJECT.sub(" ", out)         # {"tool": ...} etc.
        out = _JSON_ARRAY.sub(" ", out)
        out = _URL.sub("link", out)
        out = out.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
        out = _MULTISPACE.sub(" ", out)
        out = out.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
        out = out.strip(" -–—:;")
        return out.strip()

    def is_spoken_content(self, text: str) -> bool:
        lowered = text.lower()
        if not text.strip():
            return False
        if any(phrase in lowered for phrase in _SKIP_PHRASES):
            return False
        if text.strip().startswith(("{", "[")):
            return False
        return True

    def split_sentences(self, text: str) -> list[str]:
        """Split into natural sentence chunks (never mid-word)."""
        cleaned = self.clean(text)
        if not cleaned:
            return []
        parts = _SENTENCE_SPLIT.split(cleaned)
        chunks: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.max_chunk_chars:
                chunks.append(part)
                continue
            # Split long sentences at clause boundaries.
            clause_split = re.split(r"(?<=[,;:])\s+", part)
            current = ""
            for piece in clause_split:
                if current and len(current) + len(piece) + 1 > self.max_chunk_chars:
                    chunks.append(current.strip())
                    current = piece
                else:
                    current = f"{current} {piece}".strip()
            if current:
                chunks.append(current.strip())
        return chunks

    def brief(self, tool_name: str, success: bool) -> str:
        """Spoken summary of a tool execution — never speak raw JSON."""
        if success:
            return _BRIEF_SUCCESS.get(tool_name, "Done.")
        return _BRIEF_FAILURE.get(tool_name, "That didn't work.")


_BRIEF_SUCCESS = {
    "open_application": "It's open.",
    "close_application": "Closed.",
    "launch_application": "It's open.",
    "open_url": "Opening that now.",
    "write_clipboard": "Copied to the clipboard.",
    "type_text": "Done.",
    "press_key": "Done.",
    "hotkey": "Done.",
    "create_note": "Note saved.",
    "update_note": "Note updated.",
    "delete_note": "Note deleted.",
    "save_memory": "I'll remember that.",
    "get_current_time": "",
    "get_current_date": "",
    "get_system_info": "",
}

_BRIEF_FAILURE = {
    "open_application": "I couldn't open that.",
    "close_application": "I couldn't close that.",
    "open_url": "I couldn't open that link.",
    "write_clipboard": "I couldn't copy that.",
    "create_note": "I couldn't save the note.",
    "save_memory": "I couldn't save that.",
}
