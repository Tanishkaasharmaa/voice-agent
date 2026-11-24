"""STT model management.

faster-whisper weights live in the app's models folder as
``whisper-<size>`` snapshots (downloaded once at setup time with consent).
This module reports availability/status so the setup dialog can tell the
user what will happen before anything is downloaded.
"""

from __future__ import annotations

from stt.whisper_engine import SUPPORTED_MODELS

# Rough on-disk sizes for the quantized models (informational only — the
# actual download reports its own progress).
_APPROX_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 485,
    "medium": 1500,
}


def list_models() -> list[dict]:
    return [
        {
            "name": name,
            "approx_size_mb": _APPROX_SIZE_MB.get(name),
            "recommended": name == "small",
        }
        for name in SUPPORTED_MODELS
    ]


def describe_model(name: str) -> dict | None:
    for entry in list_models():
        if entry["name"] == name:
            return entry
    return None
