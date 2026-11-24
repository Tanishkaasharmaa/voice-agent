"""ModelManager — model detection, informed install prompts, and pulls.

Never downloads silently. Before pulling, it reports what is known:
model name, size (from Ollama /api/show or repo metadata when available),
installed/loaded state, available system RAM, and a recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from setup.installer import (
    InstallError,
    download_file,
    pull_ollama_model,
    verify_ollama_model,
)
from utils.system import ResourceMonitor, detect_hardware

LOCAL_MODELS = {
    "silero_vad.onnx": {
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "purpose": "Voice activity detection (speech start/end)",
        "approx_mb": 2,
    },
    "kokoro-v1.0.onnx": {
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "purpose": "Kokoro text-to-speech model",
        "approx_mb": 330,
    },
    "voices-v1.0.bin": {
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "purpose": "Kokoro voice definitions",
        "approx_mb": 130,
    },
}


def describe_local_models(models_dir: Path) -> list[dict]:
    entries = []
    for name, meta in LOCAL_MODELS.items():
        path = models_dir / name
        entries.append(
            {
                "name": name,
                "purpose": meta["purpose"],
                "approx_mb": meta["approx_mb"],
                "installed": path.exists(),
                "size_mb": round(path.stat().st_size / (1024**2), 1) if path.exists() else 0,
            }
        )
    return entries


def describe_ollama_model(client, name: str) -> dict:
    """Gather everything the setup dialog should show before a pull."""
    installed = False
    size = None
    loaded = None
    try:
        installed = client.model_installed(name)
        if installed:
            info = client.model_info(name)
            if info:
                size = info.get("size")
                loaded = info.get("loading") or None
    except Exception:
        pass
    hw = detect_hardware()
    return {
        "name": name,
        "installed": installed,
        "size_bytes": size,
        "size_gb": round(size / (1024**3), 2) if size else None,
        "loaded": loaded,
        "ram_total_gb": round(hw.ram_total_gb, 1),
        "ram_available_gb": round(ResourceMonitor.sample().ram_total_gb
                                  - ResourceMonitor.sample().ram_used_gb, 1),
        "recommendation": (
            "Installed — no download needed."
            if installed
            else "Requires Internet to download. Available RAM should be checked "
            "before pulling large models."
        ),
    }


class ModelManager:
    def __init__(self, models_dir: Path, ollama_host: str = "http://127.0.0.1:11434") -> None:
        self.models_dir = models_dir
        self.ollama_host = ollama_host

    # ------------------------------------------------------------ local files

    def install_local_model(self, name: str, progress: Callable) -> None:
        meta = LOCAL_MODELS.get(name)
        if meta is None:
            raise InstallError(f"Unknown local model: {name}")
        download_file(meta["url"], self.models_dir / name, progress)

    # ------------------------------------------------------------ ollama

    def pull(self, name: str, progress: Callable) -> None:
        pull_ollama_model(self.ollama_host, name, progress)
        if not verify_ollama_model(self.ollama_host, name):
            raise InstallError("Model pull reported success but the model is not listed")
