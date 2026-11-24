"""Dependency detection — run on EVERY startup.

These checks perform real detection; the install_history table in SQLite is
informational only and is never treated as proof of installation.
"""

from __future__ import annotations

import importlib
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ollama.client import OllamaError

# Categories
CORE = "core"
AI_CORE = "ai_core"
VOICE_INPUT = "voice_input"
VOICE_OUTPUT = "voice_output"
OPTIONAL = "optional"

# Provider id -> display label (mirrors ollama/providers.py; duplicated as a
# tiny static map so checks stay import-light and dependency-free).
PROVIDER_LABELS = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "groq": "Groq",
    "openrouter": "OpenRouter",
}


@dataclass
class DependencyStatus:
    name: str
    category: str
    present: bool
    version: str = ""
    message: str = ""
    installable: bool = True
    internet_required: bool = False
    optional: bool = False


def _import_version(module: str) -> str:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", "")
    except Exception:
        return ""


def _module_present(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ checks


def check_python() -> DependencyStatus:
    return DependencyStatus(
        name="Python runtime",
        category=CORE,
        present=True,
        version=sys.version.split()[0],
        message="Running",
    )


def check_sqlite() -> DependencyStatus:
    present = True
    version = ""
    try:
        version = sqlite3.sqlite_version
    except Exception:
        present = False
    return DependencyStatus(
        name="SQLite", category=CORE, present=present, version=version,
        message="OK" if present else "Missing",
    )


def check_pyside6() -> DependencyStatus:
    present = _module_present("PySide6")
    return DependencyStatus(
        name="PySide6", category=CORE, present=present,
        version=_import_version("PySide6"),
        message="" if present else "Required for the user interface",
        installable=True,
        internet_required=True,
    )


def check_audio_input() -> DependencyStatus:
    if not _module_present("sounddevice"):
        return DependencyStatus(
            name="Audio input", category=CORE, present=False,
            message="sounddevice not installed", installable=True, internet_required=True,
        )
    try:
        import sounddevice as sd  # type: ignore

        devices = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        present = len(devices) > 0
        return DependencyStatus(
            name="Audio input", category=CORE, present=present,
            message="Microphone detected" if present else "No microphone found",
        )
    except Exception as exc:
        return DependencyStatus(name="Audio input", category=CORE, present=False, message=str(exc))


def check_audio_output() -> DependencyStatus:
    if not _module_present("sounddevice"):
        return DependencyStatus(
            name="Audio output", category=CORE, present=False,
            message="sounddevice not installed", installable=True, internet_required=True,
        )
    try:
        import sounddevice as sd  # type: ignore

        devices = [d for d in sd.query_devices() if d.get("max_output_channels", 0) > 0]
        present = len(devices) > 0
        return DependencyStatus(
            name="Audio output", category=CORE, present=present,
            message="Speaker detected" if present else "No output device found",
        )
    except Exception as exc:
        return DependencyStatus(name="Audio output", category=CORE, present=False, message=str(exc))


def check_ollama(host: str = "http://127.0.0.1:11434") -> DependencyStatus:
    try:
        import requests

        resp = requests.get(f"{host}/api/version", timeout=2.5)
        present = resp.status_code == 200
        version = resp.json().get("version", "") if present else ""
        message = "Running" if present else "Ollama not running at " + host
    except Exception:
        present = False
        version = ""
        message = "Ollama not detected"
    return DependencyStatus(
        name="Ollama", category=AI_CORE, present=present, version=version,
        message=message,
        installable=True,
        internet_required=True,  # installer download requires Internet
    )


def check_ollama_model(client, model_name: str, required: bool = True) -> DependencyStatus:
    """Check a model is present. When Ollama itself is unreachable the model
    is reported as *unknown* rather than "not installed" — a user who pulled
    it earlier should not be told it is missing just because Ollama is down."""
    present = False
    version = ""
    message = "Model not installed"
    try:
        present = client.model_installed(model_name)
        info = client.model_info(model_name) if present else {}
        size_gb = info.get("size", 0) / (1024**3) if info else 0
        version = f"{size_gb:.1f} GB" if size_gb else ""
        message = "Installed" if present else "Model not installed"
    except OllamaError:
        message = "Ollama not reachable — model presence unknown"
    except Exception:
        present = False
    return DependencyStatus(
        name=model_name, category=AI_CORE, present=present, version=version,
        message=message,
        installable=True,
        internet_required=True,
        optional=not required,
    )


def check_ollama_cloud(client, model_name: str) -> DependencyStatus:
    """Probe the chosen cloud model to verify sign-in and availability.

    ``client.probe_cloud`` sends a 1-token chat: a 401 means the user is not
    signed in to their Ollama account (fix: ``ollama signin``); any other
    failure means the cloud model is unreachable. Never disables voice — it
    is reported for information and the app falls back to the local model.
    """
    name = f"Cloud model ({model_name})"
    try:
        status = client.probe_cloud(model_name)
    except Exception:  # noqa: BLE001 - a broken probe must not break startup
        status = "error"
    if status == "ok":
        return DependencyStatus(
            name=name, category=AI_CORE, present=True, version="",
            message="Signed in — cloud model ready", installable=False, optional=True,
        )
    if status == "unauthorized":
        return DependencyStatus(
            name=name, category=AI_CORE, present=False, version="",
            message="Not signed in — run 'ollama signin' in a terminal",
            installable=False, optional=True,
        )
    return DependencyStatus(
        name=name, category=AI_CORE, present=False, version="",
        message="Cloud model unavailable — run 'ollama signin' and check internet",
        installable=False, optional=True,
    )


def check_faster_whisper() -> DependencyStatus:
    present = _module_present("faster_whisper")
    return DependencyStatus(
        name="Whisper (faster-whisper)", category=VOICE_INPUT, present=present,
        version=_import_version("faster_whisper"),
        message="" if present else "Required for speech-to-text",
        installable=True,
        internet_required=True,
    )


def check_whisper_model(models_dir: Path | None = None, model: str = "small") -> DependencyStatus:
    """Check the local models directory for the whisper-<size> snapshot."""
    from stt.whisper_engine import local_model_dir

    path = local_model_dir(models_dir, model) if models_dir is not None else None
    present = path is not None and (path / "model.bin").exists()
    return DependencyStatus(
        name=f"Whisper model ({model})", category=VOICE_INPUT, present=present,
        message="Installed" if present
        else "Downloaded to the models folder (needs Internet once)",
        installable=True,
        internet_required=True,
    )


def check_onnxruntime() -> DependencyStatus:
    present = _module_present("onnxruntime")
    return DependencyStatus(
        name="ONNX Runtime", category=VOICE_INPUT, present=present,
        version=_import_version("onnxruntime"),
        message="" if present else "Required for VAD and Kokoro",
        installable=True,
        internet_required=True,
    )


def check_silero_model(models_dir: Path) -> DependencyStatus:
    present = (models_dir / "silero_vad.onnx").exists()
    return DependencyStatus(
        name="Silero VAD model", category=VOICE_INPUT, present=present,
        message="OK" if present else "Voice activity detection needs this model",
        installable=True,
        internet_required=True,
    )


def check_kokoro(models_dir: Path) -> DependencyStatus:
    onnx = (models_dir / "kokoro-v1.0.onnx").exists()
    voices = (models_dir / "voices-v1.0.bin").exists()
    # The engine needs a backend package too — model files alone are not
    # enough (KokoroEngine falls back silently without one).
    backend = _module_present("kokoro_onnx") or _module_present("kokoro")
    present = onnx and voices and backend
    if present:
        message = "OK"
    elif onnx and voices:
        message = "TTS backend (kokoro-onnx) not installed"
    else:
        message = "TTS model files missing"
    return DependencyStatus(
        name="Kokoro TTS", category=VOICE_OUTPUT, present=present,
        message=message,
        installable=True,
        internet_required=True,
    )


def check_piper() -> DependencyStatus:
    present = _module_present("piper")
    return DependencyStatus(
        name="Piper (optional)", category=VOICE_OUTPUT, present=present,
        message="Optional TTS fallback",
        installable=True,
        internet_required=True,
        optional=True,
    )


def check_pynput() -> DependencyStatus:
    present = _module_present("pynput")
    return DependencyStatus(
        name="Global hotkeys (pynput)", category=CORE, present=present,
        version=_import_version("pynput"),
        message="" if present else "Required for voice hotkeys",
        installable=True,
        internet_required=True,
    )


def check_pywin32() -> DependencyStatus:
    present = _module_present("win32api")
    return DependencyStatus(
        name="Windows automation (pywin32)", category=OPTIONAL, present=present,
        message="Enables window/clipboard tools",
        installable=True,
        internet_required=True,
        optional=True,
    )


def check_psutil() -> DependencyStatus:
    present = _module_present("psutil")
    return DependencyStatus(
        name="Resource monitor (psutil)", category=OPTIONAL, present=present,
        version=_import_version("psutil"),
        installable=True,
        internet_required=True,
        optional=True,
    )


def check_requests() -> DependencyStatus:
    present = _module_present("requests")
    return DependencyStatus(
        name="HTTP client (requests)", category=CORE, present=present,
        installable=True,
        internet_required=True,
    )


# ------------------------------------------------------------------ runner


def run_all_checks(settings=None, client=None, cloud_model: str | None = None) -> list[DependencyStatus]:
    """Run every startup check and return statuses.

    Rows are filtered by the chosen setup (``settings.llm.provider`` and
    ``settings.llm.connectivity``):

    * a non-Ollama provider (OpenAI/Groq/OpenRouter) needs no local Ollama
      server or local model — both rows report as satisfied;
    * "online" mode with the Ollama provider needs no local model either;
    * the cloud-model probe only runs when a cloud model is actually in
      play (hybrid/online with the Ollama provider).
    """
    checks: list[Callable[[], DependencyStatus]] = [
        check_python, check_sqlite, check_pyside6, check_requests,
        check_pynput, check_audio_input, check_audio_output,
        check_faster_whisper, check_onnxruntime, check_pywin32, check_psutil,
        check_piper,
    ]
    results: list[DependencyStatus] = [check() for check in checks]

    models_dir = None
    if settings is not None:
        from utils.platform import ensure_directories

        models_dir = ensure_directories()["models"]
    else:
        models_dir = Path.home() / "AppData" / "Local" / "LocalVoiceAgent" / "models"

    results.append(check_whisper_model(
        models_dir=models_dir, model=settings.whisper_model if settings else "small",
    ))
    results.append(check_silero_model(models_dir))
    results.append(check_kokoro(models_dir))

    provider = settings.provider if settings is not None else "ollama"
    mode = settings.connectivity_mode if settings is not None else "hybrid"
    local_required = settings.local_model_required if settings is not None else True

    if provider != "ollama":
        # A hosted OpenAI-compatible provider serves chat; the local Ollama
        # server/model are simply not part of this setup.
        label = PROVIDER_LABELS.get(provider, provider)
        results.append(DependencyStatus(
            name="Ollama", category=AI_CORE, present=True, version="",
            message=f"Not required — using {label}",
            installable=False,
        ))
        results.append(DependencyStatus(
            name=settings.local_model if settings else "qwen3.5:4b",
            category=AI_CORE, present=True, version="",
            message=f"Not required — using {label}",
            installable=False,
        ))
        return results

    if cloud_model:
        # Cloud mode active — the local Ollama server is not required: the
        # chosen cloud model is served by Ollama's cloud. Keep the row
        # visible but satisfied so the Setup Check never flags it as missing.
        results.append(DependencyStatus(
            name="Ollama", category=AI_CORE, present=True,
            version="", message="Cloud mode active — local server not required",
            installable=False,
        ))
    elif settings is not None:
        results.append(check_ollama(settings.llm_host))
    else:
        results.append(check_ollama())

    # The chosen local model is required whenever it can actually be used:
    # offline mode and hybrid mode (fallback). Pure online mode never runs
    # it, so no download is demanded.
    if not local_required:
        results.append(DependencyStatus(
            name=settings.local_model if settings else "qwen3.5:4b",
            category=AI_CORE, present=True, version="",
            message="Not required — online mode uses the cloud model",
            installable=False,
        ))
    elif client is not None:
        local = settings.local_model if settings else "qwen3.5:4b"
        results.append(check_ollama_model(client, local, required=True))
    else:
        results.append(check_ollama_model(
            None, settings.local_model if settings else "qwen3.5:4b"))

    if cloud_model:
        # Cloud mode active — verify the user is signed in and the model is
        # reachable. Reported as a missing item when sign-in fails; it never
        # disables voice (the app falls back to the local model).
        if client is not None:
            results.append(check_ollama_cloud(client, cloud_model))
        else:
            results.append(DependencyStatus(
                name=f"Cloud model ({cloud_model})", category=AI_CORE, present=True,
                version="", message="Ollama cloud — sign-in not verified",
                installable=False, optional=True,
            ))

    return results


def missing_statuses(statuses: list[DependencyStatus]) -> list[DependencyStatus]:
    return [s for s in statuses if not s.present]
