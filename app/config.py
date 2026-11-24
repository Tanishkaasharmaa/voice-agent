"""Central configuration for LocalVoiceAgent.

Settings live in a JSON file under %APPDATA%/LocalVoiceAgent/config.json.
Defaults come from ``config/defaults.json`` (the LOW_RESOURCE_CPU profile).
Nothing here is hardcoded to a username, absolute path, or device.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from utils.platform import app_data_dir, ensure_directories
from utils.system import detect_hardware, recommended_worker_threads

PROFILE = "LOW_RESOURCE_CPU"

# Suggested names for the assistant (editable — the user can pick or type
# their own). Shared by the setup wizard and the Settings dialog.
AGENT_NAME_SUGGESTIONS = [
    "Luna", "Jarvis", "Friday", "Nova", "Atlas", "Echo", "Sage", "Orion",
]


def _defaults() -> dict[str, Any]:
    here = Path(__file__).resolve().parent.parent / "config" / "defaults.json"
    try:
        return json.loads(here.read_text(encoding="utf-8"))
    except OSError:  # pragma: no cover - packaged executable fallback
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Settings:
    """Thread-safe-ish access to the merged configuration.

    Reads are cheap dict lookups; writes go through ``update`` which persists
    atomically (write temp file, then replace).
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            dirs = ensure_directories()
            self.path = dirs["config"] / "config.json"
        else:
            self.path = Path(path)
        self._data: dict[str, Any] = _deep_merge(_defaults(), {})
        self._apply_hardware_defaults()
        self.load()

    # ------------------------------------------------------------------ core

    def _apply_hardware_defaults(self) -> None:
        """Auto-tune CPU/RAM-aware settings on first run only (kept if the
        user later overrides them explicitly)."""
        hw = detect_hardware()
        threads = recommended_worker_threads(hw.ram_total_gb)
        p = self._data.get("performance", {})
        # Only fill when unset; don't clobber user choices on later starts.
        if not p.get("whisper_threads"):
            p["whisper_threads"] = max(1, min(threads, 4))
        if not p.get("max_worker_threads"):
            p["max_worker_threads"] = threads
        vi = self._data.get("voice_input", {})
        if not vi.get("cpu_threads"):
            vi["cpu_threads"] = max(1, min(threads, 4))
        self._data["hardware"] = {
            "cpu_name": hw.cpu_name,
            "cpu_threads": hw.cpu_threads,
            "ram_total_gb": round(hw.ram_total_gb, 1),
            "gpu": hw.gpu,
        }

    def load(self) -> None:
        if self.path.exists():
            try:
                user = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(user, dict):
                    self._data = _deep_merge(self._data, user)
            except (OSError, ValueError):
                pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # --------------------------------------------------------------- access

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-path lookup, e.g. ``settings.get("llm.local_model")``."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def update(self, mapping: dict[str, Any]) -> None:
        """Merge a nested dict (from the Settings UI) and persist."""
        self._data = _deep_merge(self._data, mapping)
        self.save()

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def reset_to_defaults(self) -> None:
        self._data = _deep_merge(_defaults(), {})
        self._apply_hardware_defaults()
        self.save()

    # ---------------------------------------------------------- properties

    @property
    def llm_host(self) -> str:
        return self.get("llm.host", "http://127.0.0.1:11434")

    # ------------------------------------------------------------- providers

    @property
    def provider(self) -> str:
        """Which LLM backend serves chat: "ollama", "openai", "groq" or
        "openrouter" (see ollama/providers.py)."""
        return self.get("llm.provider", "ollama") or "ollama"

    @property
    def connectivity_mode(self) -> str:
        """How the agent may use the internet for models:
        "online" (cloud/provider only), "offline" (local only), or
        "hybrid" (cloud when online, local model as fallback)."""
        mode = self.get("llm.connectivity", "hybrid")
        return mode if mode in ("online", "offline", "hybrid") else "hybrid"

    @property
    def provider_model(self) -> str:
        """The model served by a non-Ollama provider (e.g. gpt-4o-mini)."""
        return self.get("llm.provider_model", "") or ""

    def api_key(self, provider: str) -> str:
        return str(self.get(f"llm.api_keys.{provider}", "") or "")

    def set_api_key(self, provider: str, value: str) -> None:
        self.set(f"llm.api_keys.{provider}", value.strip())

    @property
    def provider_needs_api_key(self) -> bool:
        from ollama.providers import PROVIDERS

        return bool(PROVIDERS.get(self.provider, {}).get("needs_api_key"))

    @property
    def local_model_required(self) -> bool:
        """True when a local Ollama model must be installed: offline mode,
        hybrid mode (it is the fallback), and any Ollama-provider setup.
        Pure online setups never execute a local model."""
        if self.provider == "ollama":
            return self.connectivity_mode != "online"
        return self.connectivity_mode == "hybrid"

    @property
    def local_model(self) -> str:
        """The local Ollama model used when offline (or cloud is disabled).
        At least one local model is required as the offline fallback."""
        return self.get("llm.local_model", "qwen3.5:4b")

    @property
    def local_model_choices(self) -> list[str]:
        return self.get("llm.local_model_choices", [
            "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b",
            "qwen3.5:27b", "gemma4:e2b", "gemma4:e4b", "gemma4:12b",
            "nemotron-3-nano:4b",
        ])

    @property
    def use_cloud_models(self) -> bool:
        """Whether to prefer an Ollama cloud model when online."""
        return bool(self.get("llm.use_cloud_models", True))

    @property
    def llm_timeout(self) -> float:
        """Read timeout in seconds for Ollama API requests."""
        return float(self.get("llm.timeout", 120))

    @property
    def cloud_model(self) -> str:
        """The chosen Ollama cloud model (e.g. ``deepseek-v4-flash:cloud``)."""
        return self.get("llm.cloud_model", "deepseek-v4-flash:cloud")

    @property
    def cloud_model_choices(self) -> list[str]:
        return self.get("llm.cloud_model_choices", [
            "gemma4:cloud", "gemma4:31b-cloud", "deepseek-v4-flash:cloud",
            "glm-5.2:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud",
            "minimax-m3:cloud", "qwen3.5:cloud", "qwen3.5:397b-cloud",
            "nemotron-3-nano:30b-cloud",
        ])

    @property
    def agent_name(self) -> str:
        """What the assistant is called (set during setup, changeable in
        Settings). Used in the agent's system prompt."""
        return self.get("general.agent_name", "Luna") or "Luna"

    @property
    def user_name(self) -> str:
        """What the assistant should call the user (may be empty)."""
        return self.get("general.user_name", "") or ""

    @property
    def whisper_model(self) -> str:
        return self.get("voice_input.whisper_model", "small")

    @property
    def silence_duration(self) -> float:
        return float(self.get("voice_input.silence_duration", 0.8))

    @property
    def minimum_speech_duration(self) -> float:
        return float(self.get("voice_input.minimum_speech_duration", 0.3))

    @property
    def maximum_recording_duration(self) -> float:
        return float(self.get("voice_input.maximum_recording_duration", 60))

    @property
    def whisper_threads(self) -> int:
        return int(self.get("performance.whisper_threads", 0) or 0)

    @property
    def tts_engine(self) -> str:
        return self.get("voice_output.tts_engine", "kokoro")

    @property
    def is_debug_recording_enabled(self) -> bool:
        return bool(self.get("debug.save_recordings", False))


def config_path() -> Path:
    return app_data_dir() / "config" / "config.json"
