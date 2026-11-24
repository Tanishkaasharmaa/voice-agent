"""DependencyManager — orchestrates startup dependency handling.

Flow per missing dependency (every startup):
1. check -> missing?
2. ask the user (YES/NO) with purpose, category, Internet requirement
3. YES  -> install with progress, then RE-CHECK
4. NO   -> skip; the dependent feature is disabled and clearly explained
5. record the outcome in the install_history table (informational)

A dependency skipped during one startup is not re-prompted *in that same
startup*; the next startup checks it again from scratch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from setup import dependency_checks as checks
from setup.dependency_checks import DependencyStatus
from setup.installer import (
    InstallError, package_installed, pip_install, start_ollama,
    download_ollama_installer,
)
from setup.model_manager import ModelManager

def _approx_model_mb(name: str) -> float:
    """Rough download size (MB) for a model tag, e.g. "qwen3.5:4b" -> 4 GB."""
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
    gb = float(match.group(1)) if match else 4.0
    return gb * 1024


# pip package name -> (import module, category)
_PIP_PACKAGES = {
    "PySide6": ("PySide6", checks.CORE),
    "pynput": ("pynput", checks.CORE),
    "sounddevice": ("sounddevice", checks.CORE),
    "requests": ("requests", checks.CORE),
    "faster-whisper": ("faster_whisper", checks.VOICE_INPUT),
    "onnxruntime": ("onnxruntime", checks.VOICE_INPUT),
    "pywin32": ("win32api", checks.OPTIONAL),
    "psutil": ("psutil", checks.OPTIONAL),
}


@dataclass
class InstallPlan:
    """What the user is asked to approve, and what runs on YES."""
    name: str
    purpose: str
    category: str
    internet_required: bool
    approx_size_mb: float | None = None
    optional: bool = False
    install: Callable[[Callable], None] | None = None
    verify: Callable[[], bool] | None = None


class DependencyManager:
    def __init__(self, settings=None, client=None, memory=None,
                 models_dir=None, ollama_host: str | None = None,
                 cloud_model: str | None = None) -> None:
        self.settings = settings
        self.client = client
        self.memory = memory
        self.models_dir = models_dir
        self.cloud_model = cloud_model
        self.ollama_host = ollama_host or (settings.llm_host if settings else "http://127.0.0.1:11434")
        self.model_manager = ModelManager(self.models_dir, self.ollama_host) if models_dir else None

        # UI callbacks (set by the Application).
        self.prompt_yes_no: Callable[[str, str, str, bool], bool] | None = None
        self.show_progress: Callable[[str, Callable], None] | None = None
        self.on_status_update: Callable[[list[DependencyStatus]], None] | None = None
        # True while the first-run wizard drives setup: the user already
        # opted into installing the full local stack, so asking Yes/No for
        # every component would only repeat the decision they just made.
        self.auto_approve = False

        self._skip_this_startup: set[str] = set()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- checks

    def check_all(self) -> list[DependencyStatus]:
        statuses = checks.run_all_checks(self.settings, self.client, self.cloud_model)
        if self.on_status_update:
            self.on_status_update(statuses)
        return statuses

    # --------------------------------------------------------------- prompt

    def _ask(self, name: str, purpose: str, category: str, internet: bool,
             optional: bool, approx_mb: float | None) -> bool:
        if self.auto_approve:
            return True
        if name in self._skip_this_startup:
            return False
        if self.prompt_yes_no is None:
            return False  # headless mode: never install without a choice
        category_label = {
            checks.CORE: "Required", checks.AI_CORE: "AI core", checks.VOICE_INPUT: "Voice input",
            checks.VOICE_OUTPUT: "Voice output", checks.OPTIONAL: "Optional",
        }.get(category, category)
        detail = f"{category_label} component"
        if internet:
            detail += " — Internet access is required to install"
        if approx_mb:
            detail += f" (~{approx_mb:.0f} MB)"
        return bool(self.prompt_yes_no(name, purpose, detail, optional))

    # ---------------------------------------------------------------- plans

    def _plans(self, missing: list[DependencyStatus]) -> list[InstallPlan]:
        plans: list[InstallPlan] = []
        for status in missing:
            plan = self._plan_for(status)
            if plan is not None:
                plans.append(plan)
        return plans

    def _plan_for(self, status: DependencyStatus) -> InstallPlan | None:
        name = status.name
        if name == "PySide6":
            return self._pip_plan("PySide6", "User interface framework")
        if name == "Global hotkeys (pynput)":
            return self._pip_plan("pynput", "Global voice hotkeys")
        if name == "Audio input":
            return self._pip_plan("sounddevice", "Microphone capture")
        if name == "Audio output":
            return self._pip_plan("sounddevice", "Speaker playback")
        if name == "HTTP client (requests)":
            return self._pip_plan("requests", "Local Ollama communication")
        if name == "Whisper (faster-whisper)":
            return self._pip_plan("faster-whisper", "Offline speech-to-text")
        if name == "ONNX Runtime":
            return self._pip_plan("onnxruntime", "VAD and Kokoro inference")
        if name == "Windows automation (pywin32)":
            return self._pip_plan("pywin32", "Window and clipboard tools", optional=True)
        if name == "Resource monitor (psutil)":
            return self._pip_plan("psutil", "RAM/CPU monitoring", optional=True)
        if name == "Piper (optional)":
            return self._pip_plan("piper-tts", "Optional TTS fallback", optional=True)

        if name == "Ollama":
            return InstallPlan(
                name="Ollama", purpose="Local LLM server for the agent",
                category=status.category, internet_required=True,
                approx_size_mb=900, optional=status.optional,
                install=self._install_ollama, verify=self._verify_ollama_running,
            )
        if ":" in name and not name.endswith("cloud"):
            # Any local Ollama model (e.g. qwen3.5:4b, gemma4:e4b). Cloud
            # models are never "missing" — they are reported as satisfied.
            return InstallPlan(
                name=name, purpose="AI model for the agent (offline fallback)",
                category=status.category, internet_required=True,
                approx_size_mb=_approx_model_mb(name), optional=status.optional,
                install=lambda prog: self.model_manager.pull(name, prog)
                if self.model_manager else (_ for _ in ()).throw(InstallError("models dir missing")),
                verify=lambda: self.client is not None and self.client.model_installed(name),
            )
        if name == "Silero VAD model":
            return InstallPlan(
                name=name, purpose="Speech start/end detection",
                category=status.category, internet_required=True, approx_size_mb=2,
                optional=status.optional,
                install=lambda prog: self.model_manager.install_local_model("silero_vad.onnx", prog)
                if self.model_manager else (_ for _ in ()).throw(InstallError("models dir missing")),
                verify=lambda: (self.models_dir / "silero_vad.onnx").exists(),
            )
        if name == "Kokoro TTS":
            return InstallPlan(
                name=name, purpose="Local text-to-speech",
                category=status.category, internet_required=True,
                approx_size_mb=460, optional=status.optional,
                install=self._install_kokoro,
                verify=lambda: (self.models_dir / "kokoro-v1.0.onnx").exists()
                and (self.models_dir / "voices-v1.0.bin").exists()
                and package_installed("kokoro_onnx"),
            )
        if name.startswith("Whisper model"):
            model = name.removeprefix("Whisper model (").removesuffix(")")
            return InstallPlan(
                name=name, purpose="Speech-to-text model (stored in the models folder)",
                category=status.category, internet_required=True,
                approx_size_mb=485 if "small" in name else 145, optional=status.optional,
                install=lambda prog: self._warm_whisper(prog),
                verify=lambda: self._whisper_ready(model),
            )
        return None

    def _pip_plan(self, package: str, purpose: str, optional: bool = False) -> InstallPlan:
        module = _PIP_PACKAGES.get(package, (package, checks.OPTIONAL))[0]

        def install(progress) -> None:
            pip_install([package], progress)

        def verify() -> bool:
            try:
                __import__(module)
                return True
            except Exception:
                return False

        return InstallPlan(
            name=package, purpose=purpose,
            category=checks.CORE if not optional else checks.OPTIONAL,
            internet_required=True, optional=optional,
            install=install, verify=verify,
        )

    def _install_ollama(self, progress) -> None:
        from utils.platform import local_data_dir

        dest = local_data_dir() / "downloads" / "OllamaSetup.exe"
        download_ollama_installer(dest, progress)
        progress(100.0, "Launching Ollama installer — follow the installer prompts")
        import subprocess

        subprocess.Popen([str(dest)], shell=False)

    def _verify_ollama_running(self) -> bool:
        return self.client is not None and self.client.ping()

    def _install_kokoro(self, progress) -> None:
        if not self.model_manager:
            raise InstallError("models directory unavailable")
        progress(5.0, "Installing Kokoro TTS engine (kokoro-onnx)...")
        pip_install(["kokoro-onnx"], progress)
        progress(40.0, "Downloading Kokoro model (330 MB)...")
        self.model_manager.install_local_model("kokoro-v1.0.onnx", progress)
        progress(70.0, "Downloading Kokoro voices (130 MB)...")
        self.model_manager.install_local_model("voices-v1.0.bin", progress)

    def _whisper_ready(self, model: str) -> bool:
        """True when the whisper-<size> snapshot is present in the models dir."""
        if self.models_dir is None:
            return False
        from stt.whisper_engine import local_model_dir

        return (local_model_dir(self.models_dir, model) / "model.bin").exists()

    def _warm_whisper(self, progress) -> None:
        """Download the faster-whisper snapshot into the models folder."""
        if self.models_dir is None:
            raise InstallError("models directory unavailable")
        progress(10.0, "Downloading Whisper weights (needs Internet once)...")
        model = self.settings.whisper_model if self.settings else "small"
        self._download_whisper_model(model, progress)
        if not self._whisper_ready(model):
            raise InstallError("Whisper model download did not complete")
        progress(100.0, "Whisper model installed")

    def _download_whisper_model(self, model: str, progress) -> None:
        """Download the faster-whisper snapshot into
        <models>/whisper-<size> with a tqdm hook so the progress dialog
        shows percentage movement."""
        if self.models_dir is None:
            raise InstallError("models directory unavailable")
        import os

        # Quiet the symlink warning that confuses users on Windows.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            from huggingface_hub import snapshot_download
            from tqdm import tqdm
        except Exception as exc:  # noqa: BLE001
            raise InstallError(
                f"Could not reach the model download service: {exc}"
            ) from exc

        from stt.whisper_engine import local_model_dir

        repo = f"Systran/faster-whisper-{model}"
        label = f"Downloading {model} Whisper model"
        dest_dir = local_model_dir(self.models_dir, model)
        dest_dir.mkdir(parents=True, exist_ok=True)

        class _ProgressTqdm(tqdm):  # type: ignore[misc]
            def update(self, n=1):
                super().update(n)
                if self.total:
                    progress(min(100.0, self.n / self.total * 100), label)
                else:
                    progress(0.0, label)

        try:
            snapshot_download(
                repo_id=repo, local_dir=str(dest_dir), tqdm_class=_ProgressTqdm,
            )
        except Exception as exc:  # noqa: BLE001
            raise InstallError(f"Whisper model download failed: {exc}") from exc

    # -------------------------------------------------------------- resolve

    def resolve(self, statuses: list[DependencyStatus]) -> bool:
        """Prompt for and install missing dependencies. Returns True when all
        required dependencies are now present (or explicitly skipped)."""
        missing = checks.missing_statuses(statuses)
        if not missing:
            self._record_ok(statuses)
            return True
        plans = self._plans(missing)
        for plan in plans:
            if self._skip_this_startup and plan.name in self._skip_this_startup:
                continue
            approved = self._ask(
                plan.name, plan.purpose, plan.category,
                plan.internet_required, plan.optional, plan.approx_size_mb,
            )
            if not approved:
                self._skip_this_startup.add(plan.name)
                self._record(plan.name, "skipped")
                continue
            success = self._install(plan)
            if success:
                self._record(plan.name, "installed")
            else:
                self._skip_this_startup.add(plan.name)
                self._record(plan.name, "failed")
        # Re-check after any installation attempts.
        final = checks.run_all_checks(self.settings, self.client, self.cloud_model)
        required_missing = [
            s for s in checks.missing_statuses(final)
            if not s.optional and s.name not in self._skip_this_startup
        ]
        self._record_ok(final)
        return len(required_missing) == 0

    def _install(self, plan: InstallPlan) -> bool:
        if self.show_progress is None:
            return False

        result: dict[str, bool] = {"ok": False}

        def run(progress) -> None:
            try:
                plan.install(progress)
                if plan.verify is None or plan.verify():
                    result["ok"] = True
                else:
                    raise InstallError("Verification failed after install")
            except Exception as exc:  # noqa: BLE001
                self._record(plan.name, "failed", error=str(exc))
                raise

        try:
            self.show_progress(f"Installing {plan.name}", run)
        except InstallError as exc:
            self._log(f"Install failed for {plan.name}: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            self._log(f"Install error for {plan.name}: {exc}")
            return False
        return result["ok"]

    # -------------------------------------------------------------- records

    def _record(self, name: str, status: str, error: str | None = None) -> None:
        if self.memory is not None:
            try:
                self.memory.record_install(name, status, error=error)
            except Exception:
                pass

    def _record_ok(self, statuses: list[DependencyStatus]) -> None:
        if self.memory is None:
            return
        for s in statuses:
            if s.present:
                try:
                    self.memory.record_install(s.name, "installed", version=s.version or None)
                except Exception:
                    pass

    def _log(self, message: str) -> None:
        try:
            from utils.logging import get_logger

            get_logger("setup").warning("%s", message)
        except Exception:
            pass
