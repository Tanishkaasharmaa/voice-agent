"""Whisper STT using faster-whisper.

Deliberately uses faster-whisper (CTranslate2) instead of openai-whisper +
torch: int8 CPU inference with a much smaller RAM footprint, which matters on
a 16 GB machine. Default model: small (configurable: tiny/base/small/medium).

The weights live in the app's own models folder
(``%LOCALAPPDATA%/LocalVoiceAgent/models/whisper-<size>``), downloaded once at
setup time with user consent via ``snapshot_download(local_dir=...)``. The
engine always loads them by local path — it never resolves a repo id, so
after setup the app is fully offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from stt.base import STTEngine

SUPPORTED_MODELS = ("tiny", "base", "small", "medium")


def local_model_dir(models_dir, model: str) -> Path:
    """Directory where the faster-whisper snapshot for a size is stored.

    Mirrors the other local models (silero/kokoro) under
    ``%LOCALAPPDATA%/LocalVoiceAgent/models``. ``models_dir`` may be a
    ``Path`` or ``str``.
    """
    return Path(models_dir) / f"whisper-{model}"


_HEAVY_WARNING = {
    "medium": "The 'medium' Whisper model is heavy for CPU transcription and "
    "may be slow on this machine. Prefer 'small'.",
}


class WhisperEngine(STTEngine):
    def __init__(self, model: str = "small", language: str = "en",
                 cpu_threads: int = 0, compute_type: str = "int8",
                 local_files_only: bool = False, models_dir=None) -> None:
        self.model_name = model if model in SUPPORTED_MODELS else "small"
        self.language = language or None
        self.cpu_threads = cpu_threads
        self.compute_type = compute_type
        # When set, the engine loads ``models_dir/whisper-<size>`` by path and
        # never resolves a repo id — fully offline. When None (dev/tests
        # without a models dir), it falls back to the bare model name.
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.local_files_only = local_files_only
        self._model = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError(
                "faster-whisper is not installed. Use the Setup dialog to "
                "install the STT dependency (requires Internet once)."
            )
        # Quiet the HF symlink warning on Windows — it is harmless and only
        # confuses users during the one-time download.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from faster_whisper import WhisperModel  # type: ignore

        threads = self.cpu_threads or 0
        model_ref = self.model_name
        if self.models_dir is not None:
            path = local_model_dir(self.models_dir, self.model_name)
            if not (path / "model.bin").exists():
                raise FileNotFoundError(
                    f"Whisper model not found at {path}. Use the Setup dialog "
                    "to download it (requires Internet once)."
                )
            model_ref = str(path)
        self._model = WhisperModel(
            model_ref,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=threads if threads > 0 else None,
            local_files_only=self.local_files_only,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            self.load()
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        segments, _info = self._model.transcribe(
            audio, language=self.language, beam_size=1, vad_filter=False
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def unload(self) -> None:
        self._model = None

    @property
    def display_name(self) -> str:
        return f"Whisper ({self.model_name})"

    @staticmethod
    def warning_for(model: str) -> str | None:
        return _HEAVY_WARNING.get(model)
