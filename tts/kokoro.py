"""Kokoro TTS — the default voice engine.

Two backends, both CPU-only and fully local:
* ``kokoro-onnx`` (preferred — ONNX Runtime, lighter on RAM, no espeak-ng);
* ``kokoro`` (torch) as a fallback when only the original package is present.

The model files (kokoro-v1.0.onnx + voices-v1.0.bin) are placed in the
models directory during setup — never downloaded silently.
"""

from __future__ import annotations

import numpy as np

from tts.base import TTSEngine


class KokoroEngine(TTSEngine):
    def __init__(self, voice: str = "af_heart", model_path=None, voices_path=None) -> None:
        self.voice = voice
        self.model_path = model_path  # kokoro-v1.0.onnx
        self.voices_path = voices_path  # voices-v1.0.bin
        self._backend: str | None = None
        self._pipe = None

    def is_available(self) -> bool:
        return self.model_path is not None and self.model_path.exists()

    def load(self, voice: str | None = None) -> None:
        if voice:
            self.voice = voice
        if self._pipe is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(
                f"Kokoro model not found at {self.model_path}. Use the Setup "
                "dialog to install the TTS model."
            )
        # Try kokoro-onnx first (lighter), then the torch package.
        try:
            from kokoro_onnx import Kokoro  # type: ignore

            self._pipe = Kokoro(str(self.model_path), str(self.voices_path))
            self._backend = "onnx"
        except Exception:
            try:
                from kokoro import KPipeline  # type: ignore

                self._pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
                self._backend = "torch"
            except Exception as exc:
                raise RuntimeError(
                    "Neither kokoro-onnx nor kokoro is installed. Use the "
                    "Setup dialog to install TTS."
                ) from exc

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if self._pipe is None:
            self.load()
        if self._backend == "onnx":
            audio, rate = self._pipe.create(text, voice=self.voice)
            return np.asarray(audio, dtype=np.float32), int(rate)
        # torch backend
        import numpy as _np

        chunks: list[np.ndarray] = []
        for _gs, _ps, audio in self._pipe(text, voice=self.voice, speed=1.0):
            chunks.append(_np.asarray(audio, dtype=_np.float32))
        audio = _np.concatenate(chunks) if chunks else _np.zeros(0, dtype=_np.float32)
        return audio, 24000

    def unload(self) -> None:
        self._pipe = None

    @property
    def display_name(self) -> str:
        return f"Kokoro ({self.voice}, {self._backend or 'lazy'})"
