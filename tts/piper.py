"""Piper TTS — optional fallback engine.

Voice files are local .onnx (+ .json) placed in the models directory during
setup. piper-tts is CPU-only and fully offline.
"""

from __future__ import annotations

import io

import numpy as np

from tts.base import TTSEngine


class PiperEngine(TTSEngine):
    def __init__(self, voice_path=None) -> None:
        self.voice_path = voice_path  # Path to voice .onnx file
        self._voice = None

    def is_available(self) -> bool:
        return self.voice_path is not None and self.voice_path.exists()

    def load(self, voice: str | None = None) -> None:
        if voice and not self.voice_path:
            # Allow selecting a voice by name; caller resolves the path.
            self.voice_path = voice
        if self._voice is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(
                f"Piper voice not found at {self.voice_path}. Add a voice "
                "file in the models directory and select it in Settings."
            )
        try:
            from piper import PiperVoice  # type: ignore
        except Exception as exc:
            raise RuntimeError("piper-tts is not installed") from exc
        self._voice = PiperVoice.load(str(self.voice_path))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if self._voice is None:
            self.load()
        buffer = io.BytesIO()
        self._voice.synthesize(text, buffer)
        import wave

        buffer.seek(0)
        with wave.open(buffer, "rb") as wav:
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        return audio, rate

    def unload(self) -> None:
        self._voice = None

    @property
    def display_name(self) -> str:
        return f"Piper ({self.voice_path.name if self.voice_path else 'unset'})"
