"""Text-to-speech interfaces and the TTS manager.

``TTSManager`` accepts speech-ready sentences, synthesizes them on a worker
thread, and plays them through the Speaker. ``stop()`` interrupts playback
immediately (hotkey interruption) and clears the queue.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod

import numpy as np

from audio.speaker import Speaker
from utils.logging import get_logger

log = get_logger("tts")


class TTSEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def load(self, voice: str | None = None) -> None:
        ...

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (float32 audio, sample_rate). CPU-friendly, blocking."""
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...


class TTSManager:
    def __init__(self, engine: TTSEngine, speaker: Speaker) -> None:
        self.engine = engine
        self.speaker = speaker
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.on_state: callable | None = None  # (bool playing)
        self.on_amplitude: callable | None = None  # (float rms 0..1)
        # Sentences accepted but not fully played yet — lets callers know
        # when speaking has really finished (queue + synthesis + playback).
        self._pending = 0
        # Forward the speaker's per-chunk amplitude to our callback.
        speaker.on_amplitude = self._on_speaker_amplitude

    def _on_speaker_amplitude(self, level: float) -> None:
        if self.on_amplitude:
            try:
                self.on_amplitude(level)
            except Exception:
                pass

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="tts-manager", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                break
            try:
                if not self.engine.is_available():
                    self.engine.load()
                audio, rate = self.engine.synthesize(text)
                self._notify(True)
                self.speaker.play(audio, rate)
                while self.speaker.is_playing() and not self._stop_event.is_set():
                    import time

                    time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001 - never kill the TTS thread
                log.error("TTS failed for %r: %s", text[:80], exc)
            finally:
                self._pending = max(0, self._pending - 1)
                self._notify(False)

    def is_busy(self) -> bool:
        """True while any sentence is queued, being synthesized, or still
        audible — i.e. the agent must remain in the SPEAKING state."""
        return self._pending > 0 or self.speaker.is_playing()

    def pending_count(self) -> int:
        return self._pending

    def _notify(self, playing: bool) -> None:
        if self.on_state:
            try:
                self.on_state(playing)
            except Exception:
                pass

    # ------------------------------------------------------------------ API

    def speak(self, sentence: str) -> None:
        """Queue a speech-ready sentence."""
        if not sentence.strip():
            return
        self.start()
        # Incremented synchronously so is_busy() is reliable the moment
        # speak() returns (no race with the worker thread picking it up).
        self._pending += 1
        self._queue.put(sentence)

    def is_speaking(self) -> bool:
        return self.speaker.is_playing()

    def stop(self) -> None:
        """Interrupt immediately: stop audio, clear queue."""
        self._stop_event.set()
        self.speaker.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._pending = 0
        self._notify(False)

    def restart(self) -> None:
        self._stop_event.clear()
        self.start()

    def shutdown(self) -> None:
        self.stop()
        try:
            self.engine.unload()
        except Exception:
            pass
