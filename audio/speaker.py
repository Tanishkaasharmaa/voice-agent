"""Speaker output using sounddevice.

Playback is non-blocking (an output stream is fed from a producer thread).
``stop()`` aborts immediately so hotkey interruption works naturally.
"""

from __future__ import annotations

import queue
import threading

import numpy as np


class Speaker:
    def __init__(self, sample_rate: int = 24000, device: str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._playing = threading.Event()
        # Optional callback invoked with per-chunk RMS (float 0..~1) while
        # audio is actually being consumed — used for character lip sync.
        self.on_amplitude: callable | None = None

    def list_devices(self) -> list[dict]:
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return []
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                devices.append({"index": i, "name": dev.get("name", ""), "channels": dev["max_output_channels"]})
        return devices

    def _resolve_device_index(self) -> int | None:
        if not self.device:
            return None
        try:
            import sounddevice as sd  # type: ignore

            for i, dev in enumerate(sd.query_devices()):
                if dev.get("name", "") == self.device and dev.get("max_output_channels", 0) > 0:
                    return i
        except Exception:
            return None
        return None

    def play(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Queue audio for playback. Returns immediately."""
        if audio is None or len(audio) == 0:
            return
        # Set synchronously, NOT in the worker thread: the worker may take
        # a moment (library import + device open) before it would set this,
        # and callers rely on is_playing() being true the instant play()
        # returns — otherwise "speaking" ends before audio is heard.
        self._playing.set()
        self._queue.put((np.asarray(audio, dtype=np.float32),
                         sample_rate or self.sample_rate))
        if self._thread is None or not self._thread.is_alive():
            self._start_worker()

    def _start_worker(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    def _emit_amplitude(self, chunk: np.ndarray) -> None:
        if self.on_amplitude is None or len(chunk) == 0:
            return
        try:
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            self.on_amplitude(min(1.0, rms))
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            # No audio backend available — drop queued audio instead of
            # leaving is_playing() stuck true forever.
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._playing.clear()
            return

        idx = self._resolve_device_index()
        while not self._stop_event.is_set():
            try:
                audio, rate = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                break
            try:
                # Written in chunks so playback progress paces the amplitude
                # callbacks (a single write would give no timing feedback).
                chunk_size = max(1, rate // 10)
                with sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                                     device=idx) as stream:
                    for start in range(0, len(audio), chunk_size):
                        if self._stop_event.is_set():
                            break
                        chunk = audio[start:start + chunk_size]
                        stream.write(chunk)
                        self._emit_amplitude(chunk)
            except Exception:
                pass
            finally:
                # Only clear when nothing else is queued — otherwise a
                # queued-but-not-started item would briefly read as idle.
                if self._queue.empty() or self._stop_event.is_set():
                    self._playing.clear()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self) -> None:
        """Abort playback immediately and discard queued audio."""
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._playing.clear()
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
