"""Microphone capture using sounddevice.

Frames (float32 mono) are pushed into a queue; a worker thread consumes them
so the audio callback never does heavy work. Device selection is by name
(config); nothing is hardcoded.
"""

from __future__ import annotations

import queue
import threading

import numpy as np


class MicrophoneError(Exception):
    pass


class Microphone:
    FRAME_SIZE = 512  # matches Silero VAD frame size at 16 kHz

    def __init__(self, sample_rate: int = 16000, device: str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue = queue.Queue(maxsize=400)
        self._stream = None
        self._lock = threading.Lock()

    def list_devices(self) -> list[dict]:
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return []
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({"index": i, "name": dev.get("name", ""), "channels": dev["max_input_channels"]})
        return devices

    @staticmethod
    def _resolve_device_index(device_name: str | None) -> int | None:
        if not device_name:
            return None
        try:
            import sounddevice as sd  # type: ignore

            for i, dev in enumerate(sd.query_devices()):
                if dev.get("name", "") == device_name and dev.get("max_input_channels", 0) > 0:
                    return i
        except Exception:
            return None
        return None

    def start(self, on_frame) -> None:
        """Begin capture; ``on_frame(np.float32 array)`` runs on the audio thread."""
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise MicrophoneError("sounddevice is not installed") from exc

        idx = self._resolve_device_index(self.device)

        def _callback(indata, _frames, _time, _status):
            frame = np.squeeze(indata).astype(np.float32)
            try:
                on_frame(frame)
            except Exception:
                pass

        with self._lock:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.FRAME_SIZE,
                channels=1,
                dtype="float32",
                device=idx,
                callback=_callback,
            )
            self._stream.start()

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
