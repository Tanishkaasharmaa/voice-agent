"""Speech-to-text engine interface.

Engines are lazy: they must not import their heavy runtime at module load,
so the app (and tests) can import this package without Whisper installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a float32 mono audio array. Must not block the UI
        thread — callers run this on a worker thread."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model memory when the app wants to free RAM."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...
