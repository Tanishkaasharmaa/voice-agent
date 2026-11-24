"""VAD interfaces.

``VADEngine`` is the model wrapper (Silero ONNX). ``SpeechActivityDetector``
is the pure state machine that turns per-frame speech probabilities into
speech-start / speech-end events — it is fully unit-testable without a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class VADEngine(ABC):
    """Wraps a VAD model. Implementations must be CPU-friendly."""

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def probability(self, frame: object) -> float:
        """Speech probability (0..1) for one audio frame."""
        ...


class SpeechActivityDetector:
    """Turns VAD probabilities into speech events.

    States: SILENCE -> SPEECH -> (trailing silence) -> SILENCE.

    Rules:
    * speech starts when probability >= threshold;
    * speech ends after ``silence_duration`` seconds of silence;
    * blips shorter than ``minimum_speech_duration`` are ignored as noise;
    * ``maximum_duration`` force-ends the utterance.
    """

    def __init__(
        self,
        frame_duration_s: float = 0.032,
        silence_duration: float = 0.8,
        minimum_speech_duration: float = 0.3,
        maximum_duration: float = 60.0,
        sensitivity: float = 0.5,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[float], None] | None = None,
    ) -> None:
        self.frame_duration_s = frame_duration_s
        self.silence_duration = silence_duration
        self.minimum_speech_duration = minimum_speech_duration
        self.maximum_duration = maximum_duration
        self.sensitivity = sensitivity
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._in_speech = False
        self._speech_started_at: float | None = None
        self._silence_since: float | None = None
        self._last_timestamp: float | None = None
        self._clock = 0.0   # advances one frame_duration_s per fed frame

    # -------------------------------------------------------------- config

    @property
    def threshold(self) -> float:
        # Higher sensitivity -> lower threshold -> more aggressive detection.
        return max(0.2, 0.5 - self.sensitivity * 0.4)

    # ---------------------------------------------------------------- feed

    def feed(self, probability: float, timestamp: float | None = None) -> bool:
        """Feed one frame's probability. Returns True when an utterance
        completes (speech end or maximum duration).

        Without an explicit timestamp the internal clock advances by one
        ``frame_duration_s`` per call."""
        if timestamp is not None:
            ts = timestamp
        else:
            self._clock += self.frame_duration_s
            ts = self._clock
        self._last_timestamp = ts
        is_speech = probability >= self.threshold

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._speech_started_at = ts
                self._silence_since = None
                if self.on_speech_start:
                    self.on_speech_start()
            return False

        # In speech: track trailing silence.
        if is_speech:
            self._silence_since = None
        else:
            if self._silence_since is None:
                self._silence_since = ts

        # Maximum duration always ends the utterance.
        if self._speech_started_at is not None and ts - self._speech_started_at >= self.maximum_duration:
            return self._end_speech(ts)

        # Trailing silence ends the utterance.
        if self._silence_since is not None and ts - self._silence_since >= self.silence_duration:
            return self._end_speech(ts)

        return False

    def reset(self) -> None:
        self._in_speech = False
        self._speech_started_at = None
        self._silence_since = None
        self._last_timestamp = None
        self._clock = 0.0

    # ---------------------------------------------------------------- state

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def _end_speech(self, ts: float) -> bool:
        started = self._speech_started_at
        duration = ts - started if started is not None else 0.0
        self._in_speech = False
        self._speech_started_at = None
        self._silence_since = None
        if duration >= self.minimum_speech_duration:
            if self.on_speech_end:
                self.on_speech_end(duration)
            return True
        # Too short to be real speech — treat as noise, stay silent.
        return False


class EnergyVoiceGate:
    """RMS-energy fallback for when the neural VAD is unavailable or failing.

    Produces a binary 0/1 "probability" from the frame's RMS level using an
    adaptive noise floor plus an envelope-variance discriminator:

    * DYNAMIC signals (coefficient of variation of the recent RMS window is
      high — speech has syllable swings) are classified purely against the
      current threshold, and never move the floor — so even a 60 s nonstop
      reply cannot push the threshold out of reach.
    * STATIONARY signals (flat RMS — ambient hum/noise) calibrate the floor
      toward themselves quickly, in both directions.

    Feed its output into ``SpeechActivityDetector`` to reuse all of the
    timing/state logic.
    """

    WINDOW = 32          # ~1 s of frames at 32 ms
    STATIONARY_CV = 0.35

    def __init__(self, multiplier: float = 4.0, minimum: float = 0.0005,
                 adapt_rate: float = 0.05) -> None:
        self.multiplier = multiplier
        self.minimum = minimum
        self.adapt_rate = adapt_rate
        self.floor: float | None = None
        self._window: list[float] = []

    def probability(self, frame) -> float:
        import numpy as np

        data = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
        if self.floor is None:
            # Start low, NOT from the first frame: opening mid-speech must
            # not calibrate the floor to speech level (everything after
            # would read as silence forever).
            self.floor = self.minimum
        self._window.append(rms)
        if len(self._window) > self.WINDOW:
            self._window.pop(0)

        threshold = max(self.floor * self.multiplier, self.minimum)

        if len(self._window) >= self.WINDOW:
            mean = sum(self._window) / len(self._window)
            variance = sum((v - mean) ** 2 for v in self._window) / len(self._window)
            cv = (variance ** 0.5 / mean) if mean > 0 else 0.0
            spread = max(self._window) - min(self._window)
            # Stationary needs BOTH a low CV and a genuinely flat envelope.
            # A mostly-loud window sprinkled with a few dips has borderline
            # CV but a large spread — that is speech, not noise.
            stationary = cv < self.STATIONARY_CV and spread < 0.6 * mean
        else:
            stationary = False     # not enough history — assume speech-like

        if stationary:
            # Flat signal: ambient noise (or a pathological held tone).
            # Calibrate the floor toward it, then judge against the updated
            # threshold so steady noise settles to silence within seconds.
            if rms < self.floor:
                self.floor = rms
            else:
                self.floor += (rms - self.floor) * self.adapt_rate
            threshold = max(self.floor * self.multiplier, self.minimum)

        return 1.0 if rms >= threshold else 0.0
