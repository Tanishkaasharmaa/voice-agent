"""AudioManager — owns capture, buffering, and the VAD listener.

Two listening modes:
* push_to_talk — capture while the hotkey is held; utterance ends on release;
* auto_stop    — VAD detects speech end (default ~800 ms silence) or the
                  maximum recording duration.

Captured audio is accumulated as float32 mono at 16 kHz and delivered to
``on_utterance`` on a worker thread. By default nothing is written to disk
(debug recordings are opt-in).
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from audio.microphone import Microphone, MicrophoneError
from utils.logging import get_logger
from vad.base import EnergyVoiceGate, SpeechActivityDetector, VADEngine

log = get_logger("audio")

# Consecutive VAD inference failures before falling back to the energy gate
# (~0.8 s of frames at 32 ms).
_VAD_FAILURE_LIMIT = 25


class AudioManager:
    def __init__(
        self,
        microphone: Microphone,
        vad: VADEngine | None = None,
        settings=None,
        debug_recording: bool = False,
        recordings_dir=None,
    ) -> None:
        self.microphone = microphone
        self.vad = vad
        self.settings = settings
        self.debug_recording = debug_recording
        self.recordings_dir = recordings_dir

        self.on_utterance: Callable[[np.ndarray, int], None] | None = None
        self.on_speech_state: Callable[[bool], None] | None = None
        self.on_error: Callable[[str], None] | None = None

        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._mode: str | None = None
        self._started_at: float = 0.0
        self._detector: SpeechActivityDetector | None = None
        self._vad_failed = False          # neural VAD unusable this session
        self._vad_error_count = 0         # consecutive inference failures
        self._energy_gate: EnergyVoiceGate | None = None
        self._active = threading.Event()
        self._stop_requested = False
        self._worker: threading.Thread | None = None
        self._frames_received = 0

    # ------------------------------------------------------------ listening

    def start_listening(self, mode: str = "auto_stop") -> None:
        """mode: 'push_to_talk' or 'auto_stop'."""
        if self._active.is_set():
            return
        self._mode = mode
        self._stop_requested = False
        self._frames_received = 0
        with self._buffer_lock:
            self._buffer = []
        self._started_at = time.monotonic()
        self._setup_detector()
        self._active.set()
        self._worker = threading.Thread(target=self._capture_loop, name="audio-manager", daemon=True)
        self._worker.start()

    def ensure_mode(self, mode: str) -> None:
        """Upgrade the running capture session to another mode without
        restarting the microphone.

        CTRL+WIN+SPACE fires the push-to-talk press (CTRL+WIN) before the
        auto-stop trigger (SPACE), so when auto-stop arrives the mic is
        already capturing — switch the session instead of ignoring it.
        """
        if not self._active.is_set():
            self.start_listening(mode)
            return
        self._mode = mode
        if mode == "auto_stop" and self._detector is None:
            self._setup_detector()

    @property
    def mode(self) -> str | None:
        return self._mode

    def _setup_detector(self) -> None:
        self._detector = None
        self._vad_failed = False
        self._vad_error_count = 0
        self._energy_gate = None
        if self.vad is not None and self.vad.is_available():
            try:
                self.vad.load()
                # Fresh LSTM/context per session — leftover state from the
                # previous utterance must not skew this one.
                reset = getattr(self.vad, "reset", None)
                if callable(reset):
                    reset()
                self._detector = self._make_timing_detector()
                return
            except Exception as exc:
                log.warning("Silero VAD failed to load (%s) — falling back "
                            "to energy-based detection", exc)
        elif self.vad is not None:
            log.warning("Silero VAD model not available — using energy-based "
                        "detection for auto-stop")
        # Fallback: the timing state machine is reused, but fed a binary
        # probability from an adaptive RMS gate so auto-stop always works.
        self._energy_gate = EnergyVoiceGate()
        self._detector = self._make_timing_detector()

    def _make_timing_detector(self) -> SpeechActivityDetector:
        silence = float(self.settings.get("voice_input.silence_duration", 0.8)) if self.settings else 0.8
        min_speech = float(self.settings.get("voice_input.minimum_speech_duration", 0.3)) if self.settings else 0.3
        max_dur = float(self.settings.get("voice_input.maximum_recording_duration", 60)) if self.settings else 60
        sensitivity = float(self.settings.get("voice_input.vad_sensitivity", 0.5)) if self.settings else 0.5
        return SpeechActivityDetector(
            frame_duration_s=self.microphone.FRAME_SIZE / self.microphone.sample_rate,
            silence_duration=silence,
            minimum_speech_duration=min_speech,
            maximum_duration=max_dur,
            sensitivity=sensitivity,
            on_speech_start=lambda: self._notify_speech(True),
            on_speech_end=lambda _d: self._auto_stop_utterance(),
        )

    def stop_listening(self) -> None:
        """End capture (push-to-talk release). Finalizes the utterance."""
        self._stop_requested = True

    def cancel(self) -> None:
        """Discard capture without producing an utterance (interruption/error)."""
        self._stop_requested = True
        with self._buffer_lock:
            self._buffer = []
        self._notify_speech(False)

    # ------------------------------------------------------------- internals

    def _notify_speech(self, speaking: bool) -> None:
        if self.on_speech_state:
            try:
                self.on_speech_state(speaking)
            except Exception:
                pass

    def _capture_loop(self) -> None:
        try:
            self.microphone.start(self._on_frame)
        except MicrophoneError as exc:
            self._active.clear()
            if self.on_error:
                self.on_error(str(exc))
            return

        try:
            while self._active.is_set() and not self._stop_requested:
                # Max-duration safety net for both modes (also covers
                # auto-stop when the VAD is unavailable or no speech ever
                # starts).
                if self.settings is not None:
                    max_dur = float(self.settings.get("voice_input.maximum_recording_duration", 60))
                    if time.monotonic() - self._started_at >= max_dur:
                        break
                # No-audio watchdog: a silent/muted device delivers no
                # callbacks at all — fail fast instead of "listening" for
                # the full max duration with an empty buffer.
                if (self._frames_received == 0
                        and time.monotonic() - self._started_at >= 5.0):
                    log.warning("no audio frames received from microphone "
                                "within 5 s")
                    if self.on_error:
                        self.on_error("Microphone delivered no audio — "
                                      "check the input device/mute state.")
                    break
                time.sleep(0.02)
        finally:
            self.microphone.stop()
            self._active.clear()
            self._notify_speech(False)
            # Finalize whatever was captured on every exit path (release,
            # max duration, VAD end). Safe to run even when the VAD already
            # finalized — an empty buffer is a no-op.
            self._finalize()

    def _on_frame(self, frame: np.ndarray) -> None:
        if not self._active.is_set():
            return
        self._frames_received += 1
        with self._buffer_lock:
            self._buffer.append(frame)
        if self._detector is None or self._mode != "auto_stop":
            return
        try:
            if self._vad_failed or self._energy_gate is not None:
                prob = self._energy_gate.probability(frame)
            else:
                prob = self.vad.probability(frame)
                self._vad_error_count = 0
        except Exception as exc:
            self._vad_error_count += 1
            if self._vad_error_count == 1:
                log.warning("VAD inference failing (%s)", exc)
            if self._vad_error_count >= _VAD_FAILURE_LIMIT \
                    and self._energy_gate is None:
                # Neural VAD is unusable — switch to the energy gate so
                # auto-stop keeps working instead of listening forever.
                log.warning("VAD failed %d frames in a row — switching to "
                            "energy-based detection", self._vad_error_count)
                self._vad_failed = True
                self._energy_gate = EnergyVoiceGate()
                try:
                    self._detector.reset()
                except Exception:
                    pass
                return
            return
        try:
            self._detector.feed(prob)
        except Exception:
            pass

    def _auto_stop_utterance(self) -> None:
        # VAD detected speech end — let the capture loop wind down; the
        # finally block stops the microphone and finalizes the utterance.
        # (Calling microphone.stop() from the VAD callback thread could
        # deadlock the sounddevice audio thread.)
        # Grace: while the energy gate calibrates its noise floor (~2 s),
        # ambient noise can briefly read as "speech ended" — ignore ends in
        # that window so a session doesn't stop before the user speaks.
        if time.monotonic() - self._started_at < 1.5:
            return
        self._stop_requested = True

    def _finalize(self) -> None:
        with self._buffer_lock:
            audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
            self._buffer = []
        if self._detector is not None:
            self._detector.reset()
        if len(audio) == 0:
            return
        if self.debug_recording and self.recordings_dir is not None:
            self._save_debug(audio)
        if self.on_utterance:
            try:
                self.on_utterance(audio, self.microphone.sample_rate)
            except Exception:
                pass

    def _save_debug(self, audio: np.ndarray) -> None:
        try:
            import wave

            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            path = self.recordings_dir / f"utterance-{int(time.time())}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.microphone.sample_rate)
                wav.writeframes((audio * 32767).astype(np.int16).tobytes())
        except Exception:
            pass
