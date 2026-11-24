"""Tests proving auto-stop works even when the neural VAD is unusable."""

import numpy as np
import pytest

from audio.manager import AudioManager
from vad.base import SpeechActivityDetector, EnergyVoiceGate


class FakeMic:
    FRAME_SIZE = 512
    sample_rate = 16000

    def __init__(self):
        self.started = False

    def start(self, on_frame):
        self.started = True

    def stop(self):
        pass


class FakeVadUnavailable:
    def is_available(self):
        return False

    def load(self):
        raise FileNotFoundError("no model")

    def probability(self, frame):
        raise RuntimeError("no model")


class FakeVadBroken:
    """Available and loads, but every inference fails (e.g. model-version
    mismatch)."""

    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def load(self):
        pass

    def probability(self, frame):
        self.calls += 1
        raise RuntimeError("bad tensor names")


def make_manager(vad, settings=None):
    mic = FakeMic()
    mgr = AudioManager(mic, vad=vad, settings=settings)
    mgr._active.set()
    mgr._mode = "auto_stop"
    mgr._setup_detector()
    return mgr


def feed(mgr, rms_level, count, frame_size=512):
    frame = np.full(frame_size, rms_level, dtype=np.float32)
    for _ in range(count):
        mgr._on_frame(frame)


def test_energy_gate_binary_output():
    gate = EnergyVoiceGate()
    quiet = np.zeros(512, dtype=np.float32)
    loud = np.full(512, 0.08, dtype=np.float32)
    assert gate.probability(quiet) == 0.0
    assert gate.probability(loud) == 1.0


def test_energy_gate_adapts_to_noise_floor():
    """The floor starts low, so steady ambient noise is flagged as speech
    only during calibration (~1-2 s); it must converge to silence and stay
    there while real speech above the converged threshold still trips."""
    gate = EnergyVoiceGate()
    noise = np.full(512, 0.01, dtype=np.float32)
    speech = np.full(512, 0.06, dtype=np.float32)
    outputs = [gate.probability(noise) for _ in range(200)]
    assert all(v == 0.0 for v in outputs[100:])   # converged, stable
    assert gate.probability(speech) == 1.0


def test_energy_gate_handles_speech_from_first_frame():
    """Regression: the floor must not be seeded from the first frame —
    starting mid-speech would otherwise silence everything forever."""
    gate = EnergyVoiceGate()
    loud = np.full(512, 0.05, dtype=np.float32)
    assert gate.probability(loud) == 1.0


def test_energy_gate_stays_aware_through_long_continuous_speech():
    """Natural speech alternates voiced bursts with short inter-word dips —
    a strongly varying envelope. The variance discriminator must keep such
    signal classified as speech for a full 60 s nonstop reply."""
    gate = EnergyVoiceGate()
    rng = np.random.default_rng(7)
    levels: list[float] = []
    while len(levels) < int(60 / 0.032):
        levels += [0.07] * int(rng.integers(4, 12))     # voiced syllable
        levels += [0.008] * int(rng.integers(2, 6))     # inter-word dip
    for level in levels:
        frame = np.full(512, float(level), dtype=np.float32)
        assert gate.probability(frame) == 1.0


def test_energy_gate_calibrates_out_steady_hum():
    """A flat, constant-amplitude signal (hum/noise loop) must settle to
    silence after calibration instead of reading as speech forever."""
    gate = EnergyVoiceGate()
    hum = np.full(512, 0.01, dtype=np.float32)
    outputs = [gate.probability(hum) for _ in range(200)]
    assert all(v == 0.0 for v in outputs[100:])
    # Real speech above the calibrated floor still trips it.
    assert gate.probability(np.full(512, 0.06, dtype=np.float32)) == 1.0


def test_missing_silero_model_still_auto_stops():
    mgr = make_manager(FakeVadUnavailable())
    assert mgr._detector is not None          # timing machine exists...
    assert mgr._energy_gate is not None       # ...driven by the energy gate

    feed(mgr, 0.05, 20)                       # ~0.64 s of speech
    assert mgr._detector.in_speech
    assert not mgr._stop_requested

    feed(mgr, 0.0005, 40)                     # ~1.28 s of silence
    assert mgr._stop_requested                # auto-stop fired


def test_broken_vad_fails_over_to_energy_after_repeated_errors():
    vad = FakeVadBroken()
    mgr = make_manager(vad)
    assert mgr._energy_gate is None               # starts on the neural VAD

    feed(mgr, 0.05, 10)                       # 10 failing frames — still trying
    assert vad.calls == 10
    assert mgr._energy_gate is None           # not yet given up

    feed(mgr, 0.0005, 30)                     # crosses the failure limit
    assert mgr._vad_failed
    assert mgr._energy_gate is not None       # failover engaged

    # From here auto-stop must work again: speech then silence.
    mgr._stop_requested = False
    mgr._detector.reset()
    feed(mgr, 0.05, 20)
    feed(mgr, 0.0005, 40)
    assert mgr._stop_requested


def test_ptt_mode_does_not_feed_detector():
    mgr = make_manager(FakeVadUnavailable())
    mgr._mode = "push_to_talk"
    feed(mgr, 0.05, 100)
    assert not mgr._stop_requested            # PTT ends on key release only
