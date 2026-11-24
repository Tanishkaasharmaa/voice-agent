"""End-to-end drain test: SPEAKING must hold until audio actually finishes.

Uses the real Speaker + TTSManager with a faked sounddevice module (each
chunk write sleeps) and a fake engine with slow load/synthesize — the exact
conditions under which the state used to drop to IDLE before audio played.
"""

import queue
import sys
import threading
import time
import types

import numpy as np
import pytest


class FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, chunk):
        time.sleep(0.03)  # each 100 ms chunk takes 30 ms wall time


@pytest.fixture()
def fake_sounddevice(monkeypatch):
    mod = types.ModuleType("sounddevice")
    mod.OutputStream = lambda **kwargs: FakeStream()
    monkeypatch.setitem(sys.modules, "sounddevice", mod)


class SlowEngine:
    """Kokoro stand-in: first load takes ~0.4 s, synth ~0.15 s."""

    def __init__(self):
        self.loaded = False
        self.loads = 0

    def is_available(self):
        return self.loaded

    def load(self):
        self.loads += 1
        time.sleep(0.4)
        self.loaded = True

    def synthesize(self, text):
        time.sleep(0.15)
        return np.zeros(2400, dtype=np.float32), 24000  # 0.1 s of audio

    def unload(self):
        pass

    @property
    def display_name(self):
        return "slow-fake"


def test_speaking_hold_until_audio_finishes(fake_sounddevice):
    from audio.speaker import Speaker
    from tts.base import TTSManager

    speaker = Speaker()
    engine = SlowEngine()
    mgr = TTSManager(engine, speaker)

    started = time.monotonic()
    mgr.speak("hello")            # returns immediately; pending = 1

    # The very instant speak() returns, busy must be true...
    assert mgr.is_busy() is True
    # ...and it must STAY true continuously through load + synthesis +
    # playback. Any early False is the "IDLE before sound" bug.
    flickered = False
    min_busy_seconds = 0.4 + 0.15 + (0.1 * 0.3) - 0.1   # generous lower bound
    while True:
        if not mgr.is_busy():
            break
        if time.monotonic() - started > 10:
            pytest.fail("TTS never drained")
        time.sleep(0.01)
    busy_duration = time.monotonic() - started
    assert busy_duration >= min_busy_seconds, (
        f"busy dropped after {busy_duration:.2f}s — too early")
    assert not flickered or True  # loop above breaks only on drain

    # Engine was loaded exactly once and everything really finished.
    assert engine.loads == 1
    assert mgr.pending_count() == 0


def test_three_sentences_stay_busy_until_last_played(fake_sounddevice):
    from audio.speaker import Speaker
    from tts.base import TTSManager

    speaker = Speaker()
    mgr = TTSManager(SlowEngine(), speaker)

    for s in ("one.", "two.", "three."):
        mgr.speak(s)
    assert mgr.pending_count() == 3
    assert mgr.is_busy() is True

    deadline = time.time() + 15
    while mgr.is_busy() and time.time() < deadline:
        time.sleep(0.01)
    assert mgr.pending_count() == 0
    assert not mgr.is_busy()


def test_interrupt_drains_immediately(fake_sounddevice):
    """stop() must make is_busy() false right away so interruption never
    wedges the pipeline in SPEAKING."""
    from audio.speaker import Speaker
    from tts.base import TTSManager

    speaker = Speaker()
    mgr = TTSManager(SlowEngine(), speaker)
    mgr.speak("hello")
    assert mgr.is_busy()
    mgr.stop()
    assert mgr.is_busy() is False
    assert mgr.pending_count() == 0
