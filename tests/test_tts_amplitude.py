"""Tests for the amplitude plumbing: Speaker callback + TTSManager forward."""

import queue
import threading
import time

import numpy as np
import pytest

from audio.speaker import Speaker


class FakeTtsEngine:
    def is_available(self):
        return True

    def load(self, voice=None):
        pass

    def synthesize(self, text):
        return np.zeros(16, dtype=np.float32), 24000

    def unload(self):
        pass

    @property
    def display_name(self):
        return "fake"


def test_emit_amplitude_computes_rms_and_clamps():
    s = Speaker()
    seen = []
    s.on_amplitude = seen.append

    silence = np.zeros(100, dtype=np.float32)
    loud = np.full(100, 2.0, dtype=np.float32)   # RMS 2.0 -> clamped to 1.0
    quiet = np.full(100, 0.25, dtype=np.float32)

    s._emit_amplitude(silence)
    s._emit_amplitude(loud)
    s._emit_amplitude(quiet)

    assert seen[0] == 0.0
    assert seen[1] == 1.0
    assert abs(seen[2] - 0.25) < 1e-6


def test_emit_amplitude_without_callback_is_safe():
    s = Speaker()
    s._emit_amplitude(np.ones(10, dtype=np.float32))  # must not raise


def test_tts_manager_forwards_speaker_amplitude():
    from tts.base import TTSManager

    speaker = Speaker()
    manager = TTSManager(FakeTtsEngine(), speaker)
    # The manager wired itself as the speaker's amplitude receiver.
    assert speaker.on_amplitude is not None

    received = []
    manager.on_amplitude = lambda level: received.append(level)
    speaker._emit_amplitude(np.full(8, 0.5, dtype=np.float32))
    assert len(received) == 1
    assert abs(received[0] - 0.5) < 1e-6


def test_broken_amplitude_callback_does_not_break_playback_path():
    s = Speaker()

    def boom(_level):
        raise RuntimeError("boom")

    s.on_amplitude = boom
    s._emit_amplitude(np.ones(4, dtype=np.float32))  # swallowed internally


def test_play_reports_playing_immediately_despite_worker_latency():
    """Regression: is_playing() must be true the instant play() returns,
    even if the worker thread has not opened the audio device yet —
    otherwise the agent drops out of SPEAKING before audio is heard."""
    s = Speaker()
    started = threading.Event()

    def slow_worker(self):
        started.set()
        time.sleep(0.3)  # simulate sounddevice import + stream open
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._playing.clear()

    s._run = slow_worker.__get__(s, Speaker)  # bind as instance method
    s.play(np.zeros(100, dtype=np.float32))
    assert started.wait(1.0)
    # Worker is still "opening the device" but playing must already be true.
    assert s.is_playing() is True
    deadline = time.time() + 2.0
    while s.is_playing() and time.time() < deadline:
        time.sleep(0.02)
    assert s.is_playing() is False


def test_queued_second_item_keeps_playing_flag_between_items(monkeypatch):
    """While one item plays and another sits queued, is_playing must never
    flicker false — the pipeline drains on exactly that flag."""
    import sys
    import types

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, chunk):
            time.sleep(0.05)  # each ~10ms-chunk takes 50ms -> observable window

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = lambda **kwargs: FakeStream()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    s = Speaker()
    s.play(np.zeros(2000, dtype=np.float32))
    time.sleep(0.15)                      # worker started on item 1
    s.play(np.zeros(2000, dtype=np.float32))  # queue item 2 mid-playback

    flickered = False
    deadline = time.time() + 3.0
    while time.time() < deadline and not s._queue.empty():
        if not s.is_playing():
            flickered = True
            break
        time.sleep(0.004)
    assert not flickered
    s.stop()


def test_character_config_defaults(tmp_path):
    from app.config import Settings

    settings = Settings(path=tmp_path / "config.json")
    assert settings.get("character.enabled") is True
    assert settings.get("character.scale") == 1.0
    assert settings.get("character.offset") == [-30, -190]
