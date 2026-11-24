from vad.base import SpeechActivityDetector


def make_detector(**kwargs):
    started = []
    ended = []
    params = {
        "frame_duration_s": 0.032,
        "silence_duration": 0.8,
        "minimum_speech_duration": 0.3,
        "maximum_duration": 60.0,
        "sensitivity": 0.5,
    }
    params.update(kwargs)
    detector = SpeechActivityDetector(
        on_speech_start=lambda: started.append(True),
        on_speech_end=lambda d: ended.append(d),
        **params,
    )
    detector.started = started
    detector.ended = ended
    return detector


def _feed(detector, speech_frames: int, silence_frames: int, start_ts: float = 0.0):
    ts = start_ts
    completed = False
    for _ in range(speech_frames):
        completed = detector.feed(0.9, ts) or completed
        ts += detector.frame_duration_s
    for _ in range(silence_frames):
        completed = detector.feed(0.05, ts) or completed
        ts += detector.frame_duration_s
    return completed


def test_speech_start_and_end():
    detector = make_detector()
    assert _feed(detector, speech_frames=30, silence_frames=30) is True  # ~0.96s speech, ~0.96s silence
    assert len(detector.started) == 1
    assert len(detector.ended) == 1


def test_short_blip_ignored():
    detector = make_detector()
    # 2 frames (~64ms) — below minimum_speech_duration: no utterance
    assert _feed(detector, speech_frames=2, silence_frames=10) is False
    assert detector.ended == []


def test_silence_duration_respected():
    detector = make_detector(silence_duration=0.5)
    ts = 0.0
    for _ in range(10):  # 0.32s speech
        detector.feed(0.9, ts)
        ts += detector.frame_duration_s
    # 5 silence frames (0.16s) < 0.5s -> no end yet
    for _ in range(5):
        assert detector.feed(0.05, ts) is False
        ts += detector.frame_duration_s
    # cross the 0.5s silence boundary
    ended = False
    while not ended:
        ended = detector.feed(0.05, ts)
        ts += detector.frame_duration_s
    assert ended is True


def test_maximum_duration_ends_utterance():
    detector = make_detector(maximum_duration=1.0)
    completed = False
    ts = 0.0
    for _ in range(40):  # 1.28s of continuous speech
        completed = detector.feed(0.9, ts) or completed
        ts += detector.frame_duration_s
    assert completed is True


def test_sensitivity_lowers_threshold():
    low = make_detector(sensitivity=0.9)
    high = make_detector(sensitivity=0.0)
    assert low.threshold < high.threshold
    # ambiguous 0.35 probability counts as speech for high sensitivity
    d = make_detector(sensitivity=0.9)
    d.feed(0.35, 0.0)
    assert d.in_speech is True


def test_reset_clears_state():
    detector = make_detector()
    _feed(detector, speech_frames=20, silence_frames=5)
    assert detector.in_speech is True
    detector.reset()
    assert detector.in_speech is False


def test_silero_v5_context_regression(tmp_path):
    """Regression: the Silero v5 ONNX model needs the last 64 samples of the
    previous frame prepended to each 512-sample frame (576-sample input).
    Without it the model returns ~0.001 for ALL audio — real speech included
    — so speech start/end never fired and auto-stop kept listening forever.
    Uses Windows built-in SAPI TTS to synthesize real speech (offline); the
    old code produced no speech-start event even for this audio."""
    import os
    import subprocess
    import wave
    from pathlib import Path

    import numpy as np
    import pytest

    models_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "LocalVoiceAgent" / "models"
    model_path = models_dir / "silero_vad.onnx"
    if not model_path.exists():
        pytest.skip("silero_vad.onnx not installed")

    wav_path = tmp_path / "speech.wav"
    ps = (
        "$ErrorActionPreference='Stop';"
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$fmt=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        "[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{wav_path}',$fmt);"
        "$s.Speak('Hello there how are you today');"
        "$s.SetOutputToDefaultAudioDevice()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=30, check=True)
    except Exception:
        pytest.skip("Windows SAPI TTS unavailable")

    with wave.open(str(wav_path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16).astype(np.float32) / 32767.0

    from vad.silero import SileroVAD

    vad = SileroVAD(model_path)
    vad.load()

    started, ended = [], []
    det = SpeechActivityDetector(
        frame_duration_s=512 / 16000, silence_duration=0.8,
        minimum_speech_duration=0.3, maximum_duration=60, sensitivity=0.5,
        on_speech_start=lambda: started.append(True),
        on_speech_end=lambda d: ended.append(d),
    )
    for i in range(0, len(audio) - 511, 512):
        det.feed(vad.probability(audio[i:i + 512]))
    # Trailing silence: the utterance must complete.
    for _ in range(int(1.6 / (512 / 16000))):
        det.feed(vad.probability(np.zeros(512, dtype=np.float32)))

    assert started, "speech start never detected (context-less v5 feed bug)"
    assert ended, "speech end never detected"
