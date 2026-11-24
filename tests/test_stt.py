"""Whisper engine unit tests — no faster-whisper install required."""

from __future__ import annotations

import sys
import types

import pytest

from stt.whisper_engine import WhisperEngine, local_model_dir


def _fake_faster_whisper(monkeypatch, model_cls) -> None:
    """Stub the ``faster_whisper`` module so load() never touches the network
    or a real install."""
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = model_cls
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


def test_local_model_dir_naming(tmp_path):
    assert local_model_dir(tmp_path, "small") == tmp_path / "whisper-small"
    assert local_model_dir(tmp_path, "base") == tmp_path / "whisper-base"


def test_load_uses_local_path_when_models_dir_given(monkeypatch, tmp_path):
    captured: dict = {}

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    model_dir = local_model_dir(tmp_path, "small")
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"fake")

    engine = WhisperEngine(model="small", models_dir=tmp_path)
    monkeypatch.setattr(engine, "is_available", lambda: True)
    _fake_faster_whisper(monkeypatch, FakeWhisperModel)

    engine.load()

    assert captured["args"][0] == str(model_dir)
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["compute_type"] == "int8"


def test_load_raises_when_local_model_missing(monkeypatch, tmp_path):
    engine = WhisperEngine(model="small", models_dir=tmp_path)
    monkeypatch.setattr(engine, "is_available", lambda: True)
    _fake_faster_whisper(monkeypatch, lambda *a, **k: None)

    with pytest.raises(FileNotFoundError, match="Setup dialog"):
        engine.load()


def test_load_falls_back_to_model_name_without_models_dir(monkeypatch):
    captured: dict = {}

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            captured["args"] = args

    engine = WhisperEngine(model="small", models_dir=None)
    monkeypatch.setattr(engine, "is_available", lambda: True)
    _fake_faster_whisper(monkeypatch, FakeWhisperModel)

    engine.load()

    assert captured["args"][0] == "small"
