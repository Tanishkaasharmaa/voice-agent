import json

from app.config import Settings


def test_defaults_present(tmp_path):
    settings = Settings(path=tmp_path / "config.json")
    assert settings.local_model == "qwen3.5:4b"
    assert settings.cloud_model == "deepseek-v4-flash:cloud"
    assert settings.agent_name == "Luna"
    assert settings.user_name == ""
    assert settings.local_model_choices[0] == "qwen3.5:0.8b"
    assert settings.local_model_choices[-1] == "nemotron-3-nano:4b"
    assert settings.cloud_model_choices[0] == "gemma4:cloud"
    assert settings.cloud_model_choices[-1] == "nemotron-3-nano:30b-cloud"
    assert settings.silence_duration == 0.8
    assert settings.get("profile") == "LOW_RESOURCE_CPU"


def test_save_and_reload_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    settings = Settings(path=path)
    settings.set("voice_input.silence_duration", 1.2)
    settings.set("llm.temperature", 0.5)
    settings.save()

    reloaded = Settings(path=path)
    assert reloaded.get("voice_input.silence_duration") == 1.2
    assert reloaded.get("llm.temperature") == 0.5


def test_update_merges_and_persists(tmp_path):
    path = tmp_path / "config.json"
    settings = Settings(path=path)
    settings.update({"llm": {"local_model": "gemma4:12b", "temperature": 0.9}})
    assert settings.get("llm.local_model") == "gemma4:12b"
    assert settings.get("llm.temperature") == 0.9
    # untouched keys survive
    assert settings.get("llm.cloud_model") == "deepseek-v4-flash:cloud"

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["llm"]["local_model"] == "gemma4:12b"


def test_unknown_key_returns_default(tmp_path):
    settings = Settings(path=tmp_path / "config.json")
    assert settings.get("does.not.exist", "fallback") == "fallback"


def test_dot_path_set(tmp_path):
    settings = Settings(path=tmp_path / "config.json")
    settings.set("memory.enabled", False)
    assert settings.get("memory.enabled") is False
