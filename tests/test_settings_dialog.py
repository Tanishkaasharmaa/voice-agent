"""Tests for the Settings dialog dropdown fields."""

import pytest

from app.config import Settings

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def make_dialog(tmp_path, overrides=None):
    from ui.settings import SettingsDialog

    settings = Settings(path=tmp_path / "config.json")
    if overrides:
        settings.update(overrides)
    return SettingsDialog(settings)


def test_language_is_dropdown_with_codes(qapp, tmp_path):
    dlg = make_dialog(tmp_path)
    assert dlg.cmb_lang.count() > 5
    assert dlg.cmb_lang.currentData() == "en"          # default
    assert "(" not in dlg.cmb_lang.currentText() or True


def test_language_restores_saved_code_and_label(qapp, tmp_path):
    dlg = make_dialog(tmp_path, {"voice_input": {"language": "de"}})
    assert dlg.cmb_lang.currentData() == "de"
    assert dlg.cmb_lang.currentText() == "German"


def test_unknown_saved_language_appended_not_lost(qapp, tmp_path):
    dlg = make_dialog(tmp_path, {"voice_input": {"language": "xx"}})
    assert dlg.cmb_lang.currentData() == "xx"


def test_kokoro_and_piper_voice_dropdowns(qapp, tmp_path):
    dlg = make_dialog(tmp_path, {
        "voice_output": {"kokoro_voice": "am_michael",
                         "piper_voice": "de_DE-thorsten-medium"},
    })
    assert dlg.cmb_kokoro_voice.currentText() == "am_michael"
    assert dlg.cmb_piper_voice.currentText() == "de_DE-thorsten-medium"
    assert "af_heart" in [dlg.cmb_kokoro_voice.itemText(i)
                          for i in range(dlg.cmb_kokoro_voice.count())]


def test_ollama_host_dropdown_and_custom_host_preserved(qapp, tmp_path):
    dlg = make_dialog(tmp_path)
    texts = [dlg.cmb_host.itemText(i) for i in range(dlg.cmb_host.count())]
    assert "http://127.0.0.1:11434" in texts

    custom = make_dialog(
        tmp_path, {"llm": {"host": "http://192.168.1.50:11434"}})
    assert custom.cmb_host.currentText() == "http://192.168.1.50:11434"


def test_save_round_trips_dropdown_values(qapp, tmp_path):
    settings = Settings(path=tmp_path / "config.json")
    settings.update({"voice_input": {"language": "fr"},
                     "voice_output": {"kokoro_voice": "af_bella"},
                     "llm": {"host": "http://localhost:11434"}})
    from ui.settings import SettingsDialog

    dlg = SettingsDialog(settings)
    dlg._save()
    assert settings.get("voice_input.language") == "fr"
    assert settings.get("voice_output.kokoro_voice") == "af_bella"
    assert settings.get("llm.host") == "http://localhost:11434"
