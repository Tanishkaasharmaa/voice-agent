"""Tests for the multi-character anime presets and preview rendering."""

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ------------------------------------------------------------- registry


def test_presets_registry_shape():
    from ui.character_widget import CHARACTER_PRESETS, DEFAULT_CHARACTER

    assert DEFAULT_CHARACTER in CHARACTER_PRESETS
    assert len(CHARACTER_PRESETS) >= 4          # a real choice, not two
    for pid, preset in CHARACTER_PRESETS.items():
        assert preset["label"], pid
        assert preset["style"] in ("twin_tails", "long", "bob", "ponytail")
        for key in ("hair_dark", "hair_light", "eye", "dress_top", "dress_bot"):
            assert preset[key].startswith("#"), (pid, key)


def test_presets_are_visually_distinct():
    from ui.character_widget import CHARACTER_PRESETS

    palettes = {(p["hair_light"], p["eye"]) for p in CHARACTER_PRESETS.values()}
    assert len(palettes) == len(CHARACTER_PRESETS)


# -------------------------------------------------------------- painting


def test_render_preview_produces_opaque_pixels(qapp):
    import numpy as np
    from ui.character_widget import CHARACTER_PRESETS, render_preview

    for pid in CHARACTER_PRESETS:
        img = render_preview(pid, 100)
        assert not img.isNull()
        ptr = img.constBits()
        arr = np.asarray(ptr).reshape(img.height(), img.bytesPerLine())
        # alpha channel (ARGB32_Premultiplied little-endian: every 4th byte)
        alpha = arr[:, 3::4]
        assert (alpha > 0).sum() > 100   # actually drew something


def test_previews_differ_between_characters(qapp):
    from ui.character_widget import CHARACTER_PRESETS, render_preview

    images = [bytes(render_preview(pid, 80).constBits())
              for pid in list(CHARACTER_PRESETS)[:2]]
    assert images[0] != images[1]


def test_paint_character_all_states_smoke(qapp):
    """Every preset must survive a paint in every agent state."""
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtCore import Qt
    from ui.character_widget import (
        CHARACTER_PRESETS, CharacterAnimator, paint_character,
    )

    states = ["IDLE", "LISTENING", "TRANSCRIBING", "THINKING",
              "EXECUTING_TOOL", "SPEAKING", "ERROR"]
    for pid, preset in CHARACTER_PRESETS.items():
        for state in states:
            animator = CharacterAnimator()
            animator.set_state(state)
            if state == "SPEAKING":
                animator.set_mouth_level(1.0)
            animator.tick(0.033)
            img = QImage(140, 200, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            painter = QPainter(img)
            try:
                paint_character(painter, animator, preset, 140, 200)
            finally:
                painter.end()
            assert not img.isNull()


# ---------------------------------------------------------------- widget


def test_widget_preset_from_settings(qapp, tmp_path):
    from app.config import Settings
    from ui.character_widget import CharacterWidget

    settings = Settings(path=tmp_path / "config.json")
    settings.set("character.preset", "sakura")
    widget = CharacterWidget(settings)
    assert widget.preset_id == "sakura"
    assert widget.preset["label"] == "Sakura"


def test_widget_switch_preset_live(qapp, tmp_path):
    from app.config import Settings
    from ui.character_widget import CharacterWidget

    settings = Settings(path=tmp_path / "config.json")
    widget = CharacterWidget(settings)
    assert widget.set_preset("yuki") is True
    assert widget.preset_id == "yuki"
    assert widget.set_preset("nope") is False
    assert widget.preset_id == "yuki"           # unknown id keeps current


def test_widget_still_transparent_for_mouse(qapp, tmp_path):
    """The character window must never steal hover/clicks from the island."""
    from app.config import Settings
    from PySide6.QtCore import Qt
    from ui.character_widget import CharacterWidget

    widget = CharacterWidget(Settings(path=tmp_path / "config.json"))
    assert widget.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


# --------------------------------------------------------------- settings


def test_settings_dialog_character_page(qapp, tmp_path):
    from ui.settings import SettingsDialog
    from app.config import Settings

    settings = Settings(path=tmp_path / "config.json")
    dlg = SettingsDialog(settings)
    row = dlg.sidebar.findItems("Character",
                                __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly)
    assert row, "Character section missing from settings sidebar"
    assert dlg.cmb_character.count() >= 4
    assert dlg.cmb_character.currentData() == "luna"
    # Preview thumbnail rendered.
    assert dlg.lbl_char_preview.pixmap() is not None


def test_settings_dialog_saves_character_choice(qapp, tmp_path):
    from ui.settings import SettingsDialog
    from app.config import Settings

    settings = Settings(path=tmp_path / "config.json")
    dlg = SettingsDialog(settings)
    idx = dlg.cmb_character.findData("mikan")
    dlg.cmb_character.setCurrentIndex(idx)
    dlg.spin_char_scale.setValue(1.5)
    dlg._save()
    saved = Settings(path=tmp_path / "config.json")
    assert saved.get("character.preset") == "mikan"
    assert saved.get("character.scale") == 1.5