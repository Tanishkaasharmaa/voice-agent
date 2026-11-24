"""Smoke tests for the simplified first-run wizard."""

import pytest

from app.config import Settings

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def make_wizard(tmp_path):
    from ui.setup_wizard import FirstRunWizard

    settings = Settings(path=tmp_path / "config.json")
    wizard = FirstRunWizard(services=None)
    return wizard, settings


def test_wizard_defaults_to_hybrid_ollama(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    assert wizard.connectivity == "hybrid"
    assert wizard.provider == "ollama"
    assert wizard.agent_name == "Luna"


def test_wizard_online_mode_provider_properties(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    wizard.rb_online.setChecked(True)
    # Default provider selection is Ollama Cloud.
    assert wizard.provider == "ollama"
    # Switch to OpenRouter: API key + model come from the registry.
    idx = wizard.cmb_provider.findData("openrouter")
    wizard.cmb_provider.setCurrentIndex(idx)
    wizard._on_provider_changed()
    assert wizard.provider == "openrouter"
    assert wizard.provider_model in ("openrouter/auto",
                                     "anthropic/claude-sonnet-4.5")
    wizard.ed_api_key.setText("sk-or-test")
    assert wizard.api_key == "sk-or-test"
    assert wizard.cloud_model == ""      # non-Ollama providers use no cloud tag


def test_wizard_offline_mode_local_model(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    wizard.rb_offline.setChecked(True)
    wizard.cmb_local_offline.setCurrentText("qwen3.5:4b")
    assert wizard.connectivity == "offline"
    assert wizard.local_model == "qwen3.5:4b"
    assert wizard.cloud_model == ""
    assert wizard.api_key == ""


def test_wizard_hybrid_reports_both_models(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    assert wizard.connectivity == "hybrid"
    wizard.cmb_hybrid_cloud.setCurrentText("gemma4:cloud")
    wizard.cmb_local_hybrid.setCurrentText("qwen3.5:2b")
    assert wizard.cloud_model == "gemma4:cloud"
    assert wizard.local_model == "qwen3.5:2b"


# ------------------------------------------------------ full walkthroughs


def _walk(wizard):
    """Drive Next through every page like a user would."""
    pages = []
    while True:
        pages.append(wizard.stack.currentIndex())
        if wizard.stack.currentIndex() == wizard.stack.count() - 1:
            break
        wizard._next()
    return pages


def test_no_html_in_radio_labels(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    for rb in (wizard.rb_hybrid, wizard.rb_offline, wizard.rb_online):
        assert "<" not in rb.text(), rb.text()


def test_model_combos_are_select_only(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    assert wizard.cmb_local_offline.isEditable() is False
    assert wizard.cmb_local_hybrid.isEditable() is False
    assert wizard.cmb_hybrid_cloud.isEditable() is False
    wizard.rb_online.setChecked(True)
    assert wizard.cmb_cloud_model.isEditable() is False


def test_full_walkthrough_hybrid(qapp, tmp_path):
    wizard, settings = make_wizard(tmp_path)
    pages = _walk(wizard)
    assert len(pages) == 4                      # names -> mode -> model -> done
    # Re-entering the config page must switch to the hybrid sub-page.
    wizard.stack.setCurrentIndex(1)
    wizard.rb_hybrid.setChecked(True)
    wizard._next()
    assert wizard.cfg_stack.currentIndex() == 2
    summary = wizard.lbl_summary.text()
    assert "hybrid" in summary.lower()
    assert wizard.local_model in wizard.DEFAULT_LOCAL_MODELS
    assert wizard.cloud_model in wizard.DEFAULT_CLOUD_MODELS


def test_full_walkthrough_offline(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    wizard.rb_offline.setChecked(True)
    wizard._next()          # names -> mode page
    wizard._next()          # mode -> config page (syncs the sub-page)
    assert wizard.cfg_stack.currentIndex() == 0   # offline sub-page
    assert wizard.connectivity == "offline"
    assert wizard.provider == "ollama"
    assert wizard.cloud_model == ""
    assert "offline" in wizard._summary_text().lower()


def test_full_walkthrough_online_openrouter(qapp, tmp_path):
    wizard, _ = make_wizard(tmp_path)
    wizard.rb_online.setChecked(True)
    wizard._next()          # names -> mode page
    wizard._next()          # mode -> config page (syncs the sub-page)
    assert wizard.stack.currentIndex() == 2
    assert wizard.cfg_stack.currentIndex() == 1   # online sub-page
    idx = wizard.cmb_provider.findData("openrouter")
    wizard.cmb_provider.setCurrentIndex(idx)
    wizard._on_provider_changed()
    assert wizard.provider == "openrouter"
    assert wizard.provider_model                  # default model preselected
    assert wizard.api_key == ""                   # key empty until typed


def test_walkthrough_persists_and_checks_match(tmp_path, qapp, monkeypatch):
    """The settings a finished wizard produces must drive dependency checks
    correctly for each mode (this mirrors _run_first_run_wizard's writes)."""
    from setup import dependency_checks as checks

    def apply(settings, connectivity, provider="ollama",
              provider_model="", local_model="qwen3.5:4b", cloud_model=""):
        settings.set("llm.connectivity", connectivity)
        settings.set("llm.provider", provider)
        if provider != "ollama":
            settings.set("llm.provider_model", provider_model)
        elif cloud_model:
            settings.set("llm.cloud_model", cloud_model)
        settings.set("llm.local_model", local_model)
        settings.set("llm.use_cloud_models", connectivity != "offline")

    # Offline: local model demanded, no cloud probe.
    s = Settings(path=tmp_path / "offline.json")
    apply(s, "offline", local_model="qwen3.5:2b")
    statuses = checks.run_all_checks(s, client=None, cloud_model=None)
    assert any(s_.name == "qwen3.5:2b" and not s_.present for s_ in statuses)

    # Online + hosted provider: neither Ollama nor a local model required.
    s = Settings(path=tmp_path / "online.json")
    apply(s, "online", provider="openrouter", provider_model="openrouter/auto")
    statuses = checks.run_all_checks(s, client=None, cloud_model=None)
    missing = checks.missing_statuses(statuses)
    assert all(s_.name != "Ollama" for s_ in missing)
    assert all(s_.name != "openrouter/auto" for s_ in missing)
    assert any("Not required" in s_.message for s_ in statuses)

    # Hybrid: local model required AND cloud probe attempted.
    s = Settings(path=tmp_path / "hybrid.json")
    apply(s, "hybrid", local_model="qwen3.5:4b", cloud_model="gemma4:cloud")
    statuses = checks.run_all_checks(s, client=None, cloud_model="gemma4:cloud")
    assert any(s_.name == "qwen3.5:4b" for s_ in statuses)
    assert any(s_.name.startswith("Cloud model") for s_ in statuses)
