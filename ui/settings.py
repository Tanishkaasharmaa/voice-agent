"""Settings dialog — sections navigated from a left-hand sidebar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from app.config import AGENT_NAME_SUGGESTIONS
from audio.microphone import Microphone
from audio.speaker import Speaker
from stt.whisper_engine import SUPPORTED_MODELS

DEFAULT_LOCAL_MODELS = [
    "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b",
    "gemma4:e2b", "gemma4:e4b", "gemma4:12b", "nemotron-3-nano:4b",
]
DEFAULT_CLOUD_MODELS = [
    "gemma4:cloud", "gemma4:31b-cloud", "deepseek-v4-flash:cloud",
    "glm-5.2:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "minimax-m3:cloud",
    "qwen3.5:cloud", "qwen3.5:397b-cloud", "nemotron-3-nano:30b-cloud",
]

# Whisper languages: (display label, ISO code stored in config).
# "" means auto-detect (faster-whisper treats a falsy language as None).
STT_LANGUAGES = [
    ("Auto-detect", ""),
    ("English", "en"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Russian", "ru"),
    ("Ukrainian", "uk"),
    ("Turkish", "tr"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
]

# Kokoro v1.0 voices shipped in voices-v1.0.bin.
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_nova", "af_sky", "af_alloy",
    "af_aoede", "af_jessica", "af_kore", "af_river", "af_sarah",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

# Common Piper voice packs (medium quality unless noted).
PIPER_VOICES = [
    "en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-high",
    "en_US-arctic-medium", "en_US-hfc_female-medium", "en_US-joe-medium",
    "en_GB-alan-medium", "en_GB-alba-medium", "en_GB-northern_english_male-medium",
    "de_DE-thorsten-medium", "es_ES-sharvard-medium", "fr_FR-siwis-medium",
    "it_IT-riccardo-x_low", "pt_BR-faber-medium", "ru_RU-irina-medium",
    "tr_TR-dfki-medium", "uk_UA-ukrainian_tts-medium", "zh_CN-huayan-medium",
]

# Typical Ollama endpoints.
OLLAMA_HOSTS = [
    "http://127.0.0.1:11434",
    "http://localhost:11434",
]


class SettingsDialog(QDialog):
    def __init__(self, settings, dependency_manager=None, memory=None, parent=None,
                 cloud_auth_failed: bool = False) -> None:
        super().__init__(parent)
        self.settings = settings
        self.dependency_manager = dependency_manager
        self.memory = memory
        self.cloud_auth_failed = cloud_auth_failed
        self.setWindowTitle("LocalVoiceAgent — Settings")
        self.resize(640, 560)

        layout = QVBoxLayout(self)
        content = QHBoxLayout()
        layout.addLayout(content)

        # Sidebar navigation: section list on the left, pages on the right.
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("settingsSidebar")
        self.sidebar.setFixedWidth(150)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pages = QStackedWidget()
        content.addWidget(self.sidebar)
        content.addWidget(self.pages, 1)
        self.sidebar.currentRowChanged.connect(self._on_section_changed)

        self._build_general()
        self._build_character()
        self._build_voice_input()
        self._build_voice_output()
        self._build_llm()
        self._build_hotkeys()
        self._build_memory()
        self._build_tools()
        self._build_dependencies()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self.sidebar.count() > 0:
            self.sidebar.setCurrentRow(0)

    def _on_section_changed(self, row: int) -> None:
        self.pages.setCurrentIndex(max(0, row))

    # --------------------------------------------------------------- sections

    def _add_page(self, title: str, page: QWidget) -> QWidget:
        """Register an already-built page under a sidebar entry."""
        self.pages.addWidget(page)
        self.sidebar.addItem(title)
        return page

    def _add_tab(self, title: str, fields: list) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        for label, widget in fields:
            form.addRow(label, widget)
        return self._add_page(title, page)

    @staticmethod
    def _fill_device_combo(combo: QComboBox, devices: list[dict], current: str | None) -> None:
        """Populate a non-editable dropdown with the available audio devices
        (plus "Default"), preserving the saved selection if it still exists."""
        combo.addItem("Default")
        seen: set[str] = set()
        for dev in devices:
            name = dev.get("name", "")
            if name and name not in seen:
                seen.add(name)
                combo.addItem(name)
        if current and combo.findText(current) >= 0:
            combo.setCurrentText(current)
        elif current:
            combo.addItem(current)
            combo.setCurrentText(current)

    @staticmethod
    def _select_current(combo: QComboBox, current: str) -> None:
        """Select ``current`` in a choices dropdown, appending it if the saved
        value is no longer one of the choices."""
        if current and combo.findText(current) >= 0:
            combo.setCurrentText(current)
        elif current:
            combo.addItem(current)
            combo.setCurrentText(current)

    def _build_character(self) -> None:
        from ui.character_widget import CHARACTER_PRESETS, render_preview

        s = self.settings
        page = QWidget()
        layout = QVBoxLayout(page)

        row = QHBoxLayout()
        form = QFormLayout()
        self.cmb_character = QComboBox()
        for pid, preset in CHARACTER_PRESETS.items():
            self.cmb_character.addItem(preset["label"], pid)
        saved = str(s.get("character.preset", "luna"))
        idx = self.cmb_character.findData(saved)
        self.cmb_character.setCurrentIndex(max(0, idx))
        self.cmb_character.currentIndexChanged.connect(self._on_character_changed)
        form.addRow("Character", self.cmb_character)

        self.chk_char_enabled = QCheckBox()
        self.chk_char_enabled.setChecked(bool(s.get("character.enabled", True)))
        form.addRow("Show character", self.chk_char_enabled)

        self.spin_char_scale = QDoubleSpinBox()
        self.spin_char_scale.setRange(0.5, 2.0)
        self.spin_char_scale.setSingleStep(0.1)
        self.spin_char_scale.setValue(float(s.get("character.scale", 1.0)))
        form.addRow("Size", self.spin_char_scale)
        row.addLayout(form)

        # Live thumbnail of the selected character.
        self.lbl_char_preview = QLabel()
        self.lbl_char_preview.setMinimumSize(140, 200)
        self.lbl_char_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_character_preview()
        row.addWidget(self.lbl_char_preview)
        layout.addLayout(row)

        hint = QLabel("The character reacts to the agent: it listens while "
                      "blue, thinks while purple, speaks (with lip sync) "
                      "while green. Changes apply on Save.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8;")
        layout.addWidget(hint)
        layout.addStretch(1)
        self._add_page("Character", page)

    def _on_character_changed(self) -> None:
        self._update_character_preview()

    def _update_character_preview(self) -> None:
        from ui.character_widget import render_preview

        pid = self.cmb_character.currentData() or "luna"
        image = render_preview(pid, 120)
        from PySide6.QtGui import QPixmap

        self.lbl_char_preview.setPixmap(QPixmap.fromImage(image))

    def _build_general(self) -> None:
        s = self.settings
        self.chk_start_with_windows = QCheckBox()
        self.chk_start_with_windows.setChecked(bool(s.get("general.start_with_windows")))
        self.chk_show_widget = QCheckBox()
        self.chk_show_widget.setChecked(bool(s.get("general.show_floating_widget", True)))
        self.cmb_theme = QComboBox()
        self.cmb_theme.addItems(["dark", "light"])
        self.cmb_theme.setCurrentText(s.get("general.theme", "dark"))
        self.cmb_agent_name = QComboBox()
        self.cmb_agent_name.setEditable(True)
        self.cmb_agent_name.addItems(AGENT_NAME_SUGGESTIONS)
        self._select_current(self.cmb_agent_name, s.get("general.agent_name", "Luna"))
        self.ed_user_name = QLineEdit(s.get("general.user_name", ""))
        self.ed_user_name.setPlaceholderText("e.g. Alex")
        self._add_tab("General", [
            ("Assistant name", self.cmb_agent_name),
            ("What it calls you", self.ed_user_name),
            ("Start with Windows", self.chk_start_with_windows),
            ("Show floating widget", self.chk_show_widget),
            ("Theme", self.cmb_theme),
        ])

    def _build_voice_input(self) -> None:
        s = self.settings
        self.cmb_mic = QComboBox()
        self._fill_device_combo(self.cmb_mic, Microphone().list_devices(),
                                s.get("voice_input.microphone"))
        self.cmb_stt = QComboBox()
        self.cmb_stt.addItems(["whisper"])
        self.cmb_stt.setCurrentText(s.get("voice_input.stt_engine", "whisper"))
        self.cmb_whisper = QComboBox()
        self.cmb_whisper.addItems(list(SUPPORTED_MODELS))
        self.cmb_whisper.setCurrentText(s.get("voice_input.whisper_model", "small"))
        self.cmb_lang = QComboBox()
        for label, code in STT_LANGUAGES:
            self.cmb_lang.addItem(label, code)
        saved_lang = s.get("voice_input.language", "en")
        lang_idx = self.cmb_lang.findData(saved_lang)
        # Unknown saved codes are appended so nothing is silently lost.
        if lang_idx < 0 and saved_lang:
            self.cmb_lang.addItem(saved_lang, saved_lang)
            lang_idx = self.cmb_lang.count() - 1
        self.cmb_lang.setCurrentIndex(max(0, lang_idx))
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, 16)
        self.spin_threads.setValue(int(s.get("voice_input.cpu_threads", 0)))
        self.spin_silence = QDoubleSpinBox()
        self.spin_silence.setRange(0.1, 5.0)
        self.spin_silence.setSingleStep(0.1)
        self.spin_silence.setValue(float(s.get("voice_input.silence_duration", 0.8)))
        self.spin_min_speech = QDoubleSpinBox()
        self.spin_min_speech.setRange(0.1, 2.0)
        self.spin_min_speech.setSingleStep(0.05)
        self.spin_min_speech.setValue(float(s.get("voice_input.minimum_speech_duration", 0.3)))
        self.spin_max_rec = QSpinBox()
        self.spin_max_rec.setRange(5, 300)
        self.spin_max_rec.setValue(int(s.get("voice_input.maximum_recording_duration", 60)))
        self.spin_sensitivity = QDoubleSpinBox()
        self.spin_sensitivity.setRange(0.0, 1.0)
        self.spin_sensitivity.setSingleStep(0.1)
        self.spin_sensitivity.setValue(float(s.get("voice_input.vad_sensitivity", 0.5)))
        self._add_tab("Voice Input", [
            ("Microphone", self.cmb_mic),
            ("STT engine", self.cmb_stt),
            ("Whisper model", self.cmb_whisper),
            ("Language", self.cmb_lang),

            ("CPU threads (0=auto)", self.spin_threads),
            ("VAD sensitivity", self.spin_sensitivity),
            ("Silence duration (s)", self.spin_silence),
            ("Min speech (s)", self.spin_min_speech),
            ("Max recording (s)", self.spin_max_rec),
        ])

    def _build_voice_output(self) -> None:
        s = self.settings
        self.cmb_speaker = QComboBox()
        self._fill_device_combo(self.cmb_speaker, Speaker().list_devices(),
                                s.get("voice_output.speaker"))
        self.cmb_tts = QComboBox()
        self.cmb_tts.addItems(["kokoro", "piper"])
        self.cmb_tts.setCurrentText(s.get("voice_output.tts_engine", "kokoro"))
        self.cmb_kokoro_voice = QComboBox()
        self.cmb_kokoro_voice.addItems(KOKORO_VOICES)
        self._select_current(self.cmb_kokoro_voice,
                             s.get("voice_output.kokoro_voice", "af_heart"))
        self.cmb_piper_voice = QComboBox()
        self.cmb_piper_voice.addItems(PIPER_VOICES)
        self._select_current(self.cmb_piper_voice,
                             s.get("voice_output.piper_voice", "en_US-lessac-medium"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.5, 2.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setValue(float(s.get("voice_output.speed", 1.0)))
        self.spin_volume = QDoubleSpinBox()
        self.spin_volume.setRange(0.0, 1.0)
        self.spin_volume.setSingleStep(0.05)
        self.spin_volume.setValue(float(s.get("voice_output.volume", 1.0)))
        self._add_tab("Voice Output", [
            ("Speaker", self.cmb_speaker),
            ("TTS engine", self.cmb_tts),
            ("Kokoro voice", self.cmb_kokoro_voice),
            ("Piper voice", self.cmb_piper_voice),
            ("Speed", self.spin_speed),
            ("Volume", self.spin_volume),
        ])

    def _build_llm(self) -> None:
        s = self.settings
        self.cmb_connectivity = QComboBox()
        self.cmb_connectivity.addItem("Hybrid — cloud when online", "hybrid")
        self.cmb_connectivity.addItem("Fully online", "online")
        self.cmb_connectivity.addItem("Fully offline", "offline")
        cur = s.get("llm.connectivity", "hybrid")
        idx = self.cmb_connectivity.findData(cur)
        self.cmb_connectivity.setCurrentIndex(max(0, idx))

        from ollama.providers import PROVIDERS

        self.cmb_llm_provider = QComboBox()
        for pid, meta in PROVIDERS.items():
            self.cmb_llm_provider.addItem(meta["label"], pid)
        prov_idx = self.cmb_llm_provider.findData(s.get("llm.provider", "ollama"))
        self.cmb_llm_provider.setCurrentIndex(max(0, prov_idx))
        self.cmb_llm_provider.currentIndexChanged.connect(self._on_llm_provider_changed)

        # Host / local / cloud rows (Ollama family).
        self.cmb_host = QComboBox()
        self.cmb_host.addItems(OLLAMA_HOSTS)
        # A saved custom host (e.g. another machine on the LAN) is appended
        # so it stays selectable.
        self._select_current(self.cmb_host,
                             s.get("llm.host", "http://127.0.0.1:11434"))
        self.cmb_local = QComboBox()
        self.cmb_local.addItems(s.get("llm.local_model_choices", DEFAULT_LOCAL_MODELS))
        self._select_current(self.cmb_local, s.get("llm.local_model", "qwen3.5:4b"))
        self.cmb_cloud = QComboBox()
        self.cmb_cloud.addItems(s.get("llm.cloud_model_choices", DEFAULT_CLOUD_MODELS))
        self._select_current(self.cmb_cloud,
                             s.get("llm.cloud_model", "deepseek-v4-flash:cloud"))

        # Hosted provider rows.
        self.ed_provider_model = QLineEdit(
            s.get("llm.provider_model", "") or
            (PROVIDERS.get(s.get("llm.provider", "ollama"), {}).get("models") or [""])[0])
        self.ed_provider_key = QLineEdit()
        self.ed_provider_key.setEchoMode(QLineEdit.EchoMode.Password)
        saved_prov = s.get("llm.provider", "ollama")
        if saved_prov != "ollama":
            self.ed_provider_key.setText(s.api_key(saved_prov))

        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(1024, 32768)
        self.spin_ctx.setValue(int(s.get("llm.context_size", 4096)))
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 2.0)
        self.spin_temp.setSingleStep(0.1)
        self.spin_temp.setValue(float(s.get("llm.temperature", 0.2)))
        self.chk_tools = QCheckBox()
        self.chk_tools.setChecked(bool(s.get("llm.tools_enabled", True)))

        page = QWidget()
        form = QFormLayout(page)
        form.addRow("Connection", self.cmb_connectivity)
        form.addRow("Provider", self.cmb_llm_provider)
        form.addRow("Provider model", self.ed_provider_model)
        form.addRow("API key", self.ed_provider_key)
        form.addRow("Ollama host", self.cmb_host)
        form.addRow("Local model (offline)", self.cmb_local)
        form.addRow("Cloud model (online)", self.cmb_cloud)
        form.addRow("Context size", self.spin_ctx)
        form.addRow("Temperature", self.spin_temp)
        form.addRow("Tools enabled", self.chk_tools)
        hint = QLabel("Hybrid uses the cloud model when online and the local "
                      "model offline. Fully online needs no local model; "
                      "fully offline ignores the cloud entirely.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8;")
        form.addRow("", hint)
        self._add_page("LLM", page)

    def _on_llm_provider_changed(self) -> None:
        """Enable hosted-provider fields only for non-Ollama providers."""
        provider = self.cmb_llm_provider.currentData() or "ollama"
        hosted = provider != "ollama"
        self.ed_provider_model.setEnabled(hosted)
        self.ed_provider_key.setEnabled(hosted)

    def _build_hotkeys(self) -> None:
        s = self.settings
        self.ed_ptt = QLineEdit("+".join(s.get("hotkeys.push_to_talk", ["ctrl", "win"])))
        self.ed_auto = QLineEdit("+".join(s.get("hotkeys.auto_stop", ["ctrl", "win", "space"])))
        self._add_tab("Hotkeys", [
            ("Push-to-talk", self.ed_ptt),
            ("Auto-stop", self.ed_auto),
        ])

    def _build_memory(self) -> None:
        s = self.settings
        self.chk_memory = QCheckBox()
        self.chk_memory.setChecked(bool(s.get("memory.enabled", True)))
        self.spin_retention = QSpinBox()
        self.spin_retention.setRange(7, 3650)
        self.spin_retention.setValue(int(s.get("memory.retention_days", 365)))
        self.chk_clear = QCheckBox("Clear all memories on Save")
        self._add_tab("Memory", [
            ("Enable memory", self.chk_memory),
            ("Retention (days)", self.spin_retention),
            ("", self.chk_clear),
        ])

    def _build_tools(self) -> None:
        s = self.settings
        self.chk_shell = QCheckBox()
        self.chk_shell.setChecked(bool(s.get("tools.shell_enabled", False)))
        self.chk_confirm = QCheckBox()
        self.chk_confirm.setChecked(bool(s.get("tools.confirm_destructive", True)))
        self._add_tab("Tools", [
            ("Enable shell tool (dangerous)", self.chk_shell),
            ("Confirm destructive actions", self.chk_confirm),
        ])

    def _build_dependencies(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("Dependencies are checked on every startup and can be "
                       "repaired from the Setup dialog.")
        label.setWordWrap(True)
        layout.addWidget(label)

        if self.cloud_auth_failed:
            box = QGroupBox("Ollama cloud sign-in required")
            v = QVBoxLayout(box)
            intro = QLabel("Your cloud model returned <b>401 Unauthorized</b>, so the "
                           "assistant is using the local model instead. To use cloud "
                           "models:")
            intro.setWordWrap(True)
            v.addWidget(intro)
            step1 = QLabel('1. In a terminal, run <code>ollama signin</code> — it opens '
                           'a browser where you log in with your Ollama account.')
            step1.setWordWrap(True)
            v.addWidget(step1)
            step2 = QLabel("2. Save these settings — the app will retry the cloud model.")
            step2.setWordWrap(True)
            v.addWidget(step2)
            layout.addWidget(box)
        elif self.settings.use_cloud_models:
            hint = QLabel("Cloud models are enabled. Sign in once with "
                          "<code>ollama signin</code> in a terminal to use them.")
            hint.setWordWrap(True)
            layout.addWidget(hint)

        self.lbl_status = QLabel("Run a check to see current status.")
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)
        self._add_page("Dependencies", page)

    # ------------------------------------------------------------------ save

    def _save(self) -> None:
        s = self.settings
        s.update({
            "general": {
                "start_with_windows": self.chk_start_with_windows.isChecked(),
                "show_floating_widget": self.chk_show_widget.isChecked(),
                "theme": self.cmb_theme.currentText(),
                "agent_name": self.cmb_agent_name.currentText().strip() or "Luna",
                "user_name": self.ed_user_name.text().strip(),
            },
            "character": {
                "preset": self.cmb_character.currentData() or "luna",
                "enabled": self.chk_char_enabled.isChecked(),
                "scale": round(self.spin_char_scale.value(), 2),
            },
            "voice_input": {
                "microphone": self.cmb_mic.currentText() if self.cmb_mic.currentText() != "Default" else None,
                "stt_engine": self.cmb_stt.currentText(),
                "whisper_model": self.cmb_whisper.currentText(),
                "language": self.cmb_lang.currentData(),
                "cpu_threads": self.spin_threads.value(),
                "vad_sensitivity": self.spin_sensitivity.value(),
                "silence_duration": self.spin_silence.value(),
                "minimum_speech_duration": self.spin_min_speech.value(),
                "maximum_recording_duration": self.spin_max_rec.value(),
            },
            "voice_output": {
                "speaker": self.cmb_speaker.currentText() if self.cmb_speaker.currentText() != "Default" else None,
                "tts_engine": self.cmb_tts.currentText(),
                "kokoro_voice": self.cmb_kokoro_voice.currentText(),
                "piper_voice": self.cmb_piper_voice.currentText(),
                "speed": self.spin_speed.value(),
                "volume": self.spin_volume.value(),
            },
            "llm": {
                "host": self.cmb_host.currentText(),
                "connectivity": self.cmb_connectivity.currentData() or "hybrid",
                "provider": self.cmb_llm_provider.currentData() or "ollama",
                "provider_model": self.ed_provider_model.text().strip(),
                "local_model": self.cmb_local.currentText(),
                "cloud_model": self.cmb_cloud.currentText(),
                "use_cloud_models": (self.cmb_connectivity.currentData() or "hybrid") != "offline",
                "context_size": self.spin_ctx.value(),
                "temperature": self.spin_temp.value(),
                "tools_enabled": self.chk_tools.isChecked(),
            },
            "hotkeys": {
                "push_to_talk": self.ed_ptt.text().split("+"),
                "auto_stop": self.ed_auto.text().split("+"),
            },
            "memory": {
                "enabled": self.chk_memory.isChecked(),
                "retention_days": self.spin_retention.value(),
            },
            "tools": {
                "shell_enabled": self.chk_shell.isChecked(),
                "confirm_destructive": self.chk_confirm.isChecked(),
            },
        })
        if self.chk_clear.isChecked() and self.memory is not None:
            deleted = self.memory.clear_memories()
            self.memory.log_event("info", "settings", f"Cleared {deleted} memories")
        provider = self.cmb_llm_provider.currentData() or "ollama"
        if provider != "ollama":
            s.set_api_key(provider, self.ed_provider_key.text())
        self.accept()
