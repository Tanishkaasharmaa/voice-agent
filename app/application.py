"""VoiceAgentApplication — the QApplication that wires everything together.

Startup order:
1. logging + directories + settings + SQLite memory
2. Ollama client + dependency checks (every startup)
3. Setup check dialog + per-dependency Yes/No installs
4. first-run wizard (first launch only)
5. tray + floating widget + hotkeys + pipeline + resource monitor
"""

from __future__ import annotations

import sys
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.events import (
    EventBus, TOPIC_AGENT_STATUS, TOPIC_ERROR, TOPIC_STATE, TOPIC_TRANSCRIPT,
    TOPIC_TTS_AMPLITUDE, TOPIC_TTS_STARTED, TOPIC_TTS_STOPPED,
)
from app.resource_manager import AppResourceManager
from app.state_machine import (
    EXECUTING_TOOL, IDLE, LISTENING, SPEAKING, StateMachine, THINKING,
    TRANSCRIBING, ERROR,
)
from audio.manager import AudioManager
from audio.microphone import Microphone
from audio.speaker import Speaker
from hotkeys.manager import HotkeyManager
from memory.database import Database
from memory.manager import MemoryManager
from ollama.agent import AgentRuntime
from ollama.client import OllamaClient
from ollama.model_router import ModelRouter
from ollama.resource_manager import LLMResourceManager
from response.processor import ResponseProcessor
from setup.dependency_manager import DependencyManager
from setup.dependency_checks import VOICE_INPUT, missing_statuses, DependencyStatus
from tools.permissions import PermissionLevel
from tools.registry import ToolContext, ToolRegistry
from ui.floating_widget import FloatingWidget
from ui.history import HistoryDialog
from ui.settings import SettingsDialog
from ui.setup_wizard import FirstRunWizard, ProgressDialog, SetupCheckDialog, YesNoDialog
from ui.styles import APP_QSS
from ui.system_tray import SystemTray
from utils.logging import get_logger, setup_logging
from utils.platform import ensure_directories, is_windows
from utils.system import ResourceMonitor, has_internet

log = get_logger("application")


class _UiBridge(QObject):
    """Marshals cross-thread UI updates (state changes, warnings) onto the
    Qt thread via queued signals."""

    stateChanged = Signal(str)
    warning = Signal(str, str)
    amplitudeChanged = Signal(float)
    internetOnline = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.stateChanged.connect(app._on_state_ui, Qt.ConnectionType.QueuedConnection)
        self.warning.connect(app._on_warning_ui, Qt.ConnectionType.QueuedConnection)
        self.amplitudeChanged.connect(app._on_amplitude_ui,
                                      Qt.ConnectionType.QueuedConnection)
        self.internetOnline.connect(app._on_online_ui,
                                    Qt.ConnectionType.QueuedConnection)


class _FailBridge(QObject):
    """Marshal worker-thread failures onto the UI thread (QTimer is not
    safe to touch from a non-Qt thread)."""

    failed = Signal(str, str)

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.failed.connect(app._on_fail_ui, Qt.ConnectionType.QueuedConnection)


class _ConfirmationBridge(QObject):
    """Marshals tool confirmations from worker threads to the UI thread."""

    request = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._approved = False
        self.request.connect(self._handle, Qt.ConnectionType.BlockingQueuedConnection)

    def confirm(self, prompt: str) -> bool:
        self._approved = False
        self.request.emit(prompt)
        return self._approved

    @Slot(str)
    def _handle(self, prompt: str) -> None:
        dlg = YesNoDialog("Tool confirmation", prompt, "", optional=False)
        dlg.exec()
        self._approved = dlg.result_yes


class VoiceAgentApplication(QApplication):
    def __init__(self, argv, background: bool = False, show_logs: bool = False) -> None:
        super().__init__(argv)
        self.setApplicationName("LocalVoiceAgent")
        # The floating widget is a Qt.Tool window (not a "primary" window), so
        # without this the app quits when a dialog like Settings — the last
        # primary window — is closed, taking the floating widget down with it.
        self.setQuitOnLastWindowClosed(False)
        self.background = background
        self.show_logs = show_logs
        self._shutting_down = False

        self.dirs = ensure_directories()
        setup_logging(self.dirs["logs"], console=show_logs)
        self.setStyleSheet(APP_QSS)

        from app.config import Settings

        self.settings = Settings()
        saved_mode = str(self.settings.get("general.assistant_mode", "general")).lower()
        self.assistant_mode = saved_mode if saved_mode in {"general", "university"} else "general"
        self.bus = EventBus()
        self.state = StateMachine(self.bus)
        self.state.add_listener(self._on_state_changed)

        self.db = Database(self.dirs["data"] / "localvoiceagent.db")
        self.memory = MemoryManager(self.db)

        # Cloud vs local models: decided by llm.connectivity ("online",
        # "offline", "hybrid") and llm.provider ("ollama" or an
        # OpenAI-compatible cloud provider). See _build_llm_stack /
        # _refresh_cloud_models.
        self.cloud_model: str | None = None
        self.effective_agent_model = self.settings.local_model
        self.effective_reasoning_model = self.settings.local_model
        self._refresh_cloud_models()
        self._build_llm_stack()
        # Startup detection can lose the race against a slow Wi-Fi
        # association — keep probing in the background and switch to the
        # cloud model live once connectivity appears.
        self._net_check_stop = threading.Event()
        self._schedule_online_recheck()

        self.app_resources = AppResourceManager(self.llm_resources, self.bus,
                                                self.settings.get("performance.resource_poll_interval_s", 5))

        self.processor = ResponseProcessor()

        # Bridges must exist before tools are built — ToolRegistry binds
        # tool_context.confirm to confirm_bridge.confirm.
        self.confirm_bridge = _ConfirmationBridge(self)
        self.fail_bridge = _FailBridge(self)
        self.ui_bridge = _UiBridge(self)
        self.bus.subscribe(TOPIC_AGENT_STATUS, self._on_agent_status_topic)
        # Amplitude events come from the speaker worker thread — marshal to
        # the UI thread for the character's lip sync.
        self.bus.subscribe(TOPIC_TTS_AMPLITUDE, self._on_tts_amplitude_topic)

        self._build_tools()
        self._build_agent()

        # University mode is an optional, local FAQ-only layer. It does not
        # replace AgentRuntime and does not send university questions to the
        # LLM when the FAQ cannot support an answer.
        from university.assistant import UniversityAssistant
        self.university_assistant = UniversityAssistant(processor=self.processor)

        # Engines (lazy — loaded on first use).
        self.stt = None
        self.vad = None
        self.tts_manager = None
        self.audio_manager: AudioManager | None = None

        # UI.
        self.tray = None
        self.widget = None
        self.character = None
        self._voice_enabled = True
        self._pipeline_thread: threading.Thread | None = None
        # Set when a cloud model request comes back 401 Unauthorized; shown
        # as an Ollama sign-in hint in Settings until the user saves them.
        self.cloud_auth_failed = False

        self.dependency_manager = DependencyManager(
            settings=self.settings, client=self.client, memory=self.memory,
            models_dir=self.dirs["models"], cloud_model=self.cloud_model,
        )
        self.dependency_manager.prompt_yes_no = self._prompt_dependency
        self.dependency_manager.show_progress = self._run_progress

        # Hotkeys.
        self.hotkeys = HotkeyManager(
            push_to_talk=tuple(self.settings.get("hotkeys.push_to_talk", ["ctrl", "win"])),
            auto_stop=tuple(self.settings.get("hotkeys.auto_stop", ["ctrl", "win", "space"])),
        )
        self.hotkeys.on_ptt_press = self._on_ptt_press
        self.hotkeys.on_ptt_release = self._on_ptt_release
        self.hotkeys.on_auto_stop = self._on_auto_stop

        # Wire UI after engines are ready.
        QTimer.singleShot(0, self._startup)

    # ------------------------------------------------------------- pipeline

    def _build_tools(self) -> None:
        self.tool_context = ToolContext()
        self.tool_context.settings = self.settings
        self.tool_context.memory = self.memory
        self.tool_context.confirm = self.confirm_bridge.confirm
        self.tool_context.log = lambda m: log.info("%s", m)

        self.registry = ToolRegistry(self.tool_context)
        from tools import browser, clipboard, notes, shell, system, windows

        self.registry.register_many(windows.build_tools())
        self.registry.register_many(clipboard.build_tools())
        self.registry.register_many(browser.build_tools())
        self.registry.register_many(notes.build_tools())
        self.registry.register_many(system.build_tools())
        self.registry.register_many(shell.build_tools(enabled=bool(self.settings.get("tools.shell_enabled", False))))

    def _build_agent(self) -> None:
        self.agent = AgentRuntime(
            client=self.client,
            router=self.router,
            resources=self.llm_resources,
            registry=self.registry,
            processor=self.processor,
            memory=self.memory,
            bus=self.bus,
            settings=self.settings,
        )

    def _build_audio(self) -> None:
        """Lazily create audio engines (dependency-aware)."""
        if self.audio_manager is not None:
            return
        from stt.whisper_engine import WhisperEngine
        from vad.silero import SileroVAD, default_model_path

        mic = Microphone(sample_rate=int(self.settings.get("voice_input.sample_rate", 16000)),
                         device=self.settings.get("voice_input.microphone"))
        speaker = Speaker(device=self.settings.get("voice_output.speaker"))

        self.stt = WhisperEngine(
            model=self.settings.whisper_model,
            language=self.settings.get("voice_input.language", "en"),
            cpu_threads=int(self.settings.get("voice_input.cpu_threads", 0) or 0),
            models_dir=self.dirs["models"],
        )
        self.vad = SileroVAD(default_model_path(self.dirs["models"]))

        from tts.base import TTSManager
        from tts.kokoro import KokoroEngine
        from tts.piper import PiperEngine

        if self.settings.tts_engine == "piper":
            engine = PiperEngine()
        else:
            engine = KokoroEngine(
                voice=self.settings.get("voice_output.kokoro_voice", "af_heart"),
                model_path=self.dirs["models"] / "kokoro-v1.0.onnx",
                voices_path=self.dirs["models"] / "voices-v1.0.bin",
            )
        self.tts_manager = TTSManager(engine, speaker)
        self.tts_manager.on_state = lambda playing: (
            self.bus.publish(TOPIC_TTS_STARTED if playing else TOPIC_TTS_STOPPED)
        )
        self.tts_manager.on_amplitude = lambda level: self.bus.publish(
            TOPIC_TTS_AMPLITUDE, {"level": float(level)})

        self.audio_manager = AudioManager(
            microphone=mic,
            vad=self.vad,
            settings=self.settings,
            debug_recording=bool(self.settings.get("debug.save_recordings", False)),
            recordings_dir=self.dirs["recordings"],
        )
        self.audio_manager.on_utterance = self._on_utterance
        self.audio_manager.on_speech_state = self._on_speech_state
        self.audio_manager.on_error = lambda msg: self._fail("audio", msg)

    # --------------------------------------------------------------- models

    def _build_llm_stack(self) -> None:
        """Create the chat client, router and resource manager for the
        configured provider (Ollama or an OpenAI-compatible cloud)."""
        from ollama.providers import build_client

        self.client = build_client(self.settings.provider, self.settings)
        self.router = ModelRouter(
            self.effective_agent_model, self.effective_reasoning_model, "auto",
        )
        self.llm_resources = LLMResourceManager(
            self.client, self.effective_agent_model, self.effective_reasoning_model,
            self.settings.get("llm.keep_alive", "10m"), bus=self.bus,
        )

    def _refresh_cloud_models(self) -> None:
        """Decide the effective agent/reasoning models.

        * provider != ollama: the provider's model serves everything.
        * offline: local Ollama model only.
        * hybrid: cloud when online (current behavior).
        * online: cloud/provider only — no local fallback requirement.
        """
        from ollama.providers import PROVIDERS

        mode = self.settings.connectivity_mode
        if self.settings.provider != "ollama":
            model = self.settings.provider_model or (
                PROVIDERS.get(self.settings.provider, {}).get("models") or [""])[0]
            self.cloud_model = None
            self.effective_agent_model = model
            self.effective_reasoning_model = model
            self.internet_available = True  # provider is hosted by definition
            log.info("Provider %s active — using model %s",
                     self.settings.provider, model)
            return

        self.internet_available = False
        if mode != "offline":
            # online always probes; hybrid probes only when cloud models
            # are enabled in settings (matches the previous behavior).
            if mode == "online" or self.settings.use_cloud_models:
                self.internet_available = has_internet()
            if self.internet_available:
                cloud = self.settings.cloud_model
                self.cloud_model = cloud
                self.effective_agent_model = cloud
                self.effective_reasoning_model = cloud
                log.info("Internet detected — using Ollama cloud model %s", cloud)
                return
        if not self.internet_available:
            self.cloud_model = None
            self.effective_agent_model = self.settings.local_model
            self.effective_reasoning_model = self.settings.local_model

    def _schedule_online_recheck(self) -> None:
        """When cloud models are enabled but startup detection found no
        internet, keep probing in a daemon thread (every 2 s for ~60 s).
        Fires ``internetOnline`` on the UI thread once connectivity shows
        up so the cloud model is applied without a restart."""
        if not getattr(self.settings, "use_cloud_models", False):
            return

        def worker() -> None:
            deadline = time.time() + 60.0
            while time.time() < deadline:
                if self._net_check_stop.wait(2.0):
                    return
                if has_internet(timeout=2.0):
                    self.ui_bridge.internetOnline.emit()
                    return

        threading.Thread(target=worker, name="net-recheck", daemon=True).start()

    def _on_online_ui(self) -> None:
        # Runs on the Qt thread. Re-decide cloud vs local and apply it live,
        # mirroring what saving Settings does.
        if self._shutting_down or not self.settings.use_cloud_models:
            return
        was_cloud = bool(self.cloud_model)
        self._refresh_cloud_models()
        if not self.internet_available:
            return  # probe raced back offline — the worker keeps trying
        if was_cloud:
            return
        self.llm_resources.agent_model = self.effective_agent_model
        self.llm_resources.reasoning_model = self.effective_reasoning_model
        self.dependency_manager.cloud_model = self.cloud_model
        self.agent.reset_cloud_fallback()
        log.info("Internet detected after startup — switched to cloud model %s",
                 self.cloud_model)
        self._notify("Cloud models enabled",
                     f"Internet detected — using {self.cloud_model}.")

    # --------------------------------------------------------------- startup

    def _startup(self) -> None:
        try:
            statuses = self.dependency_manager.check_all()
            self._run_setup(statuses)
            if self._shutting_down:
                return
            # Wizard-driven auto-approval applies to the first resolve only.
            self.dependency_manager.auto_approve = False
            self._start_services()
            self._finish_startup()
        except Exception:  # noqa: BLE001
            log.exception("startup failed")
            self.quit()

    def _voice_input_blocked(self, missing: list[DependencyStatus]) -> bool:
        """True when the listening/transcription pipeline is unusable (no
        microphone, no speech-to-text). Anything else — TTS, LLM models,
        optional extras — degrades gracefully and must not disable voice."""
        return any(
            s.name == "Audio input" or s.category == VOICE_INPUT
            for s in missing
        )

    def _run_setup(self, statuses: list[DependencyStatus]) -> None:
        # First run: the simplified wizard picks the connectivity mode and
        # models BEFORE anything else, so dependency checks and installs
        # match what the user actually chose.
        if not self.settings.get("setup.first_run_done", False):
            self._run_first_run_wizard()
            statuses = self.dependency_manager.check_all()

        dialog = SetupCheckDialog(statuses)
        missing = missing_statuses(statuses)
        if missing:
            choice = dialog.exec()
            if choice == SetupCheckDialog.EXIT_CODE:
                # Exit pressed — shut the app down instead of starting it
                # half-configured. Nothing has started yet at this point.
                self._shutting_down = True
                self.quit()
                return
            if choice == 0:
                # Skipped by user — start anyway with voice enabled. Only a
                # broken mic/STT pipeline disables voice; everything else
                # (TTS, models, optional extras) degrades gracefully.
                if self._voice_input_blocked(missing):
                    self._voice_enabled = False
                    self._notify("Setup skipped",
                                 "Microphone or speech recognition is unavailable — voice input is off.")
                else:
                    self._notify("Setup skipped", "Starting with available features.")
                return
        resolved = self.dependency_manager.resolve(statuses)
        if not resolved and self._voice_input_blocked(missing):
            self._voice_enabled = False

    def _run_first_run_wizard(self) -> None:
        services = type("WizardServices", (), {"test_mic": self._test_mic})()
        wizard = FirstRunWizard(services)
        wizard.exec()
        # Persist everything the wizard chose so every later session
        # remembers it (all of it changeable in Settings).
        s = self.settings
        s.set("general.agent_name", wizard.agent_name)
        s.set("general.user_name", wizard.user_name)
        s.set("llm.connectivity", wizard.connectivity)
        s.set("llm.provider", wizard.provider)
        if wizard.provider != "ollama":
            s.set("llm.provider_model", wizard.provider_model)
            s.set_api_key(wizard.provider, wizard.api_key)
        elif wizard.cloud_model:
            s.set("llm.cloud_model", wizard.cloud_model)
        s.set("llm.local_model", wizard.local_model)
        s.set("llm.use_cloud_models", wizard.connectivity != "offline")
        s.set("setup.first_run_done", True)
        s.save()
        # Re-decide cloud vs local with the chosen mode and rebuild the
        # LLM stack for the chosen provider.
        self._refresh_cloud_models()
        old_resources = getattr(self, "llm_resources", None)
        self._build_llm_stack()
        if old_resources is not None:
            old_resources.shutdown()
        self.agent.client = self.client
        self.agent.resources = self.llm_resources
        self.agent.router = self.router
        self.dependency_manager.cloud_model = self.cloud_model
        # Offline/hybrid users explicitly opted into the local stack —
        # install missing pieces without re-asking Yes/No per component.
        self.dependency_manager.auto_approve = wizard.connectivity in ("offline", "hybrid")

    def _start_services(self) -> None:
        self._build_audio()
        self.tray = SystemTray(self._app_icon())
        self.tray.show()
        self._connect_tray()

        if self.settings.get("general.show_floating_widget", True):
            self.widget = FloatingWidget(self.settings)
            self.widget.settings_clicked.connect(self._open_settings)
            if not self.background:
                self.widget.show()
            if self.settings.get("character.enabled", True):
                from ui.character_widget import CharacterWidget

                self.character = CharacterWidget(self.settings, island=self.widget)
                self.widget.position_changed.connect(self.character.follow_island)
                if not self.background:
                    self.character.show()
            self.widget.close_clicked.connect(self._exit)
            self.widget.auto_listen_requested.connect(self._on_auto_listen)

        self.hotkeys.start()
        self.app_resources.start()
        self.app_resources.on_warning = self._on_warning_from_worker

        # Default TTS state so the widget reflects the initial condition.
        self.widget.set_state(self.state.state) if self.widget else None
        self.bus.publish(TOPIC_STATE, self.state.state)

    def _finish_startup(self) -> None:
        log.info("LocalVoiceAgent started (voice %s)", "on" if self._voice_enabled else "off")

    def _app_icon(self) -> QIcon:
        # A simple generated icon (no external asset required).
        from PySide6.QtGui import QColor, QPixmap

        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)
        from PySide6.QtGui import QPainter

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#3b82f6"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(24, 24, 16, 16)
        painter.end()
        return QIcon(pixmap)

    # ------------------------------------------------------------------ tray

    def _connect_tray(self) -> None:
        t = self.tray
        t.act_show_hide.triggered.connect(self._toggle_widget)
        t.act_settings.triggered.connect(self._open_settings)
        t.act_history.triggered.connect(self._open_history)
        t.act_general_mode.triggered.connect(
            lambda _checked=False: self._set_assistant_mode("general"))
        t.act_university_mode.triggered.connect(
            lambda _checked=False: self._set_assistant_mode("university"))
        t.set_assistant_mode(self.assistant_mode)
        t.act_enable_voice.triggered.connect(self._enable_voice)
        t.act_disable_voice.triggered.connect(self._disable_voice)
        t.act_reload_models.triggered.connect(self._reload_models)
        t.act_restart.triggered.connect(self._restart_agent)
        t.act_exit.triggered.connect(self._exit)
        t.tray.activated_callback = self._toggle_widget
        t.tray.showMessage("LocalVoiceAgent",
                           "Ready. Hold CTRL+WINDOWS and speak, or press "
                           "CTRL+WINDOWS+SPACE for auto-stop.",
                           QIcon(), 4000)

    def _toggle_widget(self) -> None:
        if self.widget:
            visible = not self.widget.isVisible()
            self.widget.setVisible(visible)
            if self.character and visible:
                self.character.follow_island(self.widget.pos())
            if self.character:
                self.character.setVisible(visible)

    def _on_auto_listen(self) -> None:
        """Double-click on the island: start continuous auto-stop listening
        (voice input must be enabled; any speech in progress is cut off)."""
        if not self._voice_enabled:
            return
        self._interrupt_speech()
        self._start_listening("auto_stop")

    def _open_settings(self) -> None:
        # Non-modal on purpose: a modal dialog would block input to every
        # other window, so the floating widget could not be dragged while
        # Settings is open. Live changes are applied on the accepted signal.
        if getattr(self, "_settings_dialog", None) is not None:
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        dialog = SettingsDialog(self.settings, self.dependency_manager, None,
                                cloud_auth_failed=self.cloud_auth_failed)
        self._settings_dialog = dialog
        dialog.accepted.connect(self._apply_settings_changes)
        dialog.finished.connect(self._on_settings_closed)
        dialog.show()

    def _on_settings_closed(self, _result: int) -> None:
        self._settings_dialog = None

    def _on_agent_status_topic(self, _topic: str, payload) -> None:
        # Published from the pipeline worker thread — marshal to the UI
        # thread before touching the tray.
        if isinstance(payload, dict) and payload.get("status") == "cloud_unavailable":
            self.cloud_auth_failed = True
            self.ui_bridge.warning.emit("Cloud model unavailable", payload.get("detail", ""))

    def _apply_settings_changes(self) -> None:
        """Apply live changes after the user saves Settings."""
        self._refresh_cloud_models()
        self.dependency_manager.cloud_model = self.cloud_model
        # The user may have signed in to Ollama — let the cloud model be
        # tried again and clear the sign-in hint.
        self.cloud_auth_failed = False
        self.agent.reset_cloud_fallback()
        # The provider/client/model trio may all have changed — rebuild.
        old_resources = self.llm_resources
        self._build_llm_stack()
        old_resources.shutdown()
        self.agent.client = self.client
        self.agent.resources = self.llm_resources
        self.agent.router = self.router
        self._apply_character_settings()
        self.hotkeys.reset_keys()
        # SettingsDialog persists the selected mode; apply it immediately and
        # keep the tray's exclusive actions synchronized.
        selected_mode = str(
            self.settings.get("general.assistant_mode", "general")
        ).lower()
        if selected_mode not in {"general", "university"}:
            selected_mode = "general"
        self.assistant_mode = selected_mode
        self.tray.set_assistant_mode(selected_mode)

    def _apply_character_settings(self) -> None:
        """Apply Character page changes live: preset, size, visibility."""
        if not self.settings.get("general.show_floating_widget", True):
            return  # no island -> no character anchor; needs restart anyway
        enabled = bool(self.settings.get("character.enabled", True))
        if self.character is None and enabled:
            from ui.character_widget import CharacterWidget

            self.character = CharacterWidget(self.settings, island=self.widget)
            self.widget.position_changed.connect(self.character.follow_island)
            self.character.show()
            return
        if self.character is None:
            return
        self.character.setVisible(enabled)
        if not enabled:
            return
        preset = str(self.settings.get("character.preset", "luna"))
        self.character.set_preset(preset)
        self.character.apply_scale()
        if self.widget is not None:
            self.character.follow_island(self.widget.pos())

    def _open_history(self) -> None:
        HistoryDialog(self.memory).exec()

    def _enable_voice(self) -> None:
        self._voice_enabled = True
        self.tray.act_enable_voice.setChecked(True)

    def _disable_voice(self) -> None:
        self._voice_enabled = False
        self.tray.act_enable_voice.setChecked(False)
        self._stop_current_flow()

    def _reload_models(self) -> None:
        self.llm_resources.shutdown()
        if self.stt is not None:
            self.stt.unload()
        if self.tts_manager is not None:
            self.tts_manager.shutdown()
        self.tray.show_message("Models", "Models unloaded. They will reload on next use.")

    def _restart_agent(self) -> None:
        self._stop_current_flow()
        self.llm_resources.shutdown()
        if self.stt is not None:
            self.stt.unload()
        self.state.reset()
        self.tray.show_message("Agent", "Agent restarted.")

    def _exit(self) -> None:
        self._shutting_down = True
        self._net_check_stop.set()
        self.hotkeys.stop()
        self.app_resources.stop()
        self._stop_current_flow()
        if self.tts_manager is not None:
            self.tts_manager.shutdown()
        self.llm_resources.shutdown()
        self.db.close_all()
        self.quit()

    def _notify(self, title: str, message: str) -> None:
        """Show a tray balloon when the tray exists; otherwise log. The tray
        is created after the setup phase, so setup/wizard code paths (e.g.
        the STT/TTS tests) must not assume it exists yet."""
        if self.tray is not None:
            self.tray.show_message(title, message)
        else:
            log.info("%s: %s", title, message)

    def _on_warning_from_worker(self, title: str, message: str) -> None:
        self.ui_bridge.warning.emit(title, message)

    def _on_warning_ui(self, title: str, message: str) -> None:
        if self.tray:
            self.tray.show_message(title, message)

    # ------------------------------------------------------------- hotkeys

    def _on_ptt_press(self) -> None:
        if not self._voice_enabled:
            return
        self._interrupt_speech()
        self._start_listening("push_to_talk")

    def _on_ptt_release(self) -> None:
        if self.audio_manager and self.state.is_in(LISTENING):
            # In auto-stop mode the VAD owns the ending — releasing the
            # keys must not cut the user off mid-sentence.
            if self.audio_manager.mode != "auto_stop":
                self.audio_manager.stop_listening()

    def _on_auto_stop(self) -> None:
        if not self._voice_enabled:
            return
        self._interrupt_speech()
        self._start_listening("auto_stop")

    def _interrupt_speech(self) -> None:
        if self.tts_manager is not None and self.tts_manager.is_speaking():
            self.tts_manager.stop()
            self.bus.publish(TOPIC_TTS_STOPPED)

    def _start_listening(self, mode: str) -> None:
        if self.audio_manager is None:
            self._build_audio()
        try:
            if self.state.is_in(LISTENING):
                # CTRL+WIN+SPACE fires the PTT press (CTRL+WIN) first and
                # the auto-stop trigger (SPACE) second — upgrade the running
                # session instead of erroring on a duplicate transition.
                self.audio_manager.ensure_mode(mode)
                return
            self._set_state(LISTENING)
            self.audio_manager.start_listening(mode)
        except Exception:  # noqa: BLE001
            log.exception("failed to start listening")
            self._fail("listening", "Could not start the microphone")

    def _stop_current_flow(self) -> None:
        if self.audio_manager is not None:
            self.audio_manager.cancel()
        if self.tts_manager is not None:
            self.tts_manager.stop()
        self.state.reset()

    # ------------------------------------------------------------- pipeline

    def _on_speech_state(self, speaking: bool) -> None:
        # Only the listening indicator changes; state machine stays LISTENING.
        pass

    def _on_utterance(self, audio, sample_rate: int) -> None:
        if self._voice_enabled is False:
            return
        self._pipeline_thread = threading.Thread(
            target=self._process_utterance, args=(audio, sample_rate),
            name="pipeline", daemon=True,
        )
        self._pipeline_thread.start()

    def _process_utterance(self, audio, sample_rate: int) -> None:
        try:
            # Continuous mode (auto-stop trigger): when the turn finishes the
            # agent goes back to listening instead of idling. Push-to-talk
            # sessions always end in IDLE.
            continuous = bool(
                self.audio_manager is not None
                and self.audio_manager.mode == "auto_stop")
            self._set_state(TRANSCRIBING)
            if self.stt is None:
                raise RuntimeError("STT not available")
            self.stt.load()
            transcript = self.stt.transcribe(audio, sample_rate)
            self.bus.publish(TOPIC_TRANSCRIPT, {"final": True, "text": transcript})
            log.info("[USER] %s", transcript)
            if not transcript.strip():
                self._finish_turn(continuous)
                return
            if self.state.is_in(LISTENING):
                return
            self._set_state(THINKING)
            self.memory.add_message("user", transcript)

            def on_sentence(sentence: str) -> None:
                if self.tts_manager is not None and self.state.is_in(THINKING, SPEAKING):
                    self._set_state(SPEAKING)
                    log.info("[KOKORO SPEAKING] %s", sentence)
                    self.tts_manager.speak(sentence)

            # Explicit spoken commands switch between the existing general
            # agent and the university Wi-Fi help desk. The selected mode is
            # persisted in Settings and restored on the next launch.
            from university.assistant import detect_mode_command
            requested_mode = detect_mode_command(transcript)
            if requested_mode is not None:
                self._set_assistant_mode(requested_mode)
                label = "University Help Desk" if requested_mode == "university" else "General Assistant"
                final = f"Switched to {label} mode."
                for sentence in self.processor.split_sentences(final):
                    on_sentence(sentence)
            elif self.assistant_mode == "university":
                # FAQ-only response path: unsupported questions are handled
                # safely by UniversityAssistant rather than guessed by Ollama.
                final = self.university_assistant.answer(transcript)
                for sentence in self.processor.split_sentences(final):
                    on_sentence(sentence)
            else:
                final = self.agent.run(transcript, on_sentence=on_sentence)
            self.memory.add_message("assistant", final)
            # Do NOT drop to IDLE yet: sentences may still be queued or
            # playing (first Kokoro load alone can take seconds). Wait until
            # the TTS pipeline is fully drained. Deliberately NOT keyed on
            # state.is_in(SPEAKING): an error/interrupt elsewhere must not
            # drop the state to IDLE while audio is still pending —
            # interruption itself drains is_busy() via TTSManager.stop().
            if self.tts_manager is not None:
                deadline = time.time() + 300.0
                waited = False
                while self.tts_manager.is_busy():
                    waited = True
                    if time.time() >= deadline:
                        log.warning("[SPEAK] drain deadline hit with %d "
                                    "sentence(s) pending",
                                    self.tts_manager.pending_count())
                        break
                    time.sleep(0.05)
                if waited:
                    if continuous:
                        log.info("[SPEAK] all sentences played — "
                                 "listening again")
                    else:
                        log.info("[SPEAK] all sentences played — going idle")
            # Flush any buffered tail already handled by agent.run.
            self._finish_turn(continuous)
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline failed")
            self._fail("pipeline", str(exc))

    def _set_assistant_mode(self, mode: str) -> None:
        """Switch assistant persona and persist the choice."""
        if mode not in {"general", "university"}:
            raise ValueError(f"Unsupported assistant mode: {mode}")
        self.assistant_mode = mode
        try:
            self.settings.set("general.assistant_mode", mode)
            self.settings.save()
        except Exception:  # noqa: BLE001
            # Keep the current session usable even if settings persistence fails.
            log.exception("Could not persist assistant mode %s", mode)
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.set_assistant_mode(mode)
        log.info("Assistant mode changed to %s", mode)

    def _finish_turn(self, was_continuous: bool) -> None:
        """Settle after a processed utterance. Continuous (auto-stop)
        sessions resume listening immediately; push-to-talk ends in IDLE."""
        if self.state.is_in(LISTENING):
            # The user already triggered a new session while this turn was
            # finishing — don't stomp it back through IDLE.
            return
        self._set_state(IDLE)
        if (was_continuous and self._voice_enabled
                and not self._shutting_down):
            self._start_listening("auto_stop")

    def _set_state(self, state: str) -> None:
        if self.state.state == state:
            # Idempotent re-entry (e.g. every streamed sentence re-requests
            # SPEAKING): falling through would raise InvalidTransition and
            # the reset() fallback would flash the UI to IDLE mid-speech.
            return
        try:
            self.state.transition(state)
        except Exception:
            # Transitions from an unexpected state should still settle safely.
            try:
                self.state.reset()
                self.state.transition(state)
            except Exception:
                pass

    def _on_state_changed(self, _old: str, new: str) -> None:
        # Called on the publishing thread (worker) — marshal to the UI thread.
        self.ui_bridge.stateChanged.emit(new)

    def _on_state_ui(self, new: str) -> None:
        if self.widget:
            self.widget.set_state(new)
        if self.character:
            self.character.set_state(new)
        if new == SPEAKING:
            self.bus.publish(TOPIC_TTS_STARTED)

    def _on_tts_amplitude_topic(self, _topic: str, payload) -> None:
        # Speaker worker thread — marshal to the UI thread.
        if isinstance(payload, dict) and self.character is not None:
            try:
                level = float(payload.get("level", 0.0))
            except (TypeError, ValueError):
                return
            self.ui_bridge.amplitudeChanged.emit(level)

    def _on_amplitude_ui(self, level: float) -> None:
        if self.character:
            self.character.set_mouth_level(level)

    # -------------------------------------------------------------- testing

    def _test_mic(self) -> None:
        self._start_listening("auto_stop")
        QTimer.singleShot(3000, self._stop_test_mic)

    def _stop_test_mic(self) -> None:
        if self.audio_manager:
            self.audio_manager.stop_listening()

    def _test_stt(self) -> None:
        QTimer.singleShot(2000, self._run_stt_test)

    def _run_stt_test(self) -> None:
        try:
            if self.stt is None:
                self._build_audio()
            self.stt.load()
            self._notify("STT", "Whisper model loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            self._notify("STT", f"STT failed: {exc}")

    def _test_tts(self) -> None:
        try:
            if self.tts_manager is None:
                self._build_audio()
            self.tts_manager.speak("Hello! LocalVoiceAgent is ready.")
            self._notify("TTS", "Speaking test sentence.")
        except Exception as exc:  # noqa: BLE001
            self._notify("TTS", f"TTS failed: {exc}")

    # ------------------------------------------------------------- dialogs

    def _prompt_dependency(self, name: str, purpose: str, detail: str, optional: bool) -> bool:
        dlg = YesNoDialog(name, purpose, detail, optional)
        dlg.exec()
        return dlg.result_yes

    def _run_progress(self, title: str, worker) -> None:
        dialog = ProgressDialog(title, worker)
        dialog.start()

    def _fail(self, component: str, message: str) -> None:
        """Called from any thread; recovery is marshalled to the UI thread."""
        self.bus.publish(TOPIC_ERROR, {"component": component, "message": message})
        log.error("%s: %s", component, message)
        self.fail_bridge.failed.emit(component, message)

    def _on_fail_ui(self, _component: str, _message: str) -> None:
        try:
            self.state.transition(ERROR)
        except Exception:
            pass
        QTimer.singleShot(2500, self._recover_from_error)

    def _recover_from_error(self) -> None:
        if self.state.state == ERROR:
            self.state.reset()
