"""Setup dialogs: the every-startup Setup Check and the first-run wizard."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QProgressBar, QPushButton, QRadioButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from app.config import AGENT_NAME_SUGGESTIONS
from setup.dependency_checks import DependencyStatus


# ---------------------------------------------------------------- check UI


class SetupCheckDialog(QDialog):
    """Shown at every startup: lists component statuses with check marks.

    Result codes: ``Accepted`` (1) = Install Missing, ``Rejected`` (0) =
    Skip, ``EXIT_CODE`` (2) = Exit (the caller must shut the app down).
    """

    EXIT_CODE = 2

    def __init__(self, statuses: list[DependencyStatus], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LocalVoiceAgent — Setup Check")
        self.resize(520, 520)

        layout = QVBoxLayout(self)
        title = QLabel("LocalVoiceAgent Setup Check")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Checking required components..."))

        self.list = QListWidget()
        layout.addWidget(self.list)
        self._populate(statuses)

        missing = [s for s in statuses if not s.present]
        self.summary = QLabel(
            f"Missing components: {len(missing)}"
            if missing else "All components ready."
        )
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.btn_install = QPushButton("Install Missing")
        self.btn_install.setEnabled(bool(missing))
        self.btn_skip = QPushButton("Skip")
        self.btn_exit = QPushButton("Exit")
        buttons.addWidget(self.btn_install)
        buttons.addWidget(self.btn_skip)
        buttons.addWidget(self.btn_exit)
        layout.addLayout(buttons)

        self.btn_install.clicked.connect(self.accept)
        self.btn_skip.clicked.connect(self.reject)
        self.btn_exit.clicked.connect(lambda: self.done(self.EXIT_CODE))

    def _populate(self, statuses: list[DependencyStatus]) -> None:
        self.list.clear()
        for status in statuses:
            icon = "\u2713" if status.present else "\u2717"
            text = f"{icon}  {status.name}"
            if status.version:
                text += f"  ({status.version})"
            item = QListWidgetItem(text)
            item.setForeground(Qt.GlobalColor.green if status.present else Qt.GlobalColor.red)
            self.list.addItem(item)
            item.setToolTip(status.message or status.name)


# ------------------------------------------------------------- prompt dialog


class YesNoDialog(QDialog):
    """Per-dependency Yes/No prompt with purpose, category and Internet info."""

    def __init__(self, name: str, purpose: str, detail: str, optional: bool,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Missing dependency")
        self.result_yes = False
        self.resize(460, 220)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>Required component missing:</b> {name}"))
        purpose_label = QLabel(purpose)
        purpose_label.setWordWrap(True)
        layout.addWidget(purpose_label)
        detail_label = QLabel(detail)
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet("color: #94a3b8;")
        layout.addWidget(detail_label)

        buttons = QHBoxLayout()
        yes = QPushButton("YES")
        no = QPushButton("NO")
        yes.clicked.connect(self._yes)
        no.clicked.connect(self._no)
        buttons.addWidget(yes)
        buttons.addWidget(no)
        layout.addLayout(buttons)
        yes.setFocus()

    def _yes(self) -> None:
        self.result_yes = True
        self.accept()

    def _no(self) -> None:
        self.result_yes = False
        self.accept()


class ProgressDialog(QDialog):
    """Modal progress shown while installing a dependency / pulling a model.

    The bar runs in indeterminate (animated) mode until a real percentage
    arrives, so long downloads always show visible activity instead of
    looking stuck.
    """

    def __init__(self, title: str, worker: callable, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(460, 140)
        layout = QVBoxLayout(self)
        self.label = QLabel("Working...")
        layout.addWidget(self.label)
        # Range 0..0 = busy indicator: animates until a real percent arrives.
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        layout.addWidget(self.bar)
        self.worker = worker
        self._thread = _ProgressThread(worker)
        self._thread.progress.connect(self._on_progress)
        self._thread.error.connect(self._on_error)
        self._thread.finished.connect(self.accept)

    def _on_progress(self, percent: float, message: str) -> None:
        self.label.setText(f"{message} ({percent:.0f}%)" if percent else message)
        if percent and percent > 0:
            # Real progress known — switch to a determinate bar.
            self.bar.setRange(0, 100)
            self.bar.setValue(int(percent))
        else:
            # Percentage unknown (e.g. no content-length) — keep animating.
            self.bar.setRange(0, 0)

    def _on_error(self, message: str) -> None:
        self.label.setText(f"<span style='color:#ef4444'>Failed: {message}</span>")

    def start(self) -> None:
        self._thread.start()
        super().exec()


class _ProgressThread(QThread):
    progress = Signal(float, str)
    error = Signal(str)

    def __init__(self, worker) -> None:
        super().__init__()
        self._worker = worker

    def run(self) -> None:
        try:
            self._worker(lambda p, m: self.progress.emit(float(p), str(m)))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.progress.emit(100.0, "Done")


# ------------------------------------------------------------ first-run UI


def _recommended_local_model() -> str:
    """Pick a default local model from available RAM (conservative)."""
    try:
        from utils.system import detect_hardware

        ram = detect_hardware().ram_total_gb
    except Exception:
        ram = 16.0
    if ram < 8:
        return "qwen3.5:2b"
    if ram >= 32:
        return "qwen3.5:9b"
    return "qwen3.5:4b"


class FirstRunWizard(QDialog):
    """Simplified first-run wizard.

    Three decisions on three short pages:
      1. names (assistant + user),
      2. connectivity mode — Fully Online / Fully Offline / Hybrid,
      3. the model for that mode (and API key when needed).
    Everything else (Whisper, VAD, TTS, hotkeys) is set up automatically
    with sensible CPU-friendly defaults.
    """

    def __init__(self, services=None, parent=None) -> None:
        super().__init__(parent)
        self.services = services  # optional: provides test_mic callback
        self.setWindowTitle("LocalVoiceAgent — Setup")
        self.resize(600, 480)

        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        self._build_pages()

        buttons = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_next = QPushButton("Next")
        self.btn_finish = QPushButton("Finish")
        self.btn_back.clicked.connect(self._back)
        self.btn_next.clicked.connect(self._next)
        self.btn_finish.clicked.connect(self.accept)
        buttons.addWidget(self.btn_back)
        buttons.addStretch(1)
        if self.services is not None and getattr(self.services, "test_mic", None):
            self.btn_test_mic = QPushButton("Test microphone")
            self.btn_test_mic.clicked.connect(self._test_mic)
            buttons.addWidget(self.btn_test_mic)
        buttons.addWidget(self.btn_next)
        buttons.addWidget(self.btn_finish)
        layout.addLayout(buttons)

        self.stack.setCurrentIndex(0)
        self._update_buttons()

    def _test_mic(self) -> None:
        try:
            self.services.test_mic()
        except Exception:
            pass

    # ---------------------------------------------------------------- pages

    def _build_pages(self) -> None:
        # Page 1 — welcome + names.
        page = QWidget()
        v = QVBoxLayout(page)
        title = QLabel("<h2>Welcome to LocalVoiceAgent</h2>")
        v.addWidget(title)
        intro = QLabel(
            "A voice assistant that can run fully offline on this PC.\n"
            "Setup takes about a minute — pick a name and how you want to "
            "connect.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        form = QFormLayout()
        self.cmb_agent_name = QComboBox()
        self.cmb_agent_name.setEditable(True)
        self.cmb_agent_name.addItems(AGENT_NAME_SUGGESTIONS)
        self.cmb_agent_name.setCurrentText("Luna")
        form.addRow("Assistant name", self.cmb_agent_name)
        self.ed_user_name = QLineEdit()
        self.ed_user_name.setPlaceholderText("e.g. Alex (optional)")
        form.addRow("Your name", self.ed_user_name)
        v.addLayout(form)
        v.addStretch(1)
        self.stack.addWidget(page)

        # Page 2 — connectivity mode.
        # NOTE: QRadioButton renders plain text only — never embed <b> in
        # its label or it shows literally. Bold emphasis comes from a
        # separate rich-text QLabel under each option.
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("<h2>How should the assistant connect?</h2>"))
        self.rb_hybrid = QRadioButton("Hybrid — recommended")
        self.rb_offline = QRadioButton("Fully offline")
        self.rb_online = QRadioButton("Fully online")
        descriptions = {
            self.rb_hybrid:
                "Uses the cloud model when online and automatically falls "
                "back to a local model offline. The local model is "
                "downloaded now so the fallback always works.",
            self.rb_offline:
                "Everything runs on this PC. Downloads the local AI models "
                "once (~1-2 GB): speech recognition, voice activity "
                "detection, text-to-speech and one local LLM of your "
                "choice. No internet needed afterwards.",
            self.rb_online:
                "No large downloads. Responses come from a cloud provider "
                "(Ollama Cloud, OpenRouter, Groq or OpenAI). Requires an "
                "account/API key and an internet connection.",
        }
        for rb in (self.rb_hybrid, self.rb_offline, self.rb_online):
            font = rb.font()
            font.setBold(True)
            rb.setFont(font)
            rb.setStyleSheet("QRadioButton { spacing: 8px; }"
                             "QRadioButton::indicator { width: 16px; height: 16px; }")
            v.addWidget(rb)
            desc = QLabel(descriptions[rb])
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #94a3b8; margin-left: 24px;")
            v.addWidget(desc)
            v.addSpacing(10)
        self.rb_hybrid.setChecked(True)
        tip = QLabel("Speech recognition and text-to-speech run locally in "
                     "all modes — your voice never has to leave this PC.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #94a3b8;")
        v.addWidget(tip)
        v.addStretch(1)
        self.stack.addWidget(page)

        # Page 3 — mode-specific configuration.
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("<h2>Choose your model</h2>"))
        self.cfg_stack = QStackedWidget()
        v.addWidget(self.cfg_stack)
        self._build_cfg_pages()
        v.addStretch(1)
        self.stack.addWidget(page)

        # Page 4 — finish summary.
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("<h2>You're all set!</h2>"))
        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        v.addWidget(self.lbl_summary)
        hk = QLabel("Hotkeys: hold <b>CTRL+WINDOWS</b> and speak "
                    "(push-to-talk), or press <b>CTRL+WINDOWS+SPACE</b> for "
                    "continuous listening that auto-detects when you stop "
                    "speaking.")
        hk.setWordWrap(True)
        v.addWidget(hk)
        v.addStretch(1)
        self.stack.addWidget(page)

    def _build_cfg_pages(self) -> None:
        # Offline: local model only.
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel(
            "This local LLM is downloaded now (~2-6 GB) and answers "
            "everything on this PC. A size matching your RAM is "
            "pre-selected."))
        self.cmb_local_offline = self._make_local_combo()
        v.addWidget(QLabel("Local model:"))
        v.addWidget(self.cmb_local_offline)
        v.addStretch(1)
        self.cfg_stack.addWidget(page)          # index 0 = offline

        # Online: provider + key + model.
        page = QWidget()
        v = QVBoxLayout(page)
        self.cmb_provider = QComboBox()
        from ollama.providers import PROVIDERS

        self._provider_ids = ["ollama", "openrouter", "groq", "openai"]
        for pid in self._provider_ids:
            self.cmb_provider.addItem(PROVIDERS[pid]["label"], pid)
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        v.addWidget(QLabel("Cloud provider:"))
        v.addWidget(self.cmb_provider)
        v.addWidget(QLabel("Model:"))
        self.cmb_cloud_model = QComboBox()
        self.cmb_cloud_model.setEditable(False)   # pick, not type
        v.addWidget(self.cmb_cloud_model)
        self.ed_api_key = QLineEdit()
        self.ed_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_api_key.setPlaceholderText("API key")
        v.addWidget(QLabel("API key:"))
        v.addWidget(self.ed_api_key)
        self.lbl_provider_hint = QLabel("")
        self.lbl_provider_hint.setWordWrap(True)
        self.lbl_provider_hint.setStyleSheet("color: #94a3b8;")
        v.addWidget(self.lbl_provider_hint)
        v.addStretch(1)
        self.cfg_stack.addWidget(page)          # index 1 = online

        # Hybrid: local fallback + Ollama cloud model.
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel(
            "When online, the Ollama cloud model below answers (sign in "
            "once with <code>ollama signin</code>). When offline, the local "
            "model takes over automatically — it downloads now."))
        v.addWidget(QLabel("Cloud model (online):"))
        self.cmb_hybrid_cloud = QComboBox()
        self.cmb_hybrid_cloud.setEditable(False)  # pick, not type
        self.cmb_hybrid_cloud.addItems(list(self.DEFAULT_CLOUD_MODELS))
        v.addWidget(self.cmb_hybrid_cloud)
        v.addWidget(QLabel("Local model (offline fallback):"))
        self.cmb_local_hybrid = self._make_local_combo()
        v.addWidget(self.cmb_local_hybrid)
        v.addStretch(1)
        self.cfg_stack.addWidget(page)          # index 2 = hybrid

    DEFAULT_LOCAL_MODELS = [
        "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b",
        "gemma4:e2b", "gemma4:e4b",
    ]
    DEFAULT_CLOUD_MODELS = [
        "gemma4:cloud", "deepseek-v4-flash:cloud", "glm-5.2:cloud",
        "minimax-m2.7:cloud", "qwen3.5:cloud", "nemotron-3-nano:30b-cloud",
    ]

    def _make_local_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(False)                  # pick, not type
        combo.addItems(list(self.DEFAULT_LOCAL_MODELS))
        recommended = _recommended_local_model()
        idx = combo.findText(recommended)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        return combo

    def _on_provider_changed(self) -> None:
        from ollama.providers import PROVIDERS

        pid = self.cmb_provider.currentData()
        meta = PROVIDERS.get(pid, {})
        models = meta.get("models") or []
        self.cmb_cloud_model.clear()
        self.cmb_cloud_model.addItems(models)
        needs_key = bool(meta.get("needs_api_key"))
        self.ed_api_key.setVisible(needs_key)
        self.lbl_provider_hint.setText(
            "Sign in once with <code>ollama signin</code> in a terminal."
            if pid == "ollama" else
            f"Create an API key at {meta.get('base_url', '')} and paste it "
            "above." if needs_key else "")

    # ------------------------------------------------------------ properties

    @property
    def agent_name(self) -> str:
        return self.cmb_agent_name.currentText().strip() or "Luna"

    @property
    def user_name(self) -> str:
        return self.ed_user_name.text().strip()

    @property
    def connectivity(self) -> str:
        if self.rb_offline.isChecked():
            return "offline"
        if self.rb_online.isChecked():
            return "online"
        return "hybrid"

    @property
    def provider(self) -> str:
        if not self.rb_online.isChecked():
            return "ollama"           # hybrid/online use the Ollama family
        pid = self.cmb_provider.currentData()
        return pid if pid else "ollama"

    @property
    def provider_model(self) -> str:
        return self.cmb_cloud_model.currentText().strip()

    @property
    def api_key(self) -> str:
        from ollama.providers import PROVIDERS

        pid = self.provider
        if pid != "ollama" and PROVIDERS.get(pid, {}).get("needs_api_key"):
            return self.ed_api_key.text().strip()
        return ""

    @property
    def local_model(self) -> str:
        if self.rb_online.isChecked():
            return self.settings_local_default()
        if self.rb_offline.isChecked():
            return self.cmb_local_offline.currentText().strip()
        return self.cmb_local_hybrid.currentText().strip()

    @staticmethod
    def settings_local_default() -> str:
        return _recommended_local_model()

    @property
    def cloud_model(self) -> str:
        if self.rb_hybrid.isChecked():
            return self.cmb_hybrid_cloud.currentText().strip()
        if (self.rb_online.isChecked()
                and self.cmb_provider.currentData() == "ollama"):
            return self.cmb_cloud_model.currentText().strip()
        return ""

    # ---------------------------------------------------------- navigation

    def _back(self) -> None:
        if self.stack.currentIndex() > 0:
            self.stack.setCurrentIndex(self.stack.currentIndex() - 1)
            self._update_buttons()

    def _next(self) -> None:
        index = self.stack.currentIndex()
        if index == 1:
            # Entering the config page: sync it to the chosen mode.
            cfg_index = {"hybrid": 2, "online": 1, "offline": 0}[self.connectivity]
            self.cfg_stack.setCurrentIndex(cfg_index)
            self._on_provider_changed()
        if index == 2:
            self.lbl_summary.setText(self._summary_text())
        if index < self.stack.count() - 1:
            self.stack.setCurrentIndex(index + 1)
            self._update_buttons()
        else:
            self.accept()

    def _summary_text(self) -> str:
        parts = [f"Assistant: <b>{self.agent_name}</b>",
                 f"Mode: <b>{self.connectivity}</b>"]
        if self.provider != "ollama":
            parts.append(f"Provider: <b>{self.provider}</b> "
                         f"(model {self.provider_model})")
        elif self.cloud_model and self.connectivity != "offline":
            parts.append(f"Cloud model: <b>{self.cloud_model}</b>")
        if self.connectivity != "online":
            parts.append(f"Local model: <b>{self.local_model}</b>")
        return " &nbsp;·&nbsp; ".join(parts)

    def _update_buttons(self) -> None:
        index = self.stack.currentIndex()
        last = index == self.stack.count() - 1
        self.btn_back.setEnabled(index > 0)
        self.btn_next.setVisible(not last)
        self.btn_finish.setVisible(last)
