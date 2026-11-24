"""System tray icon and menu."""

from __future__ import annotations

from PySide6.QtGui import QAction, QActionGroup, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon


class SystemTray:
    def __init__(self, icon: QIcon, parent=None) -> None:
        self.tray = QSystemTrayIcon(icon, parent)
        self.tray.setToolTip("LocalVoiceAgent")
        self.menu = QMenu()

        self.act_show_hide = QAction("Show/Hide Assistant")
        self.act_settings = QAction("Settings")
        self.act_history = QAction("History")
        self.menu.addAction(self.act_show_hide)
        self.menu.addAction(self.act_settings)
        self.menu.addAction(self.act_history)
        self.menu.addSeparator()

        # Assistant mode is selectable from the tray as well as Settings.
        self.mode_menu = QMenu("Assistant Mode")
        self.mode_group = QActionGroup(self.mode_menu)
        self.mode_group.setExclusive(True)
        self.act_general_mode = QAction("General assistant", self.mode_menu)
        self.act_general_mode.setCheckable(True)
        self.act_general_mode.setData("general")
        self.act_university_mode = QAction("University help desk", self.mode_menu)
        self.act_university_mode.setCheckable(True)
        self.act_university_mode.setData("university")
        self.mode_group.addAction(self.act_general_mode)
        self.mode_group.addAction(self.act_university_mode)
        self.mode_menu.addAction(self.act_general_mode)
        self.mode_menu.addAction(self.act_university_mode)
        self.menu.addMenu(self.mode_menu)
        self.set_assistant_mode("general")
        self.menu.addSeparator()

        self.act_enable_voice = QAction("Enable Voice")
        self.act_disable_voice = QAction("Disable Voice")
        self.act_enable_voice.setCheckable(True)
        self.act_enable_voice.setChecked(True)
        self.menu.addAction(self.act_enable_voice)
        self.menu.addAction(self.act_disable_voice)
        self.menu.addSeparator()

        self.microphone_menu = QMenu("Microphone")
        self.speaker_menu = QMenu("Speaker")
        self.menu.addMenu(self.microphone_menu)
        self.menu.addMenu(self.speaker_menu)
        self.menu.addSeparator()

        self.act_reload_models = QAction("Reload Models")
        self.act_restart = QAction("Restart Agent")
        self.act_exit = QAction("Exit")
        self.menu.addAction(self.act_reload_models)
        self.menu.addAction(self.act_restart)
        self.menu.addAction(self.act_exit)

        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)

    def set_assistant_mode(self, mode: str) -> None:
        """Reflect the current mode in the exclusive tray menu."""
        mode = mode if mode in {"general", "university"} else "general"
        self.act_general_mode.setChecked(mode == "general")
        self.act_university_mode.setChecked(mode == "university")

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            callback = getattr(self.tray, "activated_callback", None)
            if callback:
                callback()

    def show(self) -> None:
        self.tray.show()

    def hide(self) -> None:
        self.tray.hide()

    def show_message(self, title: str, message: str) -> None:
        self.tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 4000)

    def populate_devices(self, kind: str, devices: list[dict], current: str | None,
                         on_select) -> None:
        menu = self.microphone_menu if kind == "microphone" else self.speaker_menu
        menu.clear()
        for device in devices:
            action = QAction(device["name"], menu)
            action.setCheckable(True)
            action.setChecked(device["name"] == current)
            action.triggered.connect(lambda _checked=False, d=device: on_select(d["name"]))
            menu.addAction(action)
