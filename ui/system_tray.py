"""System tray icon and menu."""

from __future__ import annotations

from PySide6.QtGui import QAction, QIcon
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

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.tray.activated_callback:
                self.tray.activated_callback()

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
