"""UI styles and per-state colors."""

from __future__ import annotations

# State -> (core color, label)
STATE_COLORS = {
    "IDLE": ("#64748b", "slate"),
    "LISTENING": ("#3b82f6", "blue"),
    "TRANSCRIBING": ("#eab308", "yellow"),
    "THINKING": ("#a855f7", "purple"),
    "EXECUTING_TOOL": ("#f97316", "orange"),
    "SPEAKING": ("#22c55e", "green"),
    "ERROR": ("#ef4444", "red"),
}

APP_QSS = """
QMainWindow, QDialog {
    background: #0f172a;
    color: #e2e8f0;
}
QLabel { color: #e2e8f0; }
QPushButton {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 14px;
}
QPushButton:hover { background: #334155; }
QPushButton:disabled { color: #64748b; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
}
QGroupBox {
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QListWidget, QTreeWidget { background: #0f172a; border: 1px solid #334155; }
QCheckBox { color: #e2e8f0; }
QTabWidget::pane { border: 1px solid #334155; }
QToolTip { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
QListWidget#settingsSidebar {
    background: #111a2e;
    border: none;
    border-right: 1px solid #334155;
    outline: none;
    font-weight: 600;
}
QListWidget#settingsSidebar::item {
    padding: 8px 10px;
    border-radius: 6px;
    margin: 2px 6px 2px 6px;
}
QListWidget#settingsSidebar::item:hover { background: #1e293b; }
QListWidget#settingsSidebar::item:selected {
    background: #312e52;
    color: #ffffff;
}
"""
