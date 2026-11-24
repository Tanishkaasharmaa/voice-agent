"""Conversation history dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTextEdit, QVBoxLayout,
)


class HistoryDialog(QDialog):
    def __init__(self, memory, parent=None) -> None:
        super().__init__(parent)
        self.memory = memory
        self.setWindowTitle("LocalVoiceAgent — History")
        self.resize(760, 480)

        layout = QHBoxLayout(self)
        splitter = QSplitter(self)

        self.conv_list = QListWidget()
        self.conv_list.currentItemChanged.connect(self._show_conversation)
        splitter.addWidget(self.conv_list)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        splitter.addWidget(self.detail)
        splitter.setSizes([260, 500])
        layout.addWidget(splitter)

        buttons = QVBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._load_conversations)
        btn_delete = QPushButton("Delete selected")
        btn_delete.clicked.connect(self._delete_selected)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        buttons.addWidget(btn_refresh)
        buttons.addWidget(btn_delete)
        buttons.addStretch(1)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)

        self._load_conversations()

    def _load_conversations(self) -> None:
        self.conv_list.clear()
        self._conversation_ids: dict[str, int] = {}
        for conv in self.memory.list_conversations(limit=50):
            item = QListWidgetItem(conv["title"])
            item.setToolTip(f"{conv['updated_at']} · {conv['session_id']}")
            self.conv_list.addItem(item)
            self._conversation_ids[conv["session_id"]] = conv["id"]

    def _show_conversation(self, current, _previous) -> None:
        if current is None:
            return
        session = current.toolTip().split(" · ")[-1]
        conv_id = self._conversation_ids.get(session)
        if conv_id is None:
            return
        messages = self.memory.get_conversation_messages(conv_id, limit=100)
        lines = []
        for msg in reversed(messages):
            role = msg["role"].upper()
            if msg.get("tool_name"):
                lines.append(f"[TOOL {msg['tool_name']}]")
            if msg.get("error"):
                lines.append(f"[ERROR] {msg['error']}")
            lines.append(f"--- {role} ---")
            lines.append(msg.get("content") or "")
            lines.append("")
        self.detail.setPlainText("\n".join(lines))

    def _delete_selected(self) -> None:
        item = self.conv_list.currentItem()
        if item is None:
            return
        session = item.toolTip().split(" · ")[-1]
        conv_id = self._conversation_ids.get(session)
        if conv_id is not None:
            self.memory.delete_conversation(conv_id)
        self._load_conversations()
        self.detail.clear()
