"""MemoryManager — persistence for conversations, messages, and memories.

Only *useful* information is stored as long-term memory (explicit
``save_memory`` tool calls, notes, or the user's "remember that ..."
requests). Ordinary utterances live only in the conversation history.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from memory.database import Database


class MemoryManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._session_id = f"session-{uuid.uuid4().hex[:12]}"
        self._conversation_id: int | None = None

    # ----------------------------------------------------------- conversations

    def new_conversation(self, title: str = "") -> int:
        session = f"conv-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (session_id, title) VALUES (?, ?)",
                (session, title or "Voice session"),
            )
            self._conversation_id = int(cur.lastrowid)
        return self._conversation_id

    def current_conversation_id(self) -> int:
        if self._conversation_id is None:
            self._conversation_id = self.new_conversation()
        return self._conversation_id

    def add_message(
        self,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_input: Any = None,
        tool_result: str | None = None,
        error: str | None = None,
    ) -> int:
        conv_id = self.current_conversation_id()
        with self.db.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, tool_name, tool_input,
                    tool_result, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    conv_id,
                    role,
                    content,
                    tool_name,
                    __import__("json").dumps(tool_input) if tool_input is not None else None,
                    tool_result,
                    error,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conv_id,),
            )
            return int(cur.lastrowid)

    def get_conversation_messages(self, conversation_id: int | None = None,
                                  limit: int = 50) -> list[dict[str, Any]]:
        conv_id = conversation_id or self.current_conversation_id()
        return self.db.query(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conv_id, limit),
        )

    def list_conversations(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT id, session_id, title, started_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    def delete_conversation(self, conversation_id: int) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # ---------------------------------------------------------------- memories

    def save_memory(self, content: str, kind: str = "memory", tags: list[str] | None = None) -> int:
        tag_str = ",".join(tags) if tags else ""
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO memories (content, kind, tags) VALUES (?, ?, ?)",
                (content, kind, tag_str),
            )
            return int(cur.lastrowid)

    def search_memory(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Simple LIKE search across memory content and tags."""
        like = f"%{query}%"
        return self.db.query(
            "SELECT * FROM memories WHERE content LIKE ? OR tags LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (like, like, limit),
        )

    def list_memories(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if kind:
            return self.db.query(
                "SELECT * FROM memories WHERE kind = ? ORDER BY updated_at DESC LIMIT ?",
                (kind, limit),
            )
        return self.db.query("SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,))

    def clear_memories(self, kind: str | None = None) -> int:
        """Delete all memories (optionally one kind). Returns rows deleted."""
        if kind:
            cur = self.db.execute("DELETE FROM memories WHERE kind = ?", (kind,))
        else:
            cur = self.db.execute("DELETE FROM memories")
        return cur.rowcount

    def delete_memory(self, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    def update_memory(self, memory_id: int, content: str) -> bool:
        cur = self.db.execute(
            "UPDATE memories SET content = ?, updated_at = datetime('now') WHERE id = ?",
            (content, memory_id),
        )
        return cur.rowcount > 0

    def get_relevant_memories(self, text: str, limit: int | None = None,
                              retention_days: int | None = None) -> list[dict[str, Any]]:
        """Best-effort relevance: match significant words from the request."""
        import re

        words = [w for w in re.findall(r"[A-Za-z]{4,}", text.lower()) if w not in _STOP]
        if not words:
            return []
        limit = limit or 5
        rows = self.db.query(
            "SELECT * FROM memories WHERE kind = 'memory' ORDER BY updated_at DESC LIMIT 200"
        )
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            hay = f"{row['content']} {row['tags'] or ''}".lower()
            score = sum(1 for w in words if w in hay)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _, row in scored[:limit]]

    # ------------------------------------------------------ settings / events

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.db.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                   updated_at = datetime('now')""",
                (key, value),
            )

    def log_event(self, level: str, component: str, message: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO application_events (level, component, message) VALUES (?, ?, ?)",
                (level, component, message),
            )

    def record_tool_execution(self, tool_name: str, tool_input: Any, success: bool,
                              output: str | None = None, error: str | None = None,
                              duration_ms: int | None = None) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO tool_executions
                   (tool_name, input, output, success, error, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (tool_name, __import__("json").dumps(tool_input)[:2000] if tool_input else None,
                 output[:2000] if output else None, int(success),
                 (error or "")[:2000], duration_ms),
            )

    def record_install(self, name: str, status: str, version: str | None = None,
                       error: str | None = None) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO install_history
                   (dependency_name, version, installation_status, last_error)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(dependency_name) DO UPDATE SET
                     version = excluded.version,
                     installation_status = excluded.installation_status,
                     last_error = excluded.last_error,
                     last_checked = datetime('now')""",
                (name, version, status, error),
            )


_STOP = {
    "the", "and", "that", "this", "with", "from", "your", "have", "what",
    "when", "where", "which", "while", "would", "should", "could", "about",
    "there", "their", "please", "hello", "agent", "assistant", "remember",
}
