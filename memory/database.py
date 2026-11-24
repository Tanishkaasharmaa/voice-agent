"""SQLite access layer.

Rules for background-worker safety:
* one connection per thread (threading.local), never shared unsafely;
* WAL mode + busy_timeout so readers never block writers for long;
* every write is a short transaction committed immediately;
* ``execute``/``executemany`` are convenience wrappers used by the manager.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._local = threading.local()
        self._conns: set[sqlite3.Connection] = set()
        self._conns_lock = threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------ connection

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None = autocommit: raw DML never leaves an implicit
        # transaction open on a shared-thread connection. Explicit BEGIN/COMMIT
        # via transaction() still works.
        conn = sqlite3.connect(
            self.path, timeout=10.0, check_same_thread=False, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        with self._conns_lock:
            self._conns.add(conn)
        return conn

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def close_thread_connection(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                with self._conns_lock:
                    self._conns.discard(conn)
                self._local.conn = None

    def close_all(self) -> None:
        """Close every live connection. Call once at application shutdown so no
        SQLite file handle stays open on Windows."""
        with self._conns_lock:
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        with self._conns_lock:
            self._conns.clear()
        self._local.conn = None

    def _init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connection() as conn:
            conn.executescript(sql)

    # ------------------------------------------------------------- wrappers

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Short-lived write transaction."""
        conn = self.connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.connection().execute(sql, params)

    def executemany(self, sql: str, seq: list[tuple]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, seq)

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self.connection().execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: tuple = ()) -> Any:
        row = self.connection().execute(sql, params).fetchone()
        return row[0] if row else None
