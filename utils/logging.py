"""Logging configuration for LocalVoiceAgent.

Never log microphone audio, secrets, passwords, or sensitive clipboard
contents. The ``redact`` helper strips obvious secrets from messages before
they reach the handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path

_SENSITIVE = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|authorization)\s*[:=]\s*\S+"
)

# Module-level state so tests can re-configure cleanly.
_configured = False


def redact(message: str) -> str:
    """Replace obvious secrets in a message with '***'."""
    return _SENSITIVE.sub(r"\1=***", str(message))


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.msg = redact(record.getMessage())
        record.args = ()
        return super().format(record)


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """Configure root logging once. Returns the 'localvoiceagent' logger.

    ``console=False`` keeps the terminal quiet (logs still go to the file);
    pass ``--show-logs`` at launch to stream them live to stderr.
    """
    global _configured
    root = logging.getLogger("localvoiceagent")
    if _configured and root.handlers:
        return root
    root.setLevel(level)
    root.propagate = False

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "localvoiceagent.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"localvoiceagent.{name}")
