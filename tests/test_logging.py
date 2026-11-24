"""Logging configuration tests."""

from __future__ import annotations

import logging

from utils import logging as logging_util


def _reset(monkeypatch) -> None:
    root = logging.getLogger("localvoiceagent")
    root.handlers = []
    monkeypatch.setattr(logging_util, "_configured", False)


def test_setup_logging_quiet_by_default(monkeypatch):
    _reset(monkeypatch)
    logger = logging_util.setup_logging(console=False)
    try:
        assert not any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        ), "console=False must not attach a terminal handler"
    finally:
        logger.handlers = []


def test_setup_logging_show_logs_attaches_stream_handler(monkeypatch):
    _reset(monkeypatch)
    logger = logging_util.setup_logging(console=True)
    try:
        assert any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        ), "console=True must attach a terminal handler"
    finally:
        logger.handlers = []
        monkeypatch.setattr(logging_util, "_configured", False)
