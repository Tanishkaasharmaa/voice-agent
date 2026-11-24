#!/usr/bin/env python3
"""LocalVoiceAgent entry point.

Usage:
    python main.py            normal launch (floating widget + tray)
    python main.py --background   start minimized to the tray
    python main.py --show-logs    stream live logs to the terminal
"""

from __future__ import annotations

import argparse
import signal
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="LocalVoiceAgent")
    parser.add_argument("--background", action="store_true",
                        help="start in the background (tray only)")
    parser.add_argument("--show-logs", action="store_true",
                        help="stream live logs to the terminal (for testing/fixing)")
    parser.add_argument("--setup", action="store_true",
                        help="force the dependency setup dialogs on startup")
    args = parser.parse_args()

    # Importing the UI package pulls in PySide6; fail with a clear message
    # instead of a traceback when the UI dependency is missing.
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except Exception:
        print(
            "PySide6 is required to run LocalVoiceAgent.\n"
            "Install dependencies with:  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    from app.application import VoiceAgentApplication

    app = VoiceAgentApplication(sys.argv, background=args.background,
                                show_logs=args.show_logs)

    # Ctrl+C must quit cleanly. Without this, SIGINT raises KeyboardInterrupt
    # inside whatever Qt slot is running (e.g. the widget's 50 ms animation
    # timer); PySide6 swallows the exception and keeps the event loop alive,
    # so the app appears stuck and every repeated press spams a traceback.
    def _quit_on_sigint(*_args) -> None:
        app._exit()

    signal.signal(signal.SIGINT, _quit_on_sigint)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
