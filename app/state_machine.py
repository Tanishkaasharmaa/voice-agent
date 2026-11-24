"""State machine for LocalVoiceAgent.

Single source of truth for the assistant lifecycle. Workers request
transitions through the Application; the machine validates every move and
publishes ``TOPIC_STATE`` on change so the UI only ever renders state.
"""

from __future__ import annotations

import threading
from typing import Callable

from app.events import EventBus, TOPIC_STATE

# States
IDLE = "IDLE"
LISTENING = "LISTENING"
TRANSCRIBING = "TRANSCRIBING"
THINKING = "THINKING"
EXECUTING_TOOL = "EXECUTING_TOOL"
SPEAKING = "SPEAKING"
ERROR = "ERROR"

ALL_STATES = (IDLE, LISTENING, TRANSCRIBING, THINKING, EXECUTING_TOOL, SPEAKING, ERROR)

# Allowed transitions.
TRANSITIONS: dict[str, set[str]] = {
    IDLE: {LISTENING, ERROR, THINKING},
    LISTENING: {TRANSCRIBING, IDLE, ERROR},
    TRANSCRIBING: {THINKING, IDLE, ERROR, LISTENING},
    THINKING: {EXECUTING_TOOL, SPEAKING, IDLE, ERROR, LISTENING},
    EXECUTING_TOOL: {THINKING, IDLE, ERROR},
    SPEAKING: {LISTENING, IDLE, ERROR},  # LISTENING = interruption
    ERROR: {IDLE, ERROR},
}


class InvalidTransition(Exception):
    pass


class StateMachine:
    def __init__(self, bus: EventBus, initial: str = IDLE) -> None:
        if initial not in ALL_STATES:
            raise ValueError(f"Unknown state: {initial}")
        self._state = initial
        self._bus = bus
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, str], None]] = []

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def is_in(self, *states: str) -> bool:
        with self._lock:
            return self._state in states

    def add_listener(self, listener: Callable[[str, str], None]) -> None:
        """Called with (old_state, new_state) after every successful change."""
        self._listeners.append(listener)

    def can(self, target: str) -> bool:
        return target in TRANSITIONS.get(self._state, set())

    def transition(self, target: str) -> str:
        """Request a transition. Raises InvalidTransition when not allowed."""
        if target not in ALL_STATES:
            raise InvalidTransition(f"Unknown target state: {target}")
        with self._lock:
            if target not in TRANSITIONS.get(self._state, set()):
                raise InvalidTransition(f"{self._state} -> {target} is not allowed")
            old = self._state
            self._state = target
        for listener in self._listeners:
            try:
                listener(old, target)
            except Exception:
                pass
        self._bus.publish(TOPIC_STATE, target)
        return target

    def reset(self) -> None:
        """Force back to IDLE (used by ERROR recovery)."""
        with self._lock:
            if self._state != IDLE:
                old = self._state
                self._state = IDLE
                for listener in self._listeners:
                    try:
                        listener(old, IDLE)
                    except Exception:
                        pass
                self._bus.publish(TOPIC_STATE, IDLE)
