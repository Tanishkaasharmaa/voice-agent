"""A tiny thread-safe event bus.

Workers publish events; the UI and the resource manager subscribe. Handlers
run on the publisher's thread — UI subscribers must marshal to the Qt thread
(the Application provides a helper for that).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

Handler = Callable[[str, Any], None]

# Canonical topics. Keep them strings so the bus stays dependency-free.
TOPIC_STATE = "state"                 # payload: new state name
TOPIC_TRANSCRIPT = "transcript"       # payload: {"final": bool, "text": str}
TOPIC_LLM_CHUNK = "llm_chunk"         # payload: {"text": str}
TOPIC_AGENT_STATUS = "agent_status"   # payload: {"status": str, "detail": str}
TOPIC_TTS_STARTED = "tts_started"     # payload: {"text": str}
TOPIC_TTS_STOPPED = "tts_stopped"     # payload: {}
TOPIC_TTS_AMPLITUDE = "tts_amplitude"  # payload: {"level": float 0..1 RMS}
TOPIC_ERROR = "error"                 # payload: {"component": str, "message": str}
TOPIC_RESOURCE = "resource"           # payload: ResourceSnapshot
TOPIC_MODEL_STATE = "model_state"     # payload: {"current_model": str, "resident": [..]}
TOPIC_LOG = "log"                     # payload: {"level": str, "message": str}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Subscribe; returns an unsubscribe callable."""
        with self._lock:
            self._subscribers[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers[topic].remove(handler)
                except ValueError:
                    pass

        return unsubscribe

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(topic, ()))
        for handler in handlers:
            try:
                handler(topic, payload)
            except Exception:
                # A broken subscriber must never break the pipeline.
                pass
