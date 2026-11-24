"""App-level resource orchestration.

Bridges the low-level ResourceMonitor to the LLMResourceManager and warns the
user (via the tray) when RAM is critically low. Never terminates user
processes and never sacrifices Windows stability for a bigger model.
"""

from __future__ import annotations

import threading

from app.events import EventBus, TOPIC_RESOURCE
from ollama.resource_manager import LLMResourceManager
from utils.logging import get_logger
from utils.system import ResourceMonitor, ResourceSnapshot


class AppResourceManager:
    def __init__(self, llm_resources: LLMResourceManager, bus: EventBus,
                 interval_s: float = 5.0) -> None:
        self.llm_resources = llm_resources
        self.bus = bus
        self.monitor = ResourceMonitor(interval_s)
        self.on_warning: callable | None = None  # (title, message) -> None
        self._downgraded = threading.Event()
        self._last_warned_at = 0.0

    def start(self) -> None:
        self.monitor.start(self._on_snapshot)

    def stop(self) -> None:
        self.monitor.stop()

    def _on_snapshot(self, snapshot: ResourceSnapshot) -> None:
        self.bus.publish(TOPIC_RESOURCE, snapshot)
        import time

        now = time.monotonic()
        if snapshot.ram_critical and now - self._last_warned_at > 60:
            self._last_warned_at = now
            free = max(0.0, snapshot.ram_total_gb - snapshot.ram_used_gb)
            message = (
                f"RAM is critically low ({free:.1f} GB free). The 9B model "
                "will be avoided until memory frees up."
            )
            if self.on_warning:
                try:
                    self.on_warning("Low memory", message)
                except Exception:
                    pass
        try:
            downgrade_to = self.llm_resources.on_resource_snapshot(
                snapshot.ram_total_gb - snapshot.ram_used_gb
            )
            if downgrade_to is not None:
                self._downgraded.set()
        except Exception:
            get_logger("resources").exception("resource snapshot handling failed")
