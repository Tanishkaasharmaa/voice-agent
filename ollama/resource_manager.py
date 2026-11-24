"""LLMResourceManager — keep both Ollama models from being resident at once.

On a 16 GB CPU-only machine, loading the 4B and 9B models together invites
memory thrashing. This manager:

* keeps the agent model (4B) warm with a bounded keep-alive;
* before a 9B run, unloads the 4B (and vice versa when RAM is low);
* downgrades to 4B when available RAM is critically low;
* exposes the current model state to the UI via the event bus.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.events import EventBus, TOPIC_MODEL_STATE
from ollama.client import OllamaClient

# Estimated floor for safe 9B residency, in GB. The 9B q4 quant usually needs
# ~6-7 GB; keep a healthy margin on a 16 GB machine.
MIN_RAM_FOR_9B_GB = 6.0


class LLMResourceManager:
    def __init__(
        self,
        client: OllamaClient,
        agent_model: str,
        reasoning_model: str,
        keep_alive: str = "10m",
        bus: EventBus | None = None,
        ram_provider=None,
    ) -> None:
        self.client = client
        self.agent_model = agent_model
        self.reasoning_model = reasoning_model
        self.keep_alive = keep_alive
        self.bus = bus
        # ram_provider() -> available RAM in GB (injectable for tests).
        self._ram_provider = ram_provider or self._default_ram
        # RLock: prepare() -> _emit() -> resident_models re-enters the lock.
        self._lock = threading.RLock()
        self._resident: set[str] = set()
        self._last_used: dict[str, float] = {}
        self._unload_after_9b = True

    @staticmethod
    def _default_ram() -> float:
        try:
            from utils.system import ResourceMonitor

            return ResourceMonitor.sample().ram_total_gb - ResourceMonitor.sample().ram_used_gb
        except Exception:
            return 16.0

    # ---------------------------------------------------------------- state

    @property
    def resident_models(self) -> list[str]:
        with self._lock:
            return sorted(self._resident)

    def available_ram_gb(self) -> float:
        return float(self._ram_provider())

    def safe_for_9b(self) -> bool:
        return self.available_ram_gb() >= MIN_RAM_FOR_9B_GB

    # ------------------------------------------------------------ lifecycle

    def prepare(self, model: str) -> None:
        """Ensure ``model`` is the one kept resident. Sequential residency."""
        if model not in (self.agent_model, self.reasoning_model):
            return
        with self._lock:
            other = self.reasoning_model if model == self.agent_model else self.agent_model
            # Single-model setup (agent == reasoning): there is nothing to
            # swap — unload/rewarm churn would reload the model every run.
            if other != model and other in self._resident:
                self.client.unload_model(other)
                self._resident.discard(other)
            # The reasoning RAM floor only applies when the reasoning slot is
            # a distinct model; the always-allowed agent model is exempt.
            if (model == self.reasoning_model and model != self.agent_model
                    and not self.safe_for_9b()):
                return  # caller should fall back to 4B
            self.client.generate_with_keep_alive(model, self.keep_alive)
            self._resident.add(model)
            self._last_used[model] = time.time()
            self._emit()

    def choose_model(self, preferred: str) -> str:
        """Return the model to actually use, applying RAM safety rules."""
        if preferred == self.reasoning_model and not self.safe_for_9b():
            return self.agent_model
        return preferred

    def on_resource_snapshot(self, ram_available_gb: float) -> str | None:
        """Called by the resource monitor. Returns a model to downgrade to,
        or None. Never terminates user processes; only shifts model choice."""
        if ram_available_gb < MIN_RAM_FOR_9B_GB and self.reasoning_model in self._resident:
            self.client.unload_model(self.reasoning_model)
            with self._lock:
                self._resident.discard(self.reasoning_model)
            self._emit()
            return self.agent_model
        return None

    def expire_idle_models(self, max_idle_s: float = 300.0) -> None:
        """Expire models that have been idle too long (keep-alive is bounded)."""
        now = time.time()
        with self._lock:
            for model in list(self._resident):
                if model == self.agent_model:
                    continue  # agent stays warm; it is the default
                if now - self._last_used.get(model, now) >= max_idle_s:
                    self.client.unload_model(model)
                    self._resident.discard(model)
            self._emit()

    def shutdown(self) -> None:
        with self._lock:
            for model in list(self._resident):
                self.client.unload_model(model)
            self._resident.clear()
        self._emit()

    # -------------------------------------------------------------- helpers

    def _emit(self) -> None:
        if self.bus:
            self.bus.publish(
                TOPIC_MODEL_STATE,
                {"resident": self.resident_models, "agent": self.agent_model,
                 "reasoning": self.reasoning_model},
            )
