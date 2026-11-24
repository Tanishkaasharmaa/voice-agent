"""ModelRouter — decide which Ollama model should handle a request.

Strategy (AUTO mode):
* routine commands / tool commands / short requests  -> agent model (4B)
* complex reasoning, long multi-step plans, explicit
  "think deeply" requests, or very long input          -> reasoning model (9B)

Modes: "auto" (default), "4b_only", "9b_only".
"""

from __future__ import annotations

import re

# Phrases the user may say to request deeper reasoning.
_DEEP_HINTS = (
    "think deeply", "think harder", "deep reasoning", "reason about", "analyze",
    "weigh the", "plan out", "multi-step", "step by step", "explain in detail",
    "consider all", "trade-offs", "pros and cons", "in depth", "complex",
    "complicated", "evaluate", "compare", "decide whether",
)

_COMPLEX_TOOLS = {
    # These usually need planning even though they're tool-driven.
    "plan", "compare", "analyze",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ModelRouter:
    def __init__(self, agent_model: str, reasoning_model: str, mode: str = "auto") -> None:
        self.agent_model = agent_model
        self.reasoning_model = reasoning_model
        self.mode = mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("auto", "4b_only", "9b_only"):
            raise ValueError(f"Unknown routing mode: {mode}")
        self.mode = mode

    def route(self, text: str, tool_names: list[str] | None = None) -> str:
        """Return the model name that should handle ``text``."""
        if self.mode == "4b_only":
            return self.agent_model
        if self.mode == "9b_only":
            return self.reasoning_model
        if self._is_complex(text, tool_names or []):
            return self.reasoning_model
        return self.agent_model

    # --------------------------------------------------------------- scoring

    def complexity_score(self, text: str, tool_names: list[str] | None = None) -> float:
        score = 0.0
        lowered = text.lower()
        for hint in _DEEP_HINTS:
            if hint in lowered:
                score += 1.0
        words = len(re.findall(r"\w+", text))
        if words > 120:
            score += 1.0
        if words > 300:
            score += 1.0
        if re.search(r"\b(first|then|next|finally)\b", lowered):
            score += 0.5
        for tool in tool_names or []:
            if tool in _COMPLEX_TOOLS:
                score += 0.5
        return score

    def _is_complex(self, text: str, tool_names: list[str]) -> bool:
        score = self.complexity_score(text, tool_names)
        return score >= 2.0

    # ----------------------------------------------------------- classification

    def classify(self, text: str) -> str:
        """Human-readable classification for the UI/logs."""
        if self.mode == "9b_only":
            return "complex (forced 9B)"
        if self.mode == "4b_only":
            return "routine (forced 4B)"
        return "complex (9B)" if self._is_complex(text, []) else "routine (4B)"
