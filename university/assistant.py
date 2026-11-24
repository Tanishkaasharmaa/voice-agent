"""University-mode response policy: answer only from the local Wi-Fi FAQ."""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from university.knowledge_base import load_faqs
from university.retriever import FAQRetriever


_SWITCH_TO_UNIVERSITY = (
    "switch to university help desk", "university help desk mode",
    "switch to university mode", "enable university mode", "help desk mode",
)
_SWITCH_TO_GENERAL = (
    "switch to general assistant", "switch to general mode",
    "general assistant mode", "enable general mode",
    "exit university help desk", "leave university mode",
)


def detect_mode_command(transcript: str) -> str | None:
    """Return 'university' or 'general' for explicit mode-switch phrases."""
    text = re.sub(r"[^a-z0-9 ]+", " ", transcript.lower())
    text = " ".join(text.split())
    if any(phrase in text for phrase in _SWITCH_TO_UNIVERSITY):
        return "university"
    if any(phrase in text for phrase in _SWITCH_TO_GENERAL):
        return "general"
    return None


class UniversityAssistant:
    def __init__(self, faqs: list[dict[str, Any]] | None = None, processor: Any | None = None) -> None:
        self.faqs = faqs if faqs is not None else load_faqs()
        self.retriever = FAQRetriever(self.faqs)
        self.processor = processor

    def answer(self, transcript: str) -> str:
        faq, score, _ = self.retriever.search(transcript)
        if faq is None:
            return (
                "I couldn't find a sufficiently clear answer in the currently loaded Wi-Fi FAQ. "
                "Please rephrase your question with a little more detail, or contact the university help desk. "
                "You can say, switch to general assistant, to leave university help desk mode."
            )
        answer = faq["answer"].strip()
        if not faq.get("verified", False):
            answer = "According to the currently supplied Wi-Fi FAQ, " + answer[0].lower() + answer[1:]
        return answer

    def answer_sentences(self, transcript: str) -> list[str]:
        answer = self.answer(transcript)
        if self.processor is not None:
            return self.processor.split_sentences(answer)
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
