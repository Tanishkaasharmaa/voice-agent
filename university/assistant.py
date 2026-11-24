"""University-mode response policy: answer only from the local Wi-Fi FAQ."""
from __future__ import annotations

import re
from typing import Any

from university.knowledge_base import load_faqs
from university.retriever import FAQRetriever
from university.scope_classifier import UNIVERSITY_UNCLEAR, OUT_OF_SCOPE, classify_scope


_SWITCH_TO_UNIVERSITY = (
    "switch to university help desk", "university help desk mode",
    "switch to university mode", "enable university mode", "help desk mode",
)
_SWITCH_TO_GENERAL = (
    "switch to general assistant", "switch to general mode",
    "general assistant mode", "enable general mode",
    "exit university help desk", "leave university mode",
)

REPHRASE_RESPONSE = (
    "Please rephrase your question with a little more detail, such as the "
    "service, device, or error you are experiencing."
)

OUT_OF_SCOPE_RESPONSE = (
    "I can only help "
    "with university help desk questions covered by this service. "
    "That question appears to be outside my scope. "
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
    def __init__(
        self,
        faqs: list[dict[str, Any]] | None = None,
        processor: Any | None = None,
    ) -> None:
        self.faqs = faqs if faqs is not None else load_faqs()
        self.retriever = FAQRetriever(self.faqs)
        self.processor = processor

    def answer(self, transcript: str) -> str:
        faq, _score, _second_score = self.retriever.search(transcript)

        # Outcome 1: a sufficiently confident local FAQ match.
        if faq is not None:
            answer = faq["answer"].strip()
            if not faq.get("verified", False) and answer:
                answer = (
                    "As per the records, "
                    + answer[0].lower()
                    + answer[1:]
                )
            return answer

        # Outcomes 2 and 3: distinguish a likely support question from
        # something unrelated. Neither fallback calls the general LLM.
        scope = classify_scope(transcript)
        if scope == UNIVERSITY_UNCLEAR:
            return REPHRASE_RESPONSE
        if scope == OUT_OF_SCOPE:
            return OUT_OF_SCOPE_RESPONSE

        # Defensive fallback if the classifier is extended in the future.
        return REPHRASE_RESPONSE

    def answer_sentences(self, transcript: str) -> list[str]:
        answer = self.answer(transcript)
        if self.processor is not None:
            return self.processor.split_sentences(answer)
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?])\\s+", answer)
            if part.strip()
        ]
