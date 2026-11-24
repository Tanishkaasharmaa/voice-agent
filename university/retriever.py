"""Small, deterministic lexical retriever for the initial 18 Wi-Fi FAQs."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_STOP = {"a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "use", "what", "when", "where", "why", "with", "you", "your", "university", "wifi", "wi", "fi", "time"}


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {word for word in words if word not in _STOP and len(word) > 1}


class FAQRetriever:
    def __init__(self, faqs: Iterable[dict[str, Any]], min_score: float = 0.42) -> None:
        self.faqs = list(faqs)
        self.min_score = min_score
        self._indexed = []
        for faq in self.faqs:
            searchable = " ".join([faq["question"], *faq.get("keywords", [])])
            self._indexed.append((faq, _tokens(searchable)))

    def search(self, query: str) -> tuple[dict[str, Any] | None, float, float]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return None, 0.0, 0.0
        ranked: list[tuple[float, dict[str, Any]]] = []
        for faq, faq_tokens in self._indexed:
            overlap = query_tokens & faq_tokens
            # Recall-oriented score works well for short spoken queries; require
            # at least two meaningful matching terms to avoid broad false hits.
            score = len(overlap) / len(query_tokens)
            if len(overlap) >= 2:
                ranked.append((score, faq))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if not ranked:
            return None, 0.0, 0.0
        best_score, best_faq = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < self.min_score:
            return None, best_score, second_score
        # If the top two results are nearly tied, do not guess which policy
        # the student means. Ask them to rephrase with more detail.
        if second_score and best_score - second_score < 0.12:
            return None, best_score, second_score
        return best_faq, best_score, second_score
