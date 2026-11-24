"""Lightweight, deterministic classifier for university help-desk scope.

This classifier does not generate answers. It only chooses which safe fallback
to use when the local FAQ retriever cannot find a confident match.
"""
from __future__ import annotations

import re

UNIVERSITY_UNCLEAR = "university_unclear"
OUT_OF_SCOPE = "out_of_scope"

# Terms that strongly indicate the university IT/help-desk domain.
_STRONG_SUPPORT_TERMS = {
    "wifi", "internet", "network", "connectivity", "connection",
    "router", "password", "login", "signin", "account", "samarth",
    "portal", "mac", "enrolment", "enrollment", "registration",
    "verification", "provisioning", "computercentre", "computercenter",
}

# Broader university context terms. These need a support/action signal too,
# so a question such as "What time does the university library close?" is not
# automatically classified as a help-desk question.
_UNIVERSITY_CONTEXT_TERMS = {
    "university", "college", "student", "hostel", "provost",
    "chairman", "department", "campus", "library", "id",
}

_SUPPORT_INTENT_TERMS = {
    "help", "support", "problem", "issue", "error", "broken",
    "failed", "failing", "working", "connect", "connecting",
    "access", "register", "registration", "apply", "approval",
    "approve", "pending", "verify", "verification", "reset",
    "change", "update", "add", "remove", "replace", "new",
    "unable", "cannot", "cant", "doesnt",
    "request", "complaint", "blocked", "rejected", "rejection",
    "password", "login", "signin", "account", "device",
}

_ISSUE_PHRASES = (
    "not working", "doesn't work", "doesnt work", "can't connect",
    "cannot connect", "unable to connect", "having trouble",
    "having a problem", "facing a problem", "getting an error",
    "no internet", "no access", "not able to",
)


def _normalise(text: str) -> tuple[set[str], str]:
    normalised = re.sub(r"[^a-z0-9]+", " ", text.lower())
    normalised = " ".join(normalised.split())
    return set(normalised.split()), normalised


def classify_scope(text: str) -> str:
    """Return UNIVERSITY_UNCLEAR or OUT_OF_SCOPE for an unmatched query."""
    tokens, normalised = _normalise(text)
    if not tokens:
        return OUT_OF_SCOPE

    if any(phrase in normalised for phrase in _ISSUE_PHRASES):
        if tokens & (_STRONG_SUPPORT_TERMS | _UNIVERSITY_CONTEXT_TERMS):
            return UNIVERSITY_UNCLEAR

    # A strong IT/help-desk term is enough to invite clarification. This
    # catches paraphrases such as "my internet keeps dropping".
    if tokens & _STRONG_SUPPORT_TERMS:
        return UNIVERSITY_UNCLEAR

    # Broader university terms count only when paired with a support/action
    # signal, reducing false positives for general campus trivia.
    if (
        tokens & _UNIVERSITY_CONTEXT_TERMS
        and tokens & _SUPPORT_INTENT_TERMS
    ):
        return UNIVERSITY_UNCLEAR

    return OUT_OF_SCOPE
