"""Load and validate the local university FAQ dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KnowledgeBaseError(ValueError):
    """Raised when the FAQ file is malformed or incomplete."""


def default_faq_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "wifi_faq.json"


def load_faqs(path: str | Path | None = None) -> list[dict[str, Any]]:
    faq_path = Path(path) if path is not None else default_faq_path()
    try:
        payload = json.loads(faq_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"Could not load FAQ data from {faq_path}: {exc}") from exc
    faqs = payload.get("faqs") if isinstance(payload, dict) else None
    if not isinstance(faqs, list) or not faqs:
        raise KnowledgeBaseError("FAQ file must contain a non-empty 'faqs' list")
    seen: set[str] = set()
    for index, item in enumerate(faqs):
        if not isinstance(item, dict):
            raise KnowledgeBaseError(f"FAQ entry {index} must be an object")
        for field in ("id", "question", "answer"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise KnowledgeBaseError(f"FAQ entry {index} has missing or invalid '{field}'")
        if item["id"] in seen:
            raise KnowledgeBaseError(f"Duplicate FAQ id: {item['id']}")
        seen.add(item["id"])
        if not isinstance(item.get("keywords", []), list):
            raise KnowledgeBaseError(f"FAQ entry {index} keywords must be a list")
    return faqs
