from __future__ import annotations

from pathlib import Path

from .errors import ValidationFailure
from .okf import parse_markdown
from .state import public_concepts

_SNIPPET_RADIUS = 80


def _snippet(body: str, keywords: list[str]) -> str:
    lowered = body.casefold()
    for keyword in keywords:
        index = lowered.find(keyword.casefold())
        if index == -1:
            continue
        start = max(0, index - _SNIPPET_RADIUS)
        end = min(len(body), index + len(keyword) + _SNIPPET_RADIUS)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(body) else ""
        return prefix + " ".join(body[start:end].split()) + suffix
    return " ".join(body.split())[: _SNIPPET_RADIUS * 2]


def search_concepts(bundle: Path, keywords: list[str]) -> list[dict[str, object]]:
    """Find existing Concept documents whose id, title, or body contain any keyword."""
    keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
    if not keywords:
        return []
    matches: list[dict[str, object]] = []
    for concept_id, raw in public_concepts(bundle).items():
        try:
            metadata, body = parse_markdown(raw)
        except ValidationFailure:
            continue
        title = str(metadata.get("title") or "")
        haystack = f"{concept_id}\n{title}\n{body}".casefold()
        if any(keyword.casefold() in haystack for keyword in keywords):
            matches.append(
                {"concept_id": concept_id, "title": title, "snippet": _snippet(body, keywords)}
            )
    return matches
