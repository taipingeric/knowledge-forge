from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import Conflict, Evidence
from .okf import dump_markdown, parse_markdown
from .sources import sha256_text

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
MISSING = object()
TOOL_MANAGED = {"generated", "sources", "verified"}


def normalize_heading(value: str) -> str:
    """Normalize heading text for stable structural block identifiers."""

    return re.sub(r"\s+", " ", value.strip()).casefold()


def body_sections(body: str) -> dict[str, str]:
    """Split Markdown into heading-path blocks while preserving their raw text."""

    matches = list(HEADING.finditer(body))
    sections: dict[str, str] = {}
    preamble = body[: matches[0].start()] if matches else body
    sections["body:preamble"] = preamble
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = normalize_heading(match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        key = "body:" + "/".join(item[1] for item in stack)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        if key in sections:
            key = f"{key}#{index + 1}"
        sections[key] = body[match.start() : end]
    return sections


def join_sections(sections: dict[str, str]) -> str:
    """Reassemble structural Markdown blocks in mapping iteration order."""

    return "".join(sections.values()).strip() + "\n"


def _merge_value(base: Any, human: Any, candidate: Any) -> tuple[Any, bool]:
    """Apply three-way merge rules and report whether both sides changed a value."""

    if human == base:
        return candidate, False
    if candidate == base or human == candidate:
        return human, False
    return human, True


@dataclass
class MergeResult:
    """Hold merged Markdown and the conflicts discovered during reconciliation."""

    markdown: str
    conflicts: list[Conflict]
    human_changed: bool


def merge_concept(
    concept_id: str,
    baseline: str,
    human: str,
    candidate: str,
    evidence: list[Evidence] | None = None,
) -> MergeResult:
    """Three-way merge Concept frontmatter and Markdown blocks.

    Tool-managed frontmatter always comes from the candidate. Other values merge
    when only one side changed; overlapping human and candidate edits remain human-
    visible and are returned as conflicts with optional evidence.
    """

    base_meta, base_body = parse_markdown(baseline)
    human_meta, human_body = parse_markdown(human)
    candidate_meta, candidate_body = parse_markdown(candidate)
    conflicts: list[Conflict] = []
    merged_meta: dict[str, Any] = {}
    human_changed = human != baseline

    for key in dict.fromkeys([*base_meta, *human_meta, *candidate_meta]):
        base_value = base_meta.get(key, MISSING)
        human_value = human_meta.get(key, MISSING)
        candidate_value = candidate_meta.get(key, MISSING)
        if key in TOOL_MANAGED:
            value = candidate_value
        else:
            value, conflict = _merge_value(base_value, human_value, candidate_value)
            if conflict:
                conflicts.append(
                    Conflict(
                        id=sha256_text(f"{concept_id}\0frontmatter:{key}")[:16],
                        concept_id=concept_id,
                        block_id=f"frontmatter:{key}",
                        baseline=None if base_value is MISSING else repr(base_value),
                        human=None if human_value is MISSING else repr(human_value),
                        candidate=None if candidate_value is MISSING else repr(candidate_value),
                        evidence=evidence or [],
                        reason="Human and agent independently changed the same frontmatter key.",
                    )
                )
        if value is not MISSING:
            merged_meta[key] = value

    base_sections = body_sections(base_body)
    human_sections = body_sections(human_body)
    candidate_sections = body_sections(candidate_body)
    merged_sections: dict[str, str] = {}
    for key in dict.fromkeys([*base_sections, *human_sections, *candidate_sections]):
        base_value = base_sections.get(key, MISSING)
        human_value = human_sections.get(key, MISSING)
        candidate_value = candidate_sections.get(key, MISSING)
        value, conflict = _merge_value(base_value, human_value, candidate_value)
        if conflict:
            conflicts.append(
                Conflict(
                    id=sha256_text(f"{concept_id}\0{key}")[:16],
                    concept_id=concept_id,
                    block_id=key,
                    baseline=None if base_value is MISSING else str(base_value),
                    human=None if human_value is MISSING else str(human_value),
                    candidate=None if candidate_value is MISSING else str(candidate_value),
                    evidence=evidence or [],
                    reason="Human and agent independently changed the same Markdown section.",
                )
            )
            value = human_value
        if value is not MISSING:
            merged_sections[key] = str(value)

    return MergeResult(
        markdown=dump_markdown(merged_meta, join_sections(merged_sections)),
        conflicts=conflicts,
        human_changed=human_changed,
    )
