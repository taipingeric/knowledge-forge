from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from .errors import ValidationFailure
from .models import CONCEPT_TYPES, ConceptDraft, PDFSource
from .sources import sha256_text

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
CLAIM_CITATION = re.compile(r"\[\^([^\]]+)@p(?:p)?\.?([0-9,-]+)\]")
FOOTNOTE_DEFINITION = re.compile(r"^\[\^([^\]]+)@p(?:p)?\.?([0-9,-]+)\]:\s+.+$", re.MULTILINE)


def parse_markdown(raw: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER.match(raw)
    if not match:
        raise ValidationFailure("Concept must start with YAML frontmatter")
    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict):
        raise ValidationFailure("Concept frontmatter must be a mapping")
    return metadata, raw[match.end() :]


def dump_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = yaml.safe_dump(
        metadata, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{body.strip()}\n"


def compact_ranges(pages: list[int]) -> list[str]:
    numbers = sorted(set(pages))
    if not numbers:
        return []
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ranges


def expand_ranges(ranges: list[str]) -> list[int]:
    pages: list[int] = []
    for item in ranges:
        if re.fullmatch(r"[1-9][0-9]*", item):
            pages.append(int(item))
        elif match := re.fullmatch(r"([1-9][0-9]*)-([1-9][0-9]*)", item):
            start, end = map(int, match.groups())
            if end < start:
                raise ValidationFailure(f"Invalid descending page range: {item}")
            pages.extend(range(start, end + 1))
        else:
            raise ValidationFailure(f"Invalid page range: {item}")
    return sorted(set(pages))


def render_concept(
    draft: ConceptDraft,
    sources: dict[str, PDFSource],
    actor: str,
    generated_at: datetime | None = None,
) -> str:
    source_entries: list[dict[str, Any]] = []
    for evidence in sorted(draft.evidence, key=lambda item: item.source_id):
        source = sources.get(evidence.source_id)
        if source is None:
            raise ValidationFailure(f"Unknown source in concept {draft.slug}: {evidence.source_id}")
        if max(evidence.pages) > len(source.pages):
            raise ValidationFailure(f"Page outside source bounds in concept {draft.slug}")
        source_entries.append(
            {
                "id": source.id,
                "resource": source.resource,
                "content_sha256": source.content_sha256,
                "pages": compact_ranges(evidence.pages),
            }
        )
    metadata: dict[str, Any] = {
        "type": draft.type,
        "title": draft.title,
        "description": draft.description,
    }
    if draft.tags:
        metadata["tags"] = sorted(set(draft.tags))
    metadata["generated"] = {
        "by": actor,
        "at": (generated_at or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
    }
    metadata["sources"] = source_entries
    return dump_markdown(metadata, draft.body)


def managed_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata.get(key) for key in ("generated", "sources", "verified") if key in metadata
    }


def managed_fields_hash(raw: str) -> str:
    metadata, _ = parse_markdown(raw)
    return sha256_text(yaml.safe_dump(managed_fields(metadata), sort_keys=True))


def concept_version_hash(raw: str) -> str:
    metadata, body = parse_markdown(raw)
    metadata.pop("verified", None)
    metadata.pop("generated", None)
    return sha256_text(dump_markdown(metadata, body))


def validate_concept(
    raw: str, concept_id: str, source_pages: dict[str, int] | None = None
) -> list[str]:
    errors: list[str] = []
    try:
        metadata, body = parse_markdown(raw)
    except ValidationFailure as exc:
        return [f"{concept_id}: {exc}"]
    if metadata.get("type") not in CONCEPT_TYPES:
        errors.append(f"{concept_id}: unsupported type {metadata.get('type')!r}")
    source_ids: set[str] = set()
    evidence_pages: dict[str, set[int]] = {}
    source_entries = metadata.get("sources", [])
    if not isinstance(source_entries, list):
        errors.append(f"{concept_id}: sources must be a list")
        source_entries = []
    for entry in source_entries:
        if not isinstance(entry, dict):
            errors.append(f"{concept_id}: sources entries must be mappings")
            continue
        missing = {"id", "resource", "content_sha256", "pages"} - entry.keys()
        if missing:
            errors.append(f"{concept_id}: source missing {sorted(missing)}")
            continue
        source_id = str(entry["id"])
        source_ids.add(source_id)
        expected_resource = f"urn:knowledge-forge:pdf:{quote(source_id, safe='')}"
        if entry["resource"] != expected_resource:
            errors.append(f"{concept_id}: invalid logical resource for {source_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry["content_sha256"])):
            errors.append(f"{concept_id}: invalid SHA-256 for {source_id}")
        try:
            pages = expand_ranges(entry["pages"])
            evidence_pages[source_id] = set(pages)
            if (
                source_pages
                and source_id in source_pages
                and max(pages, default=0) > source_pages[source_id]
            ):
                errors.append(f"{concept_id}: page outside source bounds for {source_id}")
        except ValidationFailure as exc:
            errors.append(f"{concept_id}: {exc}")
    citations = CLAIM_CITATION.findall(body)
    definitions = set(FOOTNOTE_DEFINITION.findall(body))
    for source_id, page_spec in citations:
        if source_id not in source_ids:
            errors.append(f"{concept_id}: citation references missing source {source_id}")
        try:
            cited_pages = set(expand_ranges(page_spec.split(",")))
            if source_id in evidence_pages and not cited_pages <= evidence_pages[source_id]:
                errors.append(
                    f"{concept_id}: citation pages are outside Concept evidence for {source_id}"
                )
        except ValidationFailure as exc:
            errors.append(f"{concept_id}: {exc}")
        if (source_id, page_spec) not in definitions:
            errors.append(
                f"{concept_id}: citation {source_id}@p{page_spec} has no footnote definition"
            )
    return errors


def render_index(concepts: dict[str, str]) -> str:
    lines = ["---", 'okf_version: "0.2"', "---", "", "# Knowledge Forge", ""]
    by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for concept_id, raw in concepts.items():
        metadata, _ = parse_markdown(raw)
        by_type.setdefault(str(metadata["type"]), []).append((concept_id, metadata))
    for type_name in CONCEPT_TYPES:
        if type_name not in by_type:
            continue
        lines.extend([f"## {type_name}", ""])
        for concept_id, metadata in sorted(by_type[type_name], key=lambda item: item[0]):
            title = metadata.get("title") or Path(concept_id).stem
            description = metadata.get("description", "")
            lines.append(f"* [{title}]({concept_id}.md) - {description}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
