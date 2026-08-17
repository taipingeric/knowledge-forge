from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import quote

import yaml

from .errors import ValidationFailure
from .models import CONCEPT_TYPES, ConceptDraft, PDFSource
from .sources import sha256_text

FRONTMATTER = re.compile(
    r"\A\ufeff?---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL | re.MULTILINE,
)
CLAIM_CITATION = re.compile(r"\[\^([^\]]+)@p(?:p)?\.?([0-9,-]+)\]")
FOOTNOTE_DEFINITION = re.compile(r"^\[\^([^\]]+)@p(?:p)?\.?([0-9,-]+)\]:\s+.+$", re.MULTILINE)
PORTABLE_FOOTNOTE_REFERENCE = re.compile(r"\[\^([^\]\s]+)\](?!:)")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ACTOR = re.compile(r"^(?:[^\s:/]+:\S+|\S+/\S+)$")
PORTABLE_STATUSES = {"draft", "stable", "deprecated"}


def parse_markdown(raw: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER.match(raw)
    if not match:
        raise ValidationFailure("Concept must start with YAML frontmatter")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValidationFailure(f"Concept frontmatter is not valid YAML: {exc}") from exc
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


def _non_empty_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def _date_value(value: object) -> date | None:
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _datetime_value(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or "T" not in value.upper():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return None


def _validate_actor(value: object, field: str) -> list[str]:
    if not _non_empty_string(value):
        return [f"{field} must be a non-empty actor string"]
    if not ACTOR.fullmatch(value):
        return [
            f"{field} must use human:<id>, process:<id>, <producer>/<version>, "
            "or another <kind>:<id> actor"
        ]
    return []


def _validate_usage_window(value: object, field: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{field} must be a mapping with from and to dates"]
    errors: list[str] = []
    start = _date_value(value.get("from"))
    end = _date_value(value.get("to"))
    if start is None:
        errors.append(f"{field}.from must be a YYYY-MM-DD date")
    if end is None:
        errors.append(f"{field}.to must be a YYYY-MM-DD date")
    if start is not None and end is not None and start > end:
        errors.append(f"{field}.from must not be after {field}.to")
    return errors


def _validate_portable_sources(metadata: dict[str, Any], body: str) -> list[str]:
    errors: list[str] = []
    shared_window = metadata.get("usage_window")
    if "usage_window" in metadata:
        errors.extend(_validate_usage_window(shared_window, "usage_window"))

    sources = metadata.get("sources")
    if sources is None:
        source_ids: set[str] = set()
    elif not isinstance(sources, list):
        errors.append("sources must be a list")
        source_ids = set()
    else:
        source_ids = set()
        for index, source in enumerate(sources):
            field = f"sources[{index}]"
            if not isinstance(source, dict):
                errors.append(f"{field} must be a mapping")
                continue
            if not _non_empty_string(source.get("resource")):
                errors.append(f"{field}.resource must be a non-empty string")
            source_id = source.get("id")
            if source_id is not None:
                if not _non_empty_string(source_id):
                    errors.append(f"{field}.id must be a non-empty string")
                elif source_id in source_ids:
                    errors.append(f"{field}.id duplicates {source_id!r}")
                else:
                    source_ids.add(source_id)
            for key in ("title",):
                if key in source and not _non_empty_string(source[key]):
                    errors.append(f"{field}.{key} must be a non-empty string")
            if "author" in source:
                errors.extend(_validate_actor(source["author"], f"{field}.author"))
            usage_count = source.get("usage_count")
            if usage_count is not None:
                if (
                    isinstance(usage_count, bool)
                    or not isinstance(usage_count, int)
                    or usage_count < 0
                ):
                    errors.append(f"{field}.usage_count must be a non-negative integer")
                if "usage_window" not in source and shared_window is None:
                    errors.append(f"{field}.usage_count requires a usage_window")
            if "usage_window" in source:
                errors.extend(
                    _validate_usage_window(source["usage_window"], f"{field}.usage_window")
                )
            if "last_modified" in source and _date_value(source["last_modified"]) is None:
                errors.append(f"{field}.last_modified must be a YYYY-MM-DD date")

    for citation_id in sorted(set(PORTABLE_FOOTNOTE_REFERENCE.findall(body))):
        if citation_id not in source_ids:
            errors.append(f"citation [^{citation_id}] does not match any sources[].id")
    return errors


def _validate_portable_trust(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    generated = metadata.get("generated")
    if generated is not None:
        if not isinstance(generated, dict):
            errors.append("generated must be a mapping")
        else:
            errors.extend(_validate_actor(generated.get("by"), "generated.by"))
            if "at" in generated and _datetime_value(generated["at"]) is None:
                errors.append("generated.at must be an ISO 8601 datetime")

    verified = metadata.get("verified")
    if verified is not None:
        entries = [verified] if isinstance(verified, dict) else verified
        if not isinstance(entries, list) or not entries:
            errors.append("verified must be a mapping or non-empty list of mappings")
        else:
            for index, entry in enumerate(entries):
                field = f"verified[{index}]"
                if not isinstance(entry, dict):
                    errors.append(f"{field} must be a mapping")
                    continue
                errors.extend(_validate_actor(entry.get("by"), f"{field}.by"))
                if _datetime_value(entry.get("at")) is None:
                    errors.append(f"{field}.at must be an ISO 8601 datetime")
    return errors


def _validate_attested_computation(metadata: dict[str, Any]) -> list[str]:
    if metadata.get("type") != "Attested Computation":
        return []
    errors: list[str] = []
    if not _non_empty_string(metadata.get("runtime")):
        errors.append("runtime is required for an Attested Computation")
    parameters = metadata.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, list):
            errors.append("parameters must be a list")
        else:
            names: set[str] = set()
            for index, parameter in enumerate(parameters):
                field = f"parameters[{index}]"
                if not isinstance(parameter, dict):
                    errors.append(f"{field} must be a mapping")
                    continue
                name = parameter.get("name")
                if not _non_empty_string(name):
                    errors.append(f"{field}.name must be a non-empty string")
                elif name in names:
                    errors.append(f"{field}.name duplicates {name!r}")
                else:
                    names.add(name)
                if not _non_empty_string(parameter.get("type")):
                    errors.append(f"{field}.type must be a non-empty string")
                if not isinstance(parameter.get("required"), bool):
                    errors.append(f"{field}.required must be a boolean")
    if "computation" in metadata and not _non_empty_string(metadata["computation"]):
        errors.append("computation must be a non-empty path or URI")
    executor = metadata.get("executor")
    if executor is not None:
        if not isinstance(executor, dict):
            errors.append("executor must be a mapping")
        else:
            if not _non_empty_string(executor.get("resource")):
                errors.append("executor.resource must be a non-empty path or URI")
            receipt = executor.get("receipt")
            if not (
                isinstance(receipt, list)
                and receipt
                and all(_non_empty_string(item) for item in receipt)
            ):
                errors.append("executor.receipt must be a non-empty list of field names")
    attester = metadata.get("attester")
    if attester is not None:
        if not isinstance(attester, dict):
            errors.append("attester must be a mapping")
        elif not _non_empty_string(attester.get("resource")):
            errors.append("attester.resource must be a non-empty path or URI")
    return errors


def validate_portable_concept(raw: str, concept_id: str) -> list[str]:
    """Validate an OKF v0.2 Concept without applying a producer profile."""
    try:
        metadata, body = parse_markdown(raw)
    except ValidationFailure as exc:
        return [f"{concept_id}: {exc}"]
    errors: list[str] = []
    if not _non_empty_string(metadata.get("type")):
        errors.append("missing or empty required type")
    for key in ("title", "description", "resource"):
        if key in metadata and not _non_empty_string(metadata[key]):
            errors.append(f"{key} must be a non-empty string")
    if "tags" in metadata and not (
        isinstance(metadata["tags"], list)
        and all(_non_empty_string(tag) for tag in metadata["tags"])
    ):
        errors.append("tags must be a list of non-empty strings")
    if "status" in metadata and metadata["status"] not in PORTABLE_STATUSES:
        errors.append("status must be one of draft, stable, or deprecated")
    if "stale_after" in metadata and _date_value(metadata["stale_after"]) is None:
        errors.append("stale_after must be a YYYY-MM-DD date")
    errors.extend(_validate_portable_trust(metadata))
    errors.extend(_validate_portable_sources(metadata, body))
    errors.extend(_validate_attested_computation(metadata))
    return [f"{concept_id}: {error}" for error in errors]


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
    citations = list(dict.fromkeys(CLAIM_CITATION.findall(body)))
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
