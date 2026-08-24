from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import quote, unquote

import yaml

from .errors import ValidationFailure
from .models import CONCEPT_TYPES, ConceptDraft, KnowledgeSource
from .sources import sha256_text

FRONTMATTER = re.compile(
    r"\A\ufeff?---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL | re.MULTILINE,
)
FOOTNOTE_DEFINITION = re.compile(r"^\[\^([^\]\s]+)\]:\s+.+$", re.MULTILINE)
PORTABLE_FOOTNOTE_REFERENCE = re.compile(r"\[\^([^\]\s]+)\](?!:)")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ACTOR = re.compile(r"^(?:[^\s:/]+:\S+|\S+/\S+)$")
PORTABLE_STATUSES = {"draft", "stable", "deprecated"}
FENCED_CODE_BLOCK = re.compile(r"^```[^\n]*\n.*?^```[ \t]*$", re.DOTALL | re.MULTILINE)


def parse_markdown(raw: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and return it with the remaining Markdown body."""

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
    """Serialize metadata and body as a canonical Knowledge Forge Markdown document."""

    frontmatter = yaml.safe_dump(
        metadata, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{body.strip()}\n"


def compact_ranges(pages: list[int]) -> list[str]:
    """Convert page numbers into sorted, compact one-based ranges."""

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
    """Expand validated page numbers and inclusive ranges into unique page numbers."""

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


def source_reference_id(source_identity: str, page: int) -> str:
    """Return the stable Concept-local reference ID for one PDF page."""
    return f"{quote(source_identity, safe='/')}#pdf_page:{page}"


def source_reference_identity(reference_id: str) -> str | None:
    """Return the Source Identity encoded in a PDF Source Reference ID."""
    match = re.fullmatch(r"(.+)#pdf_page:([1-9][0-9]*)", reference_id)
    return unquote(match.group(1)) if match else None


def _locator_hash(page: int) -> str:
    """Hash a normalized PDF page locator for provenance integrity checks."""

    locator = json.dumps({"kind": "pdf_page", "page": page}, sort_keys=True, separators=(",", ":"))
    return sha256_text(locator)


def render_concept(
    draft: ConceptDraft,
    sources: dict[str, KnowledgeSource],
    actor: str,
    generated_at: datetime | None = None,
) -> str:
    """Render a Concept draft with generated metadata and PDF Source evidence provenance."""

    source_entries: list[dict[str, Any]] = []
    for evidence in sorted(draft.evidence, key=lambda item: item.source_id):
        source = sources.get(evidence.source_id)
        if source is None:
            raise ValidationFailure(f"Unknown source in concept {draft.slug}: {evidence.source_id}")
        if max(evidence.pages) > len(source.evidence):
            raise ValidationFailure(f"Page outside source bounds in concept {draft.slug}")
        for page in evidence.pages:
            source_entries.append(
                {
                    "id": source_reference_id(source.source_identity, page),
                    "resource": source.resource,
                    "content_sha256": source.content_sha256,
                    "locator": {"kind": "pdf_page", "page": page},
                    "locator_sha256": _locator_hash(page),
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
    """Return frontmatter fields managed by Knowledge Forge rather than people."""

    return {
        key: metadata.get(key) for key in ("generated", "sources", "verified") if key in metadata
    }


def managed_fields_hash(raw: str) -> str:
    """Hash a Concept's tool-managed provenance fields for tamper detection."""

    metadata, _ = parse_markdown(raw)
    return sha256_text(yaml.safe_dump(managed_fields(metadata), sort_keys=True))


def concept_version_hash(raw: str) -> str:
    """Hash semantic Concept content while ignoring generated and verification metadata."""

    metadata, body = parse_markdown(raw)
    metadata.pop("verified", None)
    metadata.pop("generated", None)
    return sha256_text(dump_markdown(metadata, body))


def _non_empty_string(value: object) -> TypeGuard[str]:
    """Return whether a value is a non-blank string suitable for OKF metadata."""

    return isinstance(value, str) and bool(value.strip())


def _date_value(value: object) -> date | None:
    """Parse a date-only OKF value, rejecting datetimes and malformed strings."""

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
    """Parse an ISO 8601 datetime and require timezone information."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and "T" in value.upper():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _date_or_datetime_value(value: object) -> date | datetime | None:
    """Parse either an OKF date or a timezone-aware ISO 8601 datetime."""

    return _date_value(value) or _datetime_value(value)


def _temporal_key(value: date | datetime) -> datetime:
    """Normalize an OKF date or datetime for chronological comparison."""

    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC)


def _validate_actor(value: object, field: str) -> list[str]:
    """Validate an actor identifier and return field-specific errors."""

    if not _non_empty_string(value):
        return [f"{field} must be a non-empty actor string"]
    if not ACTOR.fullmatch(value):
        return [
            f"{field} must use human:<id>, process:<id>, <producer>/<version>, "
            "or another <kind>:<id> actor"
        ]
    return []


def _validate_usage_window(value: object, field: str) -> list[str]:
    """Validate an inclusive date or datetime interval in OKF metadata."""

    if not isinstance(value, dict):
        return [f"{field} must be a mapping with from and to dates or datetimes"]
    errors: list[str] = []
    start = _date_or_datetime_value(value.get("from"))
    end = _date_or_datetime_value(value.get("to"))
    if start is None:
        errors.append(f"{field}.from must be a YYYY-MM-DD date or ISO 8601 datetime")
    if end is None:
        errors.append(f"{field}.to must be a YYYY-MM-DD date or ISO 8601 datetime")
    if start is not None and end is not None and _temporal_key(start) > _temporal_key(end):
        errors.append(f"{field}.from must not be after {field}.to")
    return errors


def _validate_portable_sources(metadata: dict[str, Any], body: str) -> list[str]:
    """Validate portable source metadata and ensure citations name declared sources."""

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
            if (
                "last_modified" in source
                and _date_or_datetime_value(source["last_modified"]) is None
            ):
                errors.append(
                    f"{field}.last_modified must be a YYYY-MM-DD date or ISO 8601 datetime"
                )

    for citation_id in sorted(set(PORTABLE_FOOTNOTE_REFERENCE.findall(body))):
        if citation_id not in source_ids:
            errors.append(f"citation [^{citation_id}] does not match any sources[].id")
    return errors


def _validate_portable_trust(metadata: dict[str, Any]) -> list[str]:
    """Validate portable generated and verified attribution metadata."""

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
    """Validate metadata required to describe an OKF Attested Computation."""

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


def _validate_computation_representation(metadata: dict[str, Any], body: str) -> list[str]:
    """Ensure an Attested Computation uses exactly one computation representation."""

    if metadata.get("type") != "Attested Computation":
        return []
    heading = re.search(r"^# Computation[ \t]*$", body, re.MULTILINE)
    section = body[heading.end() :] if heading else ""
    next_heading = re.search(r"^# [^#].*$", section, re.MULTILINE)
    if next_heading:
        section = section[: next_heading.start()]
    inline_count = len(FENCED_CODE_BLOCK.findall(section))
    has_file = _non_empty_string(metadata.get("computation"))
    if has_file and inline_count:
        return ["must provide computation in a file or an inline fence, not both"]
    if not has_file and inline_count != 1:
        return ["must provide either computation or one inline fence under # Computation"]
    return []


def validate_portable_concept(raw: str, concept_id: str) -> list[str]:
    """Validate an OKF v0.2 Concept without applying a producer profile.

    Return all validation messages prefixed with the supplied Concept ID instead of
    raising, so callers can report errors across an entire Bundle at once.
    """
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
    if "stale_after" in metadata and _date_or_datetime_value(metadata["stale_after"]) is None:
        errors.append("stale_after must be a YYYY-MM-DD date or ISO 8601 datetime")
    errors.extend(_validate_portable_trust(metadata))
    errors.extend(_validate_portable_sources(metadata, body))
    errors.extend(_validate_attested_computation(metadata))
    errors.extend(_validate_computation_representation(metadata, body))
    return [f"{concept_id}: {error}" for error in errors]


def validate_concept(
    raw: str, concept_id: str, source_pages: dict[str, int] | None = None
) -> list[str]:
    """Validate a managed Concept's type, provenance, evidence, and citations."""

    errors: list[str] = []
    try:
        metadata, body = parse_markdown(raw)
    except ValidationFailure as exc:
        return [f"{concept_id}: {exc}"]
    if metadata.get("type") not in CONCEPT_TYPES:
        errors.append(f"{concept_id}: unsupported type {metadata.get('type')!r}")
    source_references: dict[str, tuple[str, int]] = {}
    source_entries = metadata.get("sources", [])
    if not isinstance(source_entries, list):
        errors.append(f"{concept_id}: sources must be a list")
        source_entries = []
    for entry in source_entries:
        if not isinstance(entry, dict):
            errors.append(f"{concept_id}: sources entries must be mappings")
            continue
        missing = {"id", "resource", "content_sha256", "locator", "locator_sha256"} - entry.keys()
        if missing:
            errors.append(f"{concept_id}: source missing {sorted(missing)}")
            continue
        reference_id = entry["id"]
        locator = entry["locator"]
        if not _non_empty_string(reference_id):
            errors.append(f"{concept_id}: source reference ID must be a non-empty string")
            continue
        if (
            not isinstance(locator, dict)
            or locator.get("kind") != "pdf_page"
            or not isinstance(locator.get("page"), int)
            or locator["page"] < 1
        ):
            errors.append(
                f"{concept_id}: source reference {reference_id} has an invalid PDF page locator"
            )
            continue
        page = locator["page"]
        source_id = source_reference_identity(reference_id)
        if source_id is None:
            errors.append(f"{concept_id}: invalid source reference ID {reference_id}")
            continue
        if reference_id != source_reference_id(source_id, page):
            errors.append(
                f"{concept_id}: source reference ID does not match its locator for {reference_id}"
            )
        if reference_id in source_references:
            errors.append(f"{concept_id}: duplicate source reference {reference_id}")
            continue
        source_references[reference_id] = (source_id, page)
        expected_resource = f"urn:knowledge-forge:pdf:{quote(source_id, safe='')}"
        if entry["resource"] != expected_resource:
            errors.append(f"{concept_id}: invalid logical resource for {reference_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry["content_sha256"])):
            errors.append(f"{concept_id}: invalid document SHA-256 for {reference_id}")
        if entry["locator_sha256"] != _locator_hash(page):
            errors.append(f"{concept_id}: invalid locator SHA-256 for {reference_id}")
        if source_pages is not None:
            if source_id not in source_pages:
                errors.append(f"{concept_id}: source reference is outside referenced evidence")
            elif page > source_pages[source_id]:
                errors.append(f"{concept_id}: page outside source bounds for {reference_id}")
    citations = list(dict.fromkeys(PORTABLE_FOOTNOTE_REFERENCE.findall(body)))
    definitions = set(FOOTNOTE_DEFINITION.findall(body))
    for reference_id in citations:
        if reference_id not in source_references:
            errors.append(
                f"{concept_id}: citation references missing source reference {reference_id}"
            )
        if reference_id not in definitions:
            errors.append(f"{concept_id}: citation {reference_id} has no footnote definition")
    return errors


def render_index(concepts: dict[str, str]) -> str:
    """Render the deterministic root index for a mapping of Concept documents."""

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
