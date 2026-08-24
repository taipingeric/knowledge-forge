"""Atomic migration support for legacy PDF-only managed Bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import ValidationFailure
from .okf import (
    dump_markdown,
    expand_ranges,
    managed_fields_hash,
    parse_markdown,
    source_reference_id,
)
from .sources import sha256_text
from .state import (
    bundle_hash,
    load_baseline,
    load_state,
    public_concepts,
    write_baseline,
    write_state,
)

LEGACY_LABEL = re.compile(r"\[\^(.+)@p(?:p)?\.?([0-9,-]+)\]")
LEGACY_DEFINITION = re.compile(r"^\[\^(.+)@p(?:p)?\.?([0-9,-]+)\]:\s*(.+)$")


def migration_available(bundle: Path) -> bool:
    """Return whether a Bundle contains legacy PDF citation provenance."""
    material = json.loads((bundle / ".knowledge-forge" / "state.json").read_text(encoding="utf-8"))
    if material.get("state_version") != 3:
        return True
    return any(LEGACY_LABEL.search(raw) for raw in public_concepts(bundle).values())


def migrate_bundle(bundle: Path) -> None:
    """Rewrite supported legacy PDF provenance in an already-staged Bundle."""
    state = load_state(bundle)
    concepts = public_concepts(bundle)
    migrated: dict[str, str] = {}
    for concept_id, raw in concepts.items():
        migrated[concept_id] = _migrate_concept(raw)
        path = bundle / f"{concept_id}.md"
        path.write_text(migrated[concept_id], encoding="utf-8")
        if concept_id in state.concepts and state.concepts[concept_id].ownership == "agent":
            baseline = load_baseline(bundle, concept_id)
            write_baseline(bundle, concept_id, _migrate_concept(baseline.raw_markdown))
            state.concepts[concept_id].baseline_hash = sha256_text(
                _migrate_concept(baseline.raw_markdown)
            )
        if concept_id in state.concepts:
            state.concepts[concept_id].managed_fields_hash = managed_fields_hash(
                migrated[concept_id]
            )
    state.bundle_hash = bundle_hash(bundle)
    for name in ("index.md", "log.md"):
        path = bundle / name
        if path.is_file():
            state.tool_files[name] = sha256_text(path.read_text(encoding="utf-8"))
    write_state(bundle, state)


def _migrate_concept(raw: str) -> str:
    metadata, body = parse_markdown(raw)
    entries = metadata.get("sources", [])
    if not isinstance(entries, list):
        raise ValidationFailure("Legacy Concept sources must be a list")
    source_entries: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "pages" not in entry:
            source_entries.append(entry)
            continue
        source_id = entry.get("id")
        if not isinstance(source_id, str):
            raise ValidationFailure("Legacy source entry id must be a string")
        pages = expand_ranges(entry["pages"])
        for page in pages:
            locator = {"kind": "pdf_page", "page": page}
            source_entries.append(
                {
                    "id": source_reference_id(source_id, page),
                    "resource": entry.get("resource"),
                    "content_sha256": entry.get("content_sha256"),
                    "locator": locator,
                    "locator_sha256": sha256_text('{"kind":"pdf_page","page":' + str(page) + "}"),
                }
            )
    metadata["sources"] = source_entries
    rewritten: list[str] = []
    for line in body.splitlines():
        if match := LEGACY_DEFINITION.fullmatch(line):
            source_id, page_spec, text = match.groups()
            rewritten.extend(
                f"[^{source_reference_id(source_id, page)}]: {text}"
                for page in expand_ranges(page_spec.split(","))
            )
        else:
            rewritten.append(
                LEGACY_LABEL.sub(
                    lambda match: "".join(
                        f"[^{source_reference_id(match.group(1), page)}]"
                        for page in expand_ranges(match.group(2).split(","))
                    ),
                    line,
                )
            )
    return dump_markdown(metadata, "\n".join(rewritten))
