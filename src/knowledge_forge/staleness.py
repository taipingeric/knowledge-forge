"""Read-only detection and reporting of changed referenced PDF evidence."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ForgeState, KnowledgeSource
from .okf import parse_markdown, source_reference_identity
from .sources import sha256_text
from .state import public_concepts


def detect_staleness(
    bundle: Path, state: ForgeState, sources: list[KnowledgeSource]
) -> list[dict[str, object]]:
    """Return definite changed or missing evidence references for agent-owned Concepts."""
    current = {source.id: source for source in sources}
    stale: list[dict[str, object]] = []
    for concept_id, raw in public_concepts(bundle).items():
        owner = state.concepts.get(concept_id)
        if owner is None or owner.ownership != "agent":
            continue
        metadata, _ = parse_markdown(raw)
        for entry in metadata.get("sources", []):
            if not isinstance(entry, dict):
                continue
            reference_id = entry.get("id")
            source_id = source_reference_identity(str(reference_id))
            locator = entry.get("locator")
            if (
                source_id is None
                or not isinstance(locator, dict)
                or not isinstance(locator.get("page"), int)
            ):
                continue
            source = current.get(source_id)
            page = locator["page"]
            previous = str(entry.get("evidence_sha256", entry.get("content_sha256", "")))
            if source is None or page > len(source.evidence):
                stale.append(
                    {
                        "concept_id": concept_id,
                        "reference_id": reference_id,
                        "source_id": source_id,
                        "locator": locator,
                        "previous_hash": previous,
                        "current_hash": None,
                    }
                )
            else:
                current_hash = sha256_text(source.evidence[page - 1].text)
                if current_hash != previous:
                    stale.append(
                        {
                            "concept_id": concept_id,
                            "reference_id": reference_id,
                            "source_id": source_id,
                            "locator": locator,
                            "previous_hash": previous,
                            "current_hash": current_hash,
                        }
                    )
    return stale


def write_staleness_report(
    output: Path,
    stale_concepts: list[dict[str, object]],
    *,
    planning_stale: bool,
    generation_stale: list[str],
) -> Path:
    """Write the external human and machine-readable Regeneration Impact Report."""
    work = output.parent / f"{output.name}.staleness"
    work.mkdir(exist_ok=True)
    manifest = {
        "stale_concepts": stale_concepts,
        "planning_stale": planning_stale,
        "generation_stale": generation_stale,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = output.parent / f"{output.name}.staleness.md"
    lines = ["# Knowledge Forge Staleness", ""]
    for item in stale_concepts:
        lines.append(
            f"- `{item['concept_id']}` / `{item['reference_id']}`: `{item['source_id']}` "
            f"{item['locator']}; previous `{item['previous_hash']}`, "
            f"current `{item['current_hash'] or 'missing'}`"
        )
    if planning_stale:
        lines.append("- Planning coverage is stale because the authoritative source set changed.")
    for concept_id in generation_stale:
        lines.append(f"- `{concept_id}`: generation identity changed.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
