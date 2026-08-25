"""Read-only detection and reporting of changed referenced typed evidence."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import TypeAdapter

from .errors import ValidationFailure
from .models import EvidenceLocator, ForgeState, GenerationIdentity, KnowledgeSource
from .okf import parse_markdown, source_reference_identity
from .sources import sha256_text
from .state import public_concepts

evidence_locator_adapter = TypeAdapter(EvidenceLocator)


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
            if source_id is None or not isinstance(locator, dict):
                continue
            source = current.get(source_id)
            previous = str(entry.get("evidence_sha256", entry.get("content_sha256", "")))
            try:
                typed_locator = evidence_locator_adapter.validate_python(locator)
            except (TypeError, ValueError):
                continue
            evidence = (
                next((unit for unit in source.evidence if unit.locator == typed_locator), None)
                if source is not None
                else None
            )
            if evidence is None:
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
                current_hash = sha256_text(evidence.text)
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
    live_bundle_hash: str,
    source_set_hash: str,
    generation: GenerationIdentity,
) -> Path:
    """Write the external human and machine-readable Regeneration Impact Report."""
    work = output.parent / f"{output.name}.staleness"
    work.mkdir(exist_ok=True)
    manifest = {
        "status": "pending",
        "live_bundle_hash": live_bundle_hash,
        "source_set_hash": source_set_hash,
        "generation": generation.model_dump(mode="json"),
        "stale_concepts": stale_concepts,
        "planning_stale": planning_stale,
        "generation_stale": generation_stale,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = output.parent / f"{output.name}.staleness.md"
    lines = ["# Knowledge Forge Staleness", "", "Status: pending", ""]
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


def load_pending_staleness_report(output: Path) -> dict[str, object]:
    """Load the pending regeneration authorization record for an output Bundle."""

    manifest_path = output.parent / f"{output.name}.staleness" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValidationFailure(
            f"No valid pending Regeneration Impact Report: {manifest_path}"
        ) from exc
    required = {"status", "live_bundle_hash", "source_set_hash", "generation"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValidationFailure(f"Invalid pending Regeneration Impact Report: {manifest_path}")
    if manifest["status"] != "pending":
        raise ValidationFailure(f"Regeneration Impact Report is not pending: {manifest_path}")
    return manifest


def _resolved_staleness_audit(manifest: dict[str, object]) -> str:
    """Render the durable audit record produced by a successful regeneration."""

    generation = json.dumps(manifest["generation"], indent=2, sort_keys=True)
    impact = json.dumps(
        {
            "stale_concepts": manifest.get("stale_concepts", []),
            "planning_stale": manifest.get("planning_stale", False),
            "generation_stale": manifest.get("generation_stale", []),
        },
        indent=2,
        sort_keys=True,
    )
    return (
        "# Knowledge Forge Staleness\n\nStatus: resolved\n\n"
        "This Regeneration Impact Report authorized and recorded a successful full "
        "regeneration.\n\n"
        f"- Authorized live Bundle: `{manifest['live_bundle_hash']}`\n"
        f"- Authorized source set: `{manifest['source_set_hash']}`\n\n"
        "## Requested Generation Identity\n\n```json\n"
        f"{generation}\n```\n\n## Recorded impact\n\n```json\n{impact}\n```\n"
    )


@contextmanager
def prepared_staleness_resolution(output: Path, manifest: dict[str, object]) -> Iterator[None]:
    """Resolve a pending audit with rollback if Bundle publication does not complete."""

    report = output.parent / f"{output.name}.staleness.md"
    work = output.parent / f"{output.name}.staleness"
    archived = output.parent / f".{output.name}.staleness.resolving"
    temporary_report = output.parent / f".{output.name}.staleness.resolved.md"
    pending_report = report.read_bytes()
    temporary_report.write_text(_resolved_staleness_audit(manifest), encoding="utf-8")
    work.rename(archived)
    temporary_report.replace(report)
    try:
        yield
    except Exception:
        report.write_bytes(pending_report)
        archived.rename(work)
        raise
    else:
        shutil.rmtree(archived, ignore_errors=True)
