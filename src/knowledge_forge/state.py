from __future__ import annotations

import json
from pathlib import Path

from .errors import ValidationFailure
from .models import (
    GENERATION_POLICY_VERSION,
    LEGACY_STATE_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    BaselineSnapshot,
    ForgeState,
)
from .sources import sha256_text

PRIVATE_DIR = ".knowledge-forge"


def concept_path(bundle: Path, concept_id: str) -> Path:
    return bundle / f"{concept_id}.md"


def state_path(bundle: Path) -> Path:
    return bundle / PRIVATE_DIR / "state.json"


def baseline_path(bundle: Path, concept_id: str) -> Path:
    return bundle / PRIVATE_DIR / "baseline" / f"{concept_id}.json"


def load_state(bundle: Path) -> ForgeState:
    path = state_path(bundle)
    if not path.is_file():
        raise ValidationFailure(f"Knowledge Forge state is missing: {path}")
    try:
        material = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationFailure(f"Invalid Knowledge Forge state: {path}: {exc}") from exc
    if not isinstance(material, dict):
        raise ValidationFailure(f"Invalid Knowledge Forge state: {path}: expected an object")
    if not _has_valid_raw_integrity_hash(material):
        raise ValidationFailure(f"Knowledge Forge state integrity check failed: {path}")
    try:
        state = ForgeState.model_validate(_migrate_state(material))
    except ValidationFailure:
        raise
    except Exception as exc:
        raise ValidationFailure(f"Invalid Knowledge Forge state: {path}: {exc}") from exc
    state.integrity_hash = state_integrity_hash(state)
    return state


def _has_valid_raw_integrity_hash(material: dict[str, object]) -> bool:
    integrity_hash = material.get("integrity_hash")
    if not isinstance(integrity_hash, str):
        return False
    unsigned = {key: value for key, value in material.items() if key != "integrity_hash"}
    return integrity_hash == canonical_hash(unsigned)


def _migrate_state(material: dict[str, object]) -> dict[str, object]:
    """Convert a supported managed-state schema into the current representation."""
    version = material.get("state_version")
    if type(version) is not int:
        raise ValidationFailure(
            "Unsupported State Schema Version "
            f"{version!r}; supported versions are {LEGACY_STATE_SCHEMA_VERSION} through "
            f"{STATE_SCHEMA_VERSION}."
        )
    if version < LEGACY_STATE_SCHEMA_VERSION or version > STATE_SCHEMA_VERSION:
        raise ValidationFailure(
            "Unsupported State Schema Version "
            f"{version!r}; supported versions are {LEGACY_STATE_SCHEMA_VERSION} through "
            f"{STATE_SCHEMA_VERSION}."
        )
    migrated = dict(material)
    while migrated["state_version"] < STATE_SCHEMA_VERSION:
        if migrated["state_version"] == 1:
            legacy_generation = migrated.get("generation")
            if not isinstance(legacy_generation, dict):
                raise ValidationFailure("Invalid schema v1 state: generation must be an object")
            generation = dict(legacy_generation)
            legacy_workflow_version = generation.pop("workflow_version", None)
            if not isinstance(legacy_workflow_version, str):
                raise ValidationFailure(
                    "Invalid schema v1 state: generation.workflow_version is required"
                )
            generation["generation_policy_version"] = GENERATION_POLICY_VERSION
            migrated["generation"] = generation
            migrated["workflow_version"] = legacy_workflow_version
            migrated["state_version"] = 2
        elif migrated["state_version"] == 2:
            migrated["state_version"] = 3
    return migrated


def write_state(bundle: Path, state: ForgeState) -> None:
    path = state_path(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.integrity_hash = state_integrity_hash(state)
    path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


def state_integrity_hash(state: ForgeState) -> str:
    material = state.model_dump(mode="json", exclude={"integrity_hash"})
    return sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))


def load_baseline(bundle: Path, concept_id: str) -> BaselineSnapshot:
    path = baseline_path(bundle, concept_id)
    if not path.is_file():
        raise ValidationFailure(f"Agent baseline is missing: {path}")
    try:
        snapshot = BaselineSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationFailure(f"Invalid baseline snapshot: {path}: {exc}") from exc
    if snapshot.concept_id != concept_id or sha256_text(snapshot.raw_markdown) != snapshot.sha256:
        raise ValidationFailure(f"Agent baseline integrity check failed: {path}")
    return snapshot


def write_baseline(bundle: Path, concept_id: str, raw_markdown: str) -> str:
    digest = sha256_text(raw_markdown)
    snapshot = BaselineSnapshot(concept_id=concept_id, raw_markdown=raw_markdown, sha256=digest)
    path = baseline_path(bundle, concept_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return digest


def bundle_hash(bundle: Path, *, include_state: bool = False) -> str:
    material: list[str] = []
    if not bundle.exists():
        return sha256_text("")
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        if not include_state and relative.startswith(f"{PRIVATE_DIR}/"):
            continue
        material.append(f"{relative}\0{sha256_text(path.read_text(encoding='utf-8'))}")
    return sha256_text("\n".join(material))


def public_concepts(bundle: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    concepts = bundle / "concepts"
    if not concepts.exists():
        return result
    for path in sorted(concepts.rglob("*.md")):
        concept_id = path.relative_to(bundle).with_suffix("").as_posix()
        result[concept_id] = path.read_text(encoding="utf-8")
    return result


def canonical_hash(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))
