from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_forge import application
from knowledge_forge.errors import ValidationFailure
from knowledge_forge.models import (
    GENERATION_POLICY_VERSION,
    STATE_SCHEMA_VERSION,
    WORKFLOW_VERSION,
    ForgeState,
    GenerationIdentity,
)
from knowledge_forge.sources import sha256_text
from knowledge_forge.state import load_state, state_integrity_hash, state_path, write_state


def sample_state() -> ForgeState:
    return ForgeState(
        generation=GenerationIdentity(
            model="fake-model",
            endpoint="https://models.example/v1",
            language="auto",
            max_agent_steps=50,
        ),
        source_set_hash="sources",
        sources={},
        bundle_hash="bundle",
        tool_files={},
        concepts={},
    )


def raw_integrity_hash(material: dict[str, object]) -> str:
    return sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))


def test_state_serializes_the_three_distinct_versions_deterministically(tmp_path: Path) -> None:
    state = sample_state()
    write_state(tmp_path, state)
    first = state_path(tmp_path).read_text()
    write_state(tmp_path, state)

    serialized = json.loads(first)
    assert state_path(tmp_path).read_text() == first
    assert serialized["state_version"] == STATE_SCHEMA_VERSION
    assert serialized["workflow_version"] == WORKFLOW_VERSION
    assert serialized["generation"]["generation_policy_version"] == GENERATION_POLICY_VERSION
    assert "workflow_version" not in serialized["generation"]
    assert load_state(tmp_path).integrity_hash == state_integrity_hash(load_state(tmp_path))


def test_load_state_explicitly_migrates_schema_v1_pdf_state(tmp_path: Path) -> None:
    state = sample_state()
    write_state(tmp_path, state)
    legacy = json.loads(state_path(tmp_path).read_text())
    legacy["state_version"] = 1
    legacy["generation"]["workflow_version"] = "legacy-workflow"
    del legacy["generation"]["generation_policy_version"]
    legacy["integrity_hash"] = ""
    legacy["integrity_hash"] = raw_integrity_hash(
        {key: value for key, value in legacy.items() if key != "integrity_hash"}
    )
    state_path(tmp_path).write_text(json.dumps(legacy) + "\n")

    migrated = load_state(tmp_path)

    assert migrated.state_version == STATE_SCHEMA_VERSION
    assert migrated.workflow_version == "legacy-workflow"
    assert migrated.generation.generation_policy_version == GENERATION_POLICY_VERSION
    assert migrated.integrity_hash == state_integrity_hash(migrated)


@pytest.mark.parametrize("version", [0, STATE_SCHEMA_VERSION + 1, "unknown", True])
def test_load_state_rejects_unsupported_schema_versions(tmp_path: Path, version: object) -> None:
    state = sample_state()
    write_state(tmp_path, state)
    material = json.loads(state_path(tmp_path).read_text())
    material["state_version"] = version
    material["integrity_hash"] = raw_integrity_hash(
        {key: value for key, value in material.items() if key != "integrity_hash"}
    )
    state_path(tmp_path).write_text(json.dumps(material) + "\n")

    with pytest.raises(ValidationFailure, match="Unsupported State Schema Version"):
        load_state(tmp_path)


def test_generation_policy_version_changes_generation_identity() -> None:
    previous = sample_state().generation
    changed = previous.model_copy(update={"generation_policy_version": "next-generation-policy"})

    assert not application._same_generation_request(previous, changed)
