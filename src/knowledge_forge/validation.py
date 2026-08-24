from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .errors import ValidationFailure
from .models import ForgeState, KnowledgeSource
from .okf import (
    managed_fields_hash,
    parse_markdown,
    render_index,
    validate_concept,
    validate_portable_concept,
)
from .sources import sha256_text, source_set_hash
from .state import bundle_hash, load_baseline, load_state, public_concepts


def _file_hash(path: Path) -> str:
    """Hash a UTF-8 tool-managed file for consistency checks."""

    return sha256_text(path.read_text(encoding="utf-8"))


def _read_portable_markdown(path: Path, relative: str) -> tuple[str | None, list[str]]:
    """Read UTF-8 Markdown and return a portable validation error instead of raising."""

    try:
        return path.read_text(encoding="utf-8"), []
    except (OSError, UnicodeError) as exc:
        return None, [f"{relative}: cannot read UTF-8 Markdown: {exc}"]


def _validate_root_index(path: Path) -> list[str]:
    """Validate an optional root index's restricted OKF version frontmatter."""

    if not path.is_file():
        return []
    raw, errors = _read_portable_markdown(path, "index.md")
    if raw is None:
        return errors
    if not raw.lstrip("\ufeff").startswith("---"):
        return []
    try:
        metadata, _ = parse_markdown(raw)
    except ValidationFailure as exc:
        return [f"index.md: invalid root index frontmatter: {exc}"]
    if set(metadata) != {"okf_version"}:
        return ["index.md: root index frontmatter may only declare okf_version"]
    version = metadata["okf_version"]
    if not isinstance(version, str):
        return ['index.md: okf_version must be the string "0.2"']
    if version != "0.2":
        return [f"index.md: unsupported okf_version {version!r}; expected '0.2'"]
    return []


def _has_frontmatter_block(raw: str) -> bool:
    """Detect a complete YAML frontmatter block in reserved Markdown content."""

    lines = raw.removeprefix("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    return any(line.strip() == "---" for line in lines[1:])


def _validate_reserved_file(path: Path, relative: str) -> list[str]:
    """Validate reserved index/log files without treating them as Concept Documents."""

    raw, errors = _read_portable_markdown(path, relative)
    if raw is None:
        return errors
    if _has_frontmatter_block(raw):
        return [f"{relative}: reserved {path.name} cannot contain frontmatter"]
    if path.name != "log.md":
        return []
    errors: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("## "):
            continue
        heading = line.removeprefix("## ").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", heading):
            errors.append(f"{relative}: log date heading {heading!r} must use YYYY-MM-DD")
            continue
        try:
            date.fromisoformat(heading)
        except ValueError:
            errors.append(f"{relative}: log date heading {heading!r} is not a valid calendar date")
    return errors


def validate_portable_bundle(bundle: Path) -> None:
    """Validate the portable OKF v0.2 contract without private Forge state."""
    if not bundle.is_dir():
        raise ValidationFailure(f"Bundle directory does not exist: {bundle}")
    errors = _validate_root_index(bundle / "index.md")
    for path in sorted(bundle.rglob("*.md")):
        relative = path.relative_to(bundle).as_posix()
        if relative.startswith(".knowledge-forge/"):
            continue
        if path.name in {"index.md", "log.md"}:
            if relative != "index.md":
                errors.extend(_validate_reserved_file(path, relative))
            continue
        concept_id = path.relative_to(bundle).with_suffix("").as_posix()
        raw, read_errors = _read_portable_markdown(path, concept_id)
        if raw is None:
            errors.extend(read_errors)
            continue
        errors.extend(validate_portable_concept(raw, concept_id))
    if errors:
        raise ValidationFailure("Portable OKF 0.2 validation failed:\n- " + "\n- ".join(errors))


def validate_bundle(
    bundle: Path,
    sources: list[KnowledgeSource] | None = None,
    *,
    check_live_hash: bool = False,
    for_mutation: bool = False,
) -> ForgeState:
    """Validate a managed Bundle's files, state, sources, and integrity hashes.

    When ``sources`` is supplied, source references are checked against the supplied
    evidence counts; ``check_live_hash`` additionally verifies current source hashes.
    ``for_mutation`` permits the transient inconsistencies expected while a mutation
    is rebuilding managed files, while still enforcing the private-state contract.
    """

    errors: list[str] = []
    try:
        state = load_state(bundle)
    except ValidationFailure as exc:
        raise ValidationFailure(str(exc)) from exc
    concepts = public_concepts(bundle)
    source_pages = None if sources is None else {item.id: len(item.evidence) for item in sources}

    allowed_markdown = {"index.md", "log.md"} | {f"{concept_id}.md" for concept_id in concepts}
    for path in bundle.rglob("*.md"):
        relative = path.relative_to(bundle).as_posix()
        if relative not in allowed_markdown:
            errors.append(f"Markdown file is outside the MVP Concept namespace: {relative}")

    index = bundle / "index.md"
    log = bundle / "log.md"
    for relative, path in (("index.md", index), ("log.md", log)):
        if not path.is_file():
            errors.append(f"Missing tool-managed file: {relative}")
        elif state.tool_files.get(relative) != _file_hash(path):
            errors.append(f"Tool-managed file was modified: {relative}")
    if (
        index.is_file()
        and not for_mutation
        and index.read_text(encoding="utf-8") != render_index(concepts)
    ):
        errors.append("index.md is not the deterministic index for the current Concept set")

    for concept_id, raw in concepts.items():
        if not re.fullmatch(r"concepts/[a-z0-9]+(?:-[a-z0-9]+)*", concept_id):
            errors.append(f"Invalid Concept ID: {concept_id}")
        concept_state = state.concepts.get(concept_id)
        if concept_state is not None and concept_state.ownership == "imported":
            errors.extend(validate_portable_concept(raw, concept_id))
        else:
            errors.extend(validate_concept(raw, concept_id, source_pages))
        if concept_state is None:
            continue  # a newly human-authored Concept is registered by the next mutation
        if concept_state.managed_fields_hash != managed_fields_hash(raw):
            errors.append(f"Tool-managed provenance was modified: {concept_id}.md")
        if concept_state.ownership in {"agent", "imported"}:
            try:
                baseline = load_baseline(bundle, concept_id)
                if baseline.sha256 != concept_state.baseline_hash:
                    errors.append(f"Baseline hash disagrees with state: {concept_id}")
            except ValidationFailure as exc:
                errors.append(str(exc))

    for concept_id, concept_state in state.concepts.items():
        if concept_id not in concepts:
            if concept_state.deleted:
                continue
            if for_mutation:
                continue
            if concept_state.ownership == "human":
                errors.append(f"Human-owned Concept was deleted: {concept_id}")
            else:
                baseline = load_baseline(bundle, concept_id)
                errors.append(
                    f"Agent Concept was manually deleted: {concept_id}; baseline {baseline.sha256}"
                )

    if sources is not None:
        if source_set_hash(sources) != state.source_set_hash:
            errors.append("Authoritative source set differs from the last published state")
        actual_ids = {source.id for source in sources}
        if actual_ids != set(state.sources):
            errors.append("Source IDs differ from state")
        for source in sources:
            expected = state.sources.get(source.id)
            if expected and (
                expected.content_sha256 != source.content_sha256
                or expected.page_count != len(source.evidence)
            ):
                errors.append(f"Source hash or page count differs: {source.id}")

    if check_live_hash and bundle_hash(bundle) != state.bundle_hash:
        errors.append("Live Bundle differs from the last published state")
    if errors:
        raise ValidationFailure("Bundle validation failed:\n- " + "\n- ".join(errors))
    return state
