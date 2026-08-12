from __future__ import annotations

import re
from pathlib import Path

from .errors import ValidationFailure
from .models import ForgeState, PDFSource
from .okf import managed_fields_hash, render_index, validate_concept
from .sources import sha256_text, source_set_hash
from .state import bundle_hash, load_baseline, load_state, public_concepts


def _file_hash(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def validate_bundle(
    bundle: Path,
    sources: list[PDFSource] | None = None,
    *,
    check_live_hash: bool = False,
    for_mutation: bool = False,
) -> ForgeState:
    errors: list[str] = []
    try:
        state = load_state(bundle)
    except ValidationFailure as exc:
        raise ValidationFailure(str(exc)) from exc
    concepts = public_concepts(bundle)
    source_pages = None if sources is None else {item.id: len(item.pages) for item in sources}

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
        errors.extend(validate_concept(raw, concept_id, source_pages))
        concept_state = state.concepts.get(concept_id)
        if concept_state is None:
            continue  # a newly human-authored Concept is registered by the next mutation
        if concept_state.managed_fields_hash != managed_fields_hash(raw):
            errors.append(f"Tool-managed provenance was modified: {concept_id}.md")
        if concept_state.ownership == "agent":
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
                or expected.page_count != len(source.pages)
            ):
                errors.append(f"Source hash or page count differs: {source.id}")

    if check_live_hash and bundle_hash(bundle) != state.bundle_hash:
        errors.append("Live Bundle differs from the last published state")
    if errors:
        raise ValidationFailure("Bundle validation failed:\n- " + "\n- ".join(errors))
    return state
