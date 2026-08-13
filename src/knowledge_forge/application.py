from __future__ import annotations

import ast
import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .agent import ReasoningAgent
from .errors import ReconciliationRequired, ValidationFailure
from .merge import body_sections, join_sections, merge_concept
from .models import (
    ConceptState,
    ConditionalOverride,
    Conflict,
    ForgeState,
    GenerationIdentity,
    PDFSource,
    ReconciliationManifest,
    ResolutionFile,
    SourceState,
    VerificationEvent,
)
from .okf import (
    concept_version_hash,
    dump_markdown,
    managed_fields_hash,
    parse_markdown,
    render_index,
)
from .publish import output_lock, publish_staging, staged_bundle
from .security import generation_identity, reject_tracing
from .sources import PageIndex, extract_sources, sha256_text, source_set_hash
from .state import (
    baseline_path,
    bundle_hash,
    load_baseline,
    load_state,
    public_concepts,
    write_baseline,
    write_state,
)
from .timing import ProcessingTimer, processing_phase
from .validation import validate_bundle
from .workflow import build_workflow


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _actor(model: str) -> str:
    normalized = model.strip().replace(" ", "-")
    return f"knowledge-forge/{normalized}"


def _log_entry(action: str, detail: str) -> str:
    return f"## {_now()[:10]}\n* **{action}**: {detail}\n"


def _append_log(existing: str, entry: str) -> str:
    if not existing.strip():
        return f"# Knowledge Forge Update Log\n\n{entry}"
    marker = "# Knowledge Forge Update Log\n"
    body = existing.removeprefix(marker).lstrip()
    return f"{marker}\n{entry}\n{body}".rstrip() + "\n"


def _source_states(sources: list[PDFSource]) -> dict[str, SourceState]:
    return {
        source.id: SourceState(content_sha256=source.content_sha256, page_count=len(source.pages))
        for source in sources
    }


def _source_dependencies(raw: str) -> dict[str, str]:
    metadata, _ = parse_markdown(raw)
    return {
        str(item["id"]): str(item["content_sha256"])
        for item in metadata.get("sources", [])
        if isinstance(item, dict) and "id" in item and "content_sha256" in item
    }


def _evidence_from_raw(raw: str) -> list[dict[str, object]]:
    metadata, _ = parse_markdown(raw)
    return [
        {"source_id": item["id"], "pages": _expand_pages(item.get("pages", []))}
        for item in metadata.get("sources", [])
        if isinstance(item, dict) and "id" in item
    ]


def _conflict_evidence_hash(conflict: Conflict) -> str:
    return sha256_text(
        json.dumps(
            [item.model_dump(mode="json") for item in conflict.evidence],
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _candidate_evidence_hash(raw: str) -> str:
    return _conflict_evidence_hash(
        Conflict(
            id="evidence",
            concept_id="evidence",
            block_id="evidence",
            reason="evidence",
            evidence=_evidence_from_raw(raw),
        )
    )


def _block_value(raw: str, block_id: str) -> str:
    if block_id in {"ownership", "document:source-removal", "document:deletion"}:
        return raw
    metadata, body = parse_markdown(raw)
    if block_id.startswith("frontmatter:"):
        return repr(metadata.get(block_id.removeprefix("frontmatter:")))
    return body_sections(body).get(block_id, "")


def _override_matches(override: ConditionalOverride, conflict: Conflict, human_raw: str) -> bool:
    return (
        override.concept_id == conflict.concept_id
        and override.block_id == conflict.block_id
        and override.human_hash == sha256_text(_block_value(human_raw, conflict.block_id))
        and override.evidence_hash == _conflict_evidence_hash(conflict)
    )


def _write_bundle(
    staging: Path,
    *,
    concepts: dict[str, str],
    baselines: dict[str, str],
    ownership: dict[str, str],
    sources: list[PDFSource],
    generation: GenerationIdentity,
    previous_state: ForgeState | None,
    action: str,
    log_detail: str,
    overrides: list[ConditionalOverride] | None = None,
    deleted: dict[str, str] | None = None,
) -> ForgeState:
    deleted = deleted or {}
    concepts_dir = staging / "concepts"
    if concepts_dir.exists():
        shutil.rmtree(concepts_dir)
    for concept_id, raw in concepts.items():
        path = staging / f"{concept_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding="utf-8")

    baseline_dir = staging / ".knowledge-forge" / "baseline"
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    baseline_hashes: dict[str, str] = {}
    for concept_id, raw in baselines.items():
        baseline_hashes[concept_id] = write_baseline(staging, concept_id, raw)

    index_raw = render_index(concepts)
    (staging / "index.md").write_text(index_raw, encoding="utf-8")
    log_path = staging / "log.md"
    prior_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    log_raw = _append_log(prior_log, _log_entry(action, log_detail))
    log_path.write_text(log_raw, encoding="utf-8")

    concept_states: dict[str, ConceptState] = {
        concept_id: ConceptState(
            ownership=ownership[concept_id],
            baseline_hash=baseline_hashes.get(concept_id),
            source_dependencies=_source_dependencies(raw),
            managed_fields_hash=managed_fields_hash(raw),
        )
        for concept_id, raw in concepts.items()
    }
    for concept_id, candidate_hash in deleted.items():
        concept_states[concept_id] = ConceptState(
            ownership="human",
            deleted=True,
            deletion_candidate_hash=candidate_hash,
        )
    state = ForgeState(
        generation=generation,
        source_set_hash=source_set_hash(sources),
        sources=_source_states(sources),
        bundle_hash="",
        tool_files={
            "index.md": sha256_text(index_raw),
            "log.md": sha256_text(log_raw),
        },
        concepts=concept_states,
        overrides=(
            overrides
            if overrides is not None
            else (previous_state.overrides if previous_state else [])
        ),
        verification_history=(previous_state.verification_history if previous_state else []),
    )
    state.bundle_hash = bundle_hash(staging)
    write_state(staging, state)
    validate_bundle(staging, check_live_hash=True)
    return state


def _run_agent(
    *,
    sources: list[PDFSource],
    generation: GenerationIdentity,
    api_key: str,
    existing_ids: list[str],
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> tuple[dict[str, str], str]:
    if not api_key.strip():
        raise ValidationFailure("OPENAI_API_KEY must not be empty")
    temporary = Path(tempfile.mkdtemp(prefix="knowledge-forge-fts-"))
    report = progress or (lambda _: None)
    try:
        page_count = sum(len(source.pages) for source in sources)
        report(f"Indexing {page_count} pages from {len(sources)} PDFs...")
        with processing_phase(timing, "PDF indexing"):
            index = PageIndex(temporary / "pages.sqlite")
            try:
                index.add(sources)
            except Exception:
                index.close()
                raise

        def agent_factory() -> ReasoningAgent:
            return ReasoningAgent(
                index=index,
                sources=sources,
                model=generation.model,
                api_key=api_key,
                base_url=generation.endpoint,
                max_steps=generation.max_agent_steps,
                parallel_tool_calls=generation.parallel_tool_calls,
            )

        with index:
            graph = build_workflow(
                agent_factory,
                sources,
                _actor(generation.model),
                progress=progress,
                timing=timing,
            )
            result = graph.invoke(
                {"language": generation.language, "existing_ids": existing_ids},
                {"recursion_limit": 10},
            )
            return dict(result["concepts"]), str(result["language"])
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def generate(
    *,
    source: Path,
    output: Path,
    model: str,
    api_key: str,
    base_url: str | None,
    language: str,
    max_agent_steps: int,
    parallel_tool_calls: bool = True,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> None:
    reject_tracing()
    identity = generation_identity(
        model=model,
        base_url=base_url,
        language=language,
        max_agent_steps=max_agent_steps,
        parallel_tool_calls=parallel_tool_calls,
    )
    with output_lock(output.resolve()):
        _generate_locked(
            source=source,
            output=output,
            api_key=api_key,
            generation=identity,
            progress=progress,
            timing=timing,
        )


def _generate_locked(
    *,
    source: Path,
    output: Path,
    api_key: str,
    generation: GenerationIdentity,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> None:
    report = progress or (lambda _: None)
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValidationFailure("generate requires a missing or empty --out directory")
    report(
        "Tool-call mode: parallel."
        if generation.parallel_tool_calls
        else "Tool-call mode: non-parallel compatibility."
    )
    report("Reading PDF sources...")
    with processing_phase(timing, "PDF Source reading"):
        sources = extract_sources(source)
    report(f"Loaded {len(sources)} PDFs with {sum(len(item.pages) for item in sources)} pages.")
    concepts, output_language = _run_agent(
        sources=sources,
        generation=generation,
        api_key=api_key,
        existing_ids=[],
        progress=progress,
        timing=timing,
    )
    generation.output_language = output_language
    report("Writing and validating the candidate bundle...")
    with staged_bundle(output, copy_existing=False) as staging:
        with processing_phase(timing, "Candidate Bundle writing and validation"):
            _write_bundle(
                staging,
                concepts=concepts,
                baselines=concepts,
                ownership={concept_id: "agent" for concept_id in concepts},
                sources=sources,
                generation=generation,
                previous_state=None,
                action="Generation",
                log_detail=f"Created {len(concepts)} Concepts from {len(sources)} PDF sources.",
            )
        report("Publishing the bundle atomically...")
        with processing_phase(timing, "Atomic publication"):
            publish_staging(staging, output)


def _register_human_concepts(current: dict[str, str], state: ForgeState) -> dict[str, str]:
    ownership = {concept_id: item.ownership for concept_id, item in state.concepts.items()}
    for concept_id in current.keys() - state.concepts.keys():
        ownership[concept_id] = "human"
    return ownership


def _preserve_verification(current: str, merged: str) -> str:
    current_meta, _ = parse_markdown(current)
    if "verified" not in current_meta or concept_version_hash(current) != concept_version_hash(
        merged
    ):
        return merged
    merged_meta, merged_body = parse_markdown(merged)
    merged_meta["verified"] = current_meta["verified"]
    return dump_markdown(merged_meta, merged_body)


def _same_generation_request(left: GenerationIdentity, right: GenerationIdentity) -> bool:
    return left.model_dump(exclude={"output_language"}) == right.model_dump(
        exclude={"output_language"}
    )


def update(
    *,
    source: Path,
    output: Path,
    model: str,
    api_key: str,
    base_url: str | None,
    language: str,
    max_agent_steps: int,
    progress: Callable[[str], None] | None = None,
) -> bool:
    reject_tracing()
    with output_lock(output.resolve()):
        return _update_locked(
            source=source,
            output=output,
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            max_agent_steps=max_agent_steps,
            progress=progress,
        )


def _update_locked(
    *,
    source: Path,
    output: Path,
    model: str,
    api_key: str,
    base_url: str | None,
    language: str,
    max_agent_steps: int,
    progress: Callable[[str], None] | None = None,
) -> bool:
    report = progress or (lambda _: None)
    output = output.resolve()
    report("Validating the current bundle...")
    state = validate_bundle(output, for_mutation=True)
    report("Reading PDF sources...")
    sources = extract_sources(source)
    report(f"Loaded {len(sources)} PDFs with {sum(len(item.pages) for item in sources)} pages.")
    identity = generation_identity(
        model=model,
        base_url=base_url,
        language=language,
        max_agent_steps=max_agent_steps,
        parallel_tool_calls=state.generation.parallel_tool_calls,
    )
    current = public_concepts(output)
    if (
        source_set_hash(sources) == state.source_set_hash
        and _same_generation_request(identity, state.generation)
        and bundle_hash(output) == state.bundle_hash
    ):
        report("The source set and bundle are unchanged.")
        return False

    ownership = _register_human_concepts(current, state)
    deleted = {
        concept_id: item.deletion_candidate_hash or ""
        for concept_id, item in state.concepts.items()
        if item.deleted
    }
    source_unchanged = source_set_hash(sources) == state.source_set_hash
    if source_unchanged and _same_generation_request(identity, state.generation):
        report("Reusing the previous agent baseline; source evidence is unchanged.")
        identity = state.generation
        candidates = {
            concept_id: load_baseline(output, concept_id).raw_markdown
            for concept_id, owner in ownership.items()
            if owner == "agent" and concept_id in state.concepts
        }
    else:
        candidates, output_language = _run_agent(
            sources=sources,
            generation=identity,
            api_key=api_key,
            existing_ids=sorted(current),
            progress=progress,
        )
        identity.output_language = output_language

    report("Merging agent candidates with human edits...")
    published: dict[str, str] = {}
    baselines: dict[str, str] = {}
    conflicts: list[Conflict] = []
    for concept_id, owner in ownership.items():
        human_raw = current.get(concept_id)
        candidate_raw = candidates.get(concept_id)
        if owner == "human":
            prior = state.concepts.get(concept_id)
            if prior and prior.deleted:
                if candidate_raw is None:
                    continue
                candidate_evidence_hash = _candidate_evidence_hash(candidate_raw)
                if candidate_evidence_hash == prior.deletion_candidate_hash:
                    continue
                conflicts.append(
                    Conflict(
                        id=sha256_text(f"{concept_id}\0deleted-change")[:16],
                        concept_id=concept_id,
                        block_id="document:deletion",
                        human="<deleted>",
                        candidate=candidate_raw,
                        evidence=_evidence_from_raw(candidate_raw),
                        reason="Evidence for a human-deleted Concept changed.",
                    )
                )
                published[concept_id] = candidate_raw
                baselines[concept_id] = candidate_raw
                deleted.pop(concept_id, None)
                ownership[concept_id] = "agent"
                continue
            if human_raw is not None:
                published[concept_id] = human_raw
            if candidate_raw is not None:
                conflict = Conflict(
                    id=sha256_text(f"{concept_id}\0ownership")[:16],
                    concept_id=concept_id,
                    block_id="ownership",
                    human=human_raw,
                    candidate=candidate_raw,
                    evidence=_evidence_from_raw(candidate_raw),
                    reason="Agent candidate collides with a persistently human-owned Concept.",
                )
                if not any(
                    _override_matches(override, conflict, human_raw or "")
                    for override in state.overrides
                ):
                    conflicts.append(conflict)
            continue

        baseline = load_baseline(output, concept_id).raw_markdown
        if human_raw is None:
            if candidate_raw is not None:
                conflicts.append(
                    Conflict(
                        id=sha256_text(f"{concept_id}\0deletion")[:16],
                        concept_id=concept_id,
                        block_id="document:deletion",
                        baseline=baseline,
                        candidate=candidate_raw,
                        evidence=_evidence_from_raw(candidate_raw),
                        reason=(
                            "A human deleted an agent-owned Concept that still has source support."
                        ),
                    )
                )
                published[concept_id] = candidate_raw
                baselines[concept_id] = candidate_raw
            continue
        if candidate_raw is None:
            if human_raw != baseline:
                conflict = Conflict(
                    id=sha256_text(f"{concept_id}\0source-removal")[:16],
                    concept_id=concept_id,
                    block_id="document:source-removal",
                    baseline=baseline,
                    human=human_raw,
                    reason="Source support was removed from a human-edited Concept.",
                )
                if not any(
                    _override_matches(override, conflict, human_raw) for override in state.overrides
                ):
                    conflicts.append(conflict)
                published[concept_id] = human_raw
                baselines[concept_id] = baseline
            continue
        evidence = _evidence_from_raw(candidate_raw)
        result = merge_concept(concept_id, baseline, human_raw, candidate_raw, evidence)
        active_conflicts = [
            conflict
            for conflict in result.conflicts
            if not any(
                _override_matches(override, conflict, human_raw) for override in state.overrides
            )
        ]
        conflicts.extend(active_conflicts)
        merged = _preserve_verification(human_raw, result.markdown)
        published[concept_id] = merged
        baselines[concept_id] = candidate_raw

    for concept_id, candidate_raw in candidates.items():
        if concept_id not in ownership:
            published[concept_id] = candidate_raw
            baselines[concept_id] = candidate_raw
            ownership[concept_id] = "agent"

    report("Writing and validating the candidate bundle...")
    with staged_bundle(output, copy_existing=True) as staging:
        _write_bundle(
            staging,
            concepts=published,
            baselines=baselines,
            ownership=ownership,
            sources=sources,
            generation=identity,
            previous_state=state,
            action="Update",
            log_detail=f"Reconciled {len(published)} Concepts from {len(sources)} PDF sources.",
            deleted=deleted,
        )
        if conflicts:
            report("Conflicts require human reconciliation; preserving the current bundle.")
            _write_reconciliation(output, staging, state, sources, identity, conflicts)
            raise ReconciliationRequired(str(output.parent / f"{output.name}.reconciliation.md"))
        report("Publishing the bundle atomically...")
        publish_staging(staging, output)
    return True


def _expand_pages(values: list[str]) -> list[int]:
    from .okf import expand_ranges

    return expand_ranges(values)


def _write_reconciliation(
    output: Path,
    staging: Path,
    prior_state: ForgeState,
    sources: list[PDFSource],
    identity: GenerationIdentity,
    conflicts: list[Conflict],
) -> None:
    work = output.parent / f"{output.name}.reconciliation"
    report = output.parent / f"{output.name}.reconciliation.md"
    if work.exists():
        shutil.rmtree(work)
    pending = work / "pending"
    shutil.copytree(staging, pending)
    manual = work / "manual"
    for conflict in conflicts:
        concept = pending / f"{conflict.concept_id}.md"
        if concept.is_file():
            destination = manual / f"{conflict.concept_id}.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(concept, destination)
    manifest = ReconciliationManifest(
        output_path=str(output),
        live_bundle_hash=bundle_hash(output, include_state=True),
        source_set_hash=source_set_hash(sources),
        candidate_hash=bundle_hash(pending, include_state=True),
        generation=identity,
        conflicts=conflicts,
    )
    work.mkdir(parents=True, exist_ok=True)
    (work / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    template = {
        "resolutions": [
            {
                "conflict_id": conflict.id,
                "action": "keep-human",
                "artifact": f"manual/{conflict.concept_id}.md",
            }
            for conflict in conflicts
        ]
    }
    (work / "resolution.yaml").write_text(yaml.safe_dump(template, sort_keys=False))
    lines = [
        "# Knowledge Forge Reconciliation",
        "",
        "Status: pending",
        "",
        "Choose `keep-human`, `use-source`, or `manual` for each conflict in the sibling",
        "`resolution.yaml`. For `manual`, edit the matching artifact under `manual/`.",
        "",
    ]
    for conflict in conflicts:
        lines.extend(
            [
                f"## {conflict.id}: `{conflict.concept_id}` / `{conflict.block_id}`",
                "",
                conflict.reason,
                "",
                f"- Evidence: `{_conflict_evidence_hash(conflict)}`",
                "",
                "### Current human value",
                "",
                _report_value(conflict.human),
                "",
                "### Agent candidate",
                "",
                _report_value(conflict.candidate),
                "",
                "### Provenance",
                "",
                *(
                    [
                        f"- `{item.source_id}` pages " + ", ".join(str(page) for page in item.pages)
                        for item in conflict.evidence
                    ]
                    or ["- No current source evidence; this is an ownership/lifecycle conflict."]
                ),
                "",
            ]
        )
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _report_value(value: str | None) -> str:
    if value is None:
        return "_(deleted or absent)_"
    return "\n".join(f"    {line}" for line in value.splitlines()) or "_(empty)_"


def _set_conflict_value(raw: str, conflict: Conflict, value: str | None) -> str:
    if conflict.block_id.startswith("frontmatter:"):
        metadata, body = parse_markdown(raw)
        key = conflict.block_id.removeprefix("frontmatter:")
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = ast.literal_eval(value)
        return dump_markdown(metadata, body)
    if conflict.block_id.startswith("body:"):
        metadata, body = parse_markdown(raw)
        sections = body_sections(body)
        if value is None:
            sections.pop(conflict.block_id, None)
        else:
            sections[conflict.block_id] = value
        return dump_markdown(metadata, join_sections(sections))
    return raw


def reconcile(*, source: Path, output: Path, resolution_path: Path) -> None:
    reject_tracing()
    with output_lock(output.resolve()):
        _reconcile_locked(source=source, output=output, resolution_path=resolution_path)


def _reconcile_locked(*, source: Path, output: Path, resolution_path: Path) -> None:
    output = output.resolve()
    work = output.parent / f"{output.name}.reconciliation"
    report = output.parent / f"{output.name}.reconciliation.md"
    manifest = ReconciliationManifest.model_validate_json(
        (work / "manifest.json").read_text(encoding="utf-8")
    )
    resolutions = ResolutionFile.model_validate(
        yaml.safe_load(resolution_path.read_text(encoding="utf-8"))
    )
    sources = extract_sources(source)
    pending = work / "pending"
    if manifest.output_path != str(output):
        raise ValidationFailure("Reconciliation output path does not match the manifest")
    if bundle_hash(output, include_state=True) != manifest.live_bundle_hash:
        raise ValidationFailure("Live Bundle changed; run update again")
    if source_set_hash(sources) != manifest.source_set_hash:
        raise ValidationFailure("Source set changed; run update again")
    if bundle_hash(pending, include_state=True) != manifest.candidate_hash:
        raise ValidationFailure("Pending candidate changed outside the resolution workflow")
    choices = {item.conflict_id: item for item in resolutions.resolutions}
    if set(choices) != {item.id for item in manifest.conflicts}:
        raise ValidationFailure("Resolution file must resolve every conflict exactly once")
    conflicts_by_concept: dict[str, list[Conflict]] = {}
    for conflict in manifest.conflicts:
        conflicts_by_concept.setdefault(conflict.concept_id, []).append(conflict)
    for concept_id, concept_conflicts in conflicts_by_concept.items():
        manual_choices = [
            choices[item.id] for item in concept_conflicts if choices[item.id].action == "manual"
        ]
        if manual_choices and (
            len(manual_choices) != len(concept_conflicts)
            or len({item.artifact for item in manual_choices}) != 1
        ):
            raise ValidationFailure(
                f"All conflicts for {concept_id} must use the same artifact when resolving manually"
            )

    pending_concepts = public_concepts(pending)
    pending_state = load_state(pending)
    overrides = list(pending_state.overrides)
    ownership = {concept_id: item.ownership for concept_id, item in pending_state.concepts.items()}
    deleted = {
        concept_id: item.deletion_candidate_hash or ""
        for concept_id, item in pending_state.concepts.items()
        if item.deleted
    }
    baseline_overrides: dict[str, str] = {}
    for conflict in manifest.conflicts:
        choice = choices[conflict.id]
        raw = pending_concepts.get(conflict.concept_id, "")
        if choice.action == "keep-human":
            if conflict.block_id == "document:deletion":
                pending_concepts.pop(conflict.concept_id, None)
                ownership[conflict.concept_id] = "human"
                deleted[conflict.concept_id] = _candidate_evidence_hash(conflict.candidate or "")
                continue
            if conflict.human is None and not conflict.block_id.startswith(
                ("frontmatter:", "body:")
            ):
                raise ValidationFailure(f"Conflict {conflict.id} has no human value to keep")
            raw = _set_conflict_value(raw, conflict, conflict.human)
            overrides.append(
                ConditionalOverride(
                    concept_id=conflict.concept_id,
                    block_id=conflict.block_id,
                    human_hash=sha256_text(_block_value(raw, conflict.block_id)),
                    evidence_hash=_conflict_evidence_hash(conflict),
                )
            )
        elif choice.action == "use-source":
            if conflict.block_id == "document:source-removal":
                pending_concepts.pop(conflict.concept_id, None)
                ownership.pop(conflict.concept_id, None)
                deleted.pop(conflict.concept_id, None)
                continue
            if conflict.block_id in {"document:deletion", "ownership"}:
                if conflict.candidate is None:
                    raise ValidationFailure(f"Conflict {conflict.id} has no candidate")
                raw = conflict.candidate
                ownership[conflict.concept_id] = "agent"
                deleted.pop(conflict.concept_id, None)
                baseline_overrides[conflict.concept_id] = conflict.candidate
            raw = _set_conflict_value(raw, conflict, conflict.candidate)
            overrides = [
                item
                for item in overrides
                if not (
                    item.concept_id == conflict.concept_id and item.block_id == conflict.block_id
                )
            ]
        else:
            if not choice.artifact:
                raise ValidationFailure(f"Manual resolution {conflict.id} requires artifact")
            artifact = (work / choice.artifact).resolve()
            if work.resolve() not in artifact.parents:
                raise ValidationFailure(
                    "Manual resolution artifact must remain inside work directory"
                )
            raw = artifact.read_text(encoding="utf-8")
        if raw:
            pending_concepts[conflict.concept_id] = raw
        else:
            pending_concepts.pop(conflict.concept_id, None)

    baselines = {
        concept_id: load_baseline(pending, concept_id).raw_markdown
        for concept_id, item in pending_state.concepts.items()
        if ownership.get(concept_id) == "agent" and baseline_path(pending, concept_id).is_file()
    }
    baselines.update(baseline_overrides)
    baselines = {
        concept_id: raw
        for concept_id, raw in baselines.items()
        if ownership.get(concept_id) == "agent"
    }
    with staged_bundle(output, copy_existing=True) as staging:
        _write_bundle(
            staging,
            concepts=pending_concepts,
            baselines=baselines,
            ownership=ownership,
            sources=sources,
            generation=manifest.generation,
            previous_state=pending_state,
            action="Reconciliation",
            log_detail=f"Resolved {len(manifest.conflicts)} conflicts.",
            overrides=overrides,
            deleted=deleted,
        )
        publish_staging(staging, output)
    shutil.rmtree(work)
    prior = report.read_text(encoding="utf-8") if report.exists() else ""
    report.write_text(
        prior.replace("Status: pending", "Status: resolved") + f"\nResolved at: {_now()}\n",
        encoding="utf-8",
    )


def verify(*, source: Path, output: Path, concept_id: str, actor: str) -> None:
    reject_tracing()
    with output_lock(output.resolve()):
        _verify_locked(source=source, output=output, concept_id=concept_id, actor=actor)


def _verify_locked(*, source: Path, output: Path, concept_id: str, actor: str) -> None:
    if not actor.startswith("human:") or len(actor) <= len("human:"):
        raise ValidationFailure("--by must use the actor form human:<id>")
    output = output.resolve()
    sources = extract_sources(source)
    state = validate_bundle(output, sources, for_mutation=True)
    concepts = public_concepts(output)
    missing = [
        concept_id
        for concept_id, item in state.concepts.items()
        if concept_id not in concepts and not item.deleted
    ]
    if missing:
        raise ValidationFailure(
            "verify cannot reconcile deleted Concepts; run update first: " + ", ".join(missing)
        )
    normalized_id = concept_id.removesuffix(".md").lstrip("/")
    if normalized_id not in concepts:
        raise ValidationFailure(f"Unknown Concept ID: {normalized_id}")
    raw = concepts[normalized_id]
    metadata, body = parse_markdown(raw)
    event = {"by": actor, "at": _now()}
    current_version = concept_version_hash(raw)
    version_was_verified = any(
        item.concept_id == normalized_id and item.version_hash == current_version
        for item in state.verification_history
    )
    existing = metadata.get("verified", []) if version_was_verified else []
    if isinstance(existing, dict):
        existing = [existing]
    metadata["verified"] = [*existing, event]
    concepts[normalized_id] = dump_markdown(metadata, body)
    state.verification_history.append(
        VerificationEvent(
            concept_id=normalized_id,
            by=actor,
            at=event["at"],
            version_hash=concept_version_hash(concepts[normalized_id]),
        )
    )
    ownership = _register_human_concepts(concepts, state)
    baselines = {
        concept_id: load_baseline(output, concept_id).raw_markdown
        for concept_id, owner in ownership.items()
        if owner == "agent"
    }
    deleted = {
        concept_id: item.deletion_candidate_hash or ""
        for concept_id, item in state.concepts.items()
        if item.deleted
    }
    with staged_bundle(output, copy_existing=True) as staging:
        _write_bundle(
            staging,
            concepts=concepts,
            baselines=baselines,
            ownership=ownership,
            sources=sources,
            generation=state.generation,
            previous_state=state,
            action="Verification",
            log_detail=f"{actor} verified {normalized_id}.",
            deleted=deleted,
        )
        publish_staging(staging, output)
