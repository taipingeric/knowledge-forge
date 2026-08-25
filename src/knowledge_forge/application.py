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
from .errors import ReconciliationRequired, StalenessDetected, ValidationFailure
from .merge import body_sections, join_sections, merge_concept
from .migrations import migrate_bundle, migration_available
from .models import (
    ConceptState,
    ConditionalOverride,
    Conflict,
    ForgeState,
    GenerationIdentity,
    KnowledgeSource,
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
    source_reference_identity,
)
from .publish import output_lock, publish_staging, staged_bundle
from .security import generation_identity, reject_tracing, resolve_disjoint_trees
from .sources import EvidenceIndex, extract_sources, sha256_text, source_set_hash
from .staleness import (
    detect_staleness,
    load_pending_staleness_report,
    prepared_staleness_resolution,
    write_staleness_report,
)
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
from .validation import validate_bundle, validate_portable_bundle
from .workflow import build_workflow


def _now() -> str:
    """Return the current UTC timestamp in the Bundle's canonical format."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _actor(model: str) -> str:
    """Normalize a model name into the process actor identifier stored in metadata."""

    normalized = model.strip().replace(" ", "-")
    return f"knowledge-forge/{normalized}"


def _log_entry(action: str, detail: str) -> str:
    """Render one dated operation entry for the human-readable Bundle log."""

    return f"## {_now()[:10]}\n* **{action}**: {detail}\n"


def _append_log(existing: str, entry: str) -> str:
    """Prepend an operation entry while preserving the log's stable heading."""

    if not existing.strip():
        return f"# Knowledge Forge Update Log\n\n{entry}"
    marker = "# Knowledge Forge Update Log\n"
    body = existing.removeprefix(marker).lstrip()
    return f"{marker}\n{entry}\n{body}".rstrip() + "\n"


def _report_tool_call_mode(report: Callable[[str], None], parallel: bool) -> None:
    """Report whether the agent uses native parallel or compatibility tool calls."""

    mode = "parallel" if parallel else "non-parallel compatibility"
    report(f"Tool-call mode: {mode}.")


def _source_states(sources: list[KnowledgeSource]) -> dict[str, SourceState]:
    """Snapshot source hashes and evidence counts for managed-state validation."""

    return {
        source.id: SourceState(
            content_sha256=source.content_sha256, page_count=len(source.evidence)
        )
        for source in sources
    }


def _source_dependencies(raw: str) -> dict[str, str]:
    """Extract the source identity-to-content-hash dependencies from a Concept."""

    metadata, _ = parse_markdown(raw)
    dependencies: dict[str, str] = {}
    for item in metadata.get("sources", []):
        if not isinstance(item, dict) or "id" not in item or "content_sha256" not in item:
            continue
        source_id = source_reference_identity(str(item["id"]))
        if source_id is not None:
            dependencies[source_id] = str(item["content_sha256"])
    return dependencies


def _evidence_from_raw(raw: str) -> list[dict[str, object]]:
    """Convert rendered Concept source entries into evidence values for conflicts."""

    metadata, _ = parse_markdown(raw)
    evidence: list[dict[str, object]] = []
    for item in metadata.get("sources", []):
        if not isinstance(item, dict):
            continue
        source_id = source_reference_identity(str(item.get("id", "")))
        locator = item.get("locator")
        if (
            source_id is not None
            and isinstance(locator, dict)
            and isinstance(locator.get("page"), int)
        ):
            evidence.append({"source_id": source_id, "pages": [locator["page"]]})
    return evidence


def _conflict_evidence_hash(conflict: Conflict) -> str:
    """Hash conflict evidence so conditional resolutions remain evidence-bound."""

    return sha256_text(
        json.dumps(
            [item.model_dump(mode="json") for item in conflict.evidence],
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _candidate_evidence_hash(raw: str) -> str:
    """Hash the source evidence declared by a candidate Concept document."""

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
    """Return the exact value used to bind an override to a conflict block."""

    if block_id in {"ownership", "document:source-removal", "document:deletion"}:
        return raw
    metadata, body = parse_markdown(raw)
    if block_id.startswith("frontmatter:"):
        return repr(metadata.get(block_id.removeprefix("frontmatter:")))
    return body_sections(body).get(block_id, "")


def _override_matches(override: ConditionalOverride, conflict: Conflict, human_raw: str) -> bool:
    """Check whether a conditional override still matches human text and evidence."""

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
    sources: list[KnowledgeSource],
    generation: GenerationIdentity,
    previous_state: ForgeState | None,
    action: str,
    log_detail: str,
    overrides: list[ConditionalOverride] | None = None,
    deleted: dict[str, str] | None = None,
) -> ForgeState:
    """Write a candidate Bundle, managed state, baselines, and deterministic artifacts."""

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
    sources: list[KnowledgeSource],
    generation: GenerationIdentity,
    api_key: str,
    existing_ids: list[str],
    output: Path,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> tuple[dict[str, str], str]:
    """Run the planning and synthesis workflow against temporary indexed evidence."""

    if not api_key.strip():
        raise ValidationFailure("OPENAI_API_KEY must not be empty")
    temporary = Path(tempfile.mkdtemp(prefix="knowledge-forge-fts-"))
    report = progress or (lambda _: None)
    try:
        page_count = sum(len(source.evidence) for source in sources)
        report(f"Indexing {page_count} evidence units from {len(sources)} Knowledge Sources...")
        with processing_phase(timing, "Knowledge Source indexing"):
            index = EvidenceIndex(temporary / "pages.sqlite")
            try:
                index.add(sources)
            except Exception:
                index.close()
                raise

        def agent_factory() -> ReasoningAgent:
            """Create an agent sharing the temporary evidence index for this run."""

            return ReasoningAgent(
                index=index,
                sources=sources,
                model=generation.model,
                api_key=api_key,
                base_url=generation.endpoint,
                max_steps=generation.max_agent_steps,
                parallel_tool_calls=generation.parallel_tool_calls,
                bundle=output,
            )

        with index:
            graph = build_workflow(
                agent_factory,
                sources,
                _actor(generation.model),
                concept_concurrency=generation.concept_concurrency,
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
    concept_concurrency: int = 4,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> None:
    """Generate and atomically publish a new managed Bundle from PDF sources."""

    source, output = resolve_disjoint_trees(source, output)
    reject_tracing()
    imported = _recognized_import_concepts(source)
    if imported is not None:
        with output_lock(output.resolve()):
            _generate_import_locked(source, output, imported, progress=progress, timing=timing)
        return
    identity = generation_identity(
        model=model,
        base_url=base_url,
        language=language,
        max_agent_steps=max_agent_steps,
        parallel_tool_calls=parallel_tool_calls,
        concept_concurrency=concept_concurrency,
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


def _recognized_import_concepts(source: Path) -> dict[str, str] | None:
    """Return a portable-valid marked Bundle's Concepts, or None for ordinary sources."""
    index = source / "index.md"
    if not index.is_file():
        return None
    try:
        metadata, _ = parse_markdown(index.read_text(encoding="utf-8"))
    except ValidationFailure:
        return None
    if metadata != {"okf_version": "0.2"}:
        return None
    validate_portable_bundle(source)
    concepts: dict[str, str] = {}
    for path in sorted(source.rglob("*.md")):
        if path.name in {"index.md", "log.md"} or ".knowledge-forge" in path.parts:
            continue
        concept_id = path.relative_to(source).with_suffix("").as_posix()
        concepts[concept_id] = path.read_text(encoding="utf-8")
    return concepts


def _generate_import_locked(
    source: Path,
    output: Path,
    concepts: dict[str, str],
    *,
    progress: Callable[[str], None] | None,
    timing: ProcessingTimer | None,
) -> None:
    """Publish a recognized import Bundle without model configuration or reasoning."""
    report = progress or (lambda _: None)
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValidationFailure("generate requires a missing or empty --out directory")
    report("Validating and importing the recognized OKF Bundle...")
    identity = GenerationIdentity(
        model="import-only",
        endpoint="local://import",
        language="preserved",
        max_agent_steps=1,
    )
    with staged_bundle(output, copy_existing=False) as staging:
        with processing_phase(timing, "OKF import"):
            _write_bundle(
                staging,
                concepts=concepts,
                baselines=concepts,
                ownership={concept_id: "imported" for concept_id in concepts},
                sources=[],
                generation=identity,
                previous_state=None,
                action="Import",
                log_detail=f"Imported {len(concepts)} Concepts without reasoning.",
            )
        with processing_phase(timing, "Atomic publication"):
            publish_staging(staging, output)


def _generate_locked(
    *,
    source: Path,
    output: Path,
    api_key: str,
    generation: GenerationIdentity,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> None:
    """Build and publish a new Bundle while the caller holds the output lock."""

    report = progress or (lambda _: None)
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValidationFailure("generate requires a missing or empty --out directory")
    _report_tool_call_mode(report, generation.parallel_tool_calls)
    report("Reading Knowledge Sources...")
    with processing_phase(timing, "Knowledge Source reading"):
        sources = extract_sources(source)
    evidence_count = sum(len(item.evidence) for item in sources)
    report(f"Loaded {len(sources)} Knowledge Sources with {evidence_count} evidence units.")
    concepts, output_language = _run_agent(
        sources=sources,
        generation=generation,
        api_key=api_key,
        existing_ids=[],
        output=output,
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
                log_detail=(
                    f"Created {len(concepts)} Concepts from {len(sources)} Knowledge Sources."
                ),
            )
        report("Publishing the bundle atomically...")
        with processing_phase(timing, "Atomic publication"):
            publish_staging(staging, output)


def _register_human_concepts(current: dict[str, str], state: ForgeState) -> dict[str, str]:
    """Preserve recorded ownership and register newly added Concepts as human-owned."""

    ownership = {concept_id: item.ownership for concept_id, item in state.concepts.items()}
    for concept_id in current.keys() - state.concepts.keys():
        ownership[concept_id] = "human"
    return ownership


def _preserve_verification(current: str, merged: str) -> str:
    """Carry verification metadata across a merge only when semantic content is unchanged."""

    current_meta, _ = parse_markdown(current)
    if "verified" not in current_meta or concept_version_hash(current) != concept_version_hash(
        merged
    ):
        return merged
    merged_meta, merged_body = parse_markdown(merged)
    merged_meta["verified"] = current_meta["verified"]
    return dump_markdown(merged_meta, merged_body)


def _same_generation_request(left: GenerationIdentity, right: GenerationIdentity) -> bool:
    """Compare generation settings while ignoring the output language discovered by planning."""

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
    parallel_tool_calls: bool = True,
    regenerate_all: bool = False,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> bool:
    """Update a managed Bundle while preserving human edits and detecting conflicts."""

    source, output = resolve_disjoint_trees(source, output)
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
            parallel_tool_calls=parallel_tool_calls,
            regenerate_all=regenerate_all,
            progress=progress,
            timing=timing,
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
    parallel_tool_calls: bool = True,
    regenerate_all: bool = False,
    regeneration_output: Path | None = None,
    resolve_regeneration: bool = True,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
) -> bool:
    """Reconcile a changed source set or Bundle while the output lock is held."""

    if migration_available(output):
        with staged_bundle(output, copy_existing=True) as staging:
            migrate_bundle(staging)
            changed = _update_locked(
                source=source,
                output=staging,
                model=model,
                api_key=api_key,
                base_url=base_url,
                language=language,
                max_agent_steps=max_agent_steps,
                parallel_tool_calls=parallel_tool_calls,
                regenerate_all=regenerate_all,
                regeneration_output=output if regenerate_all else None,
                resolve_regeneration=False,
                progress=progress,
                timing=timing,
            )
            if regenerate_all:
                with prepared_staleness_resolution(output, load_pending_staleness_report(output)):
                    publish_staging(staging, output)
            else:
                publish_staging(staging, output)
            return changed
    report = progress or (lambda _: None)
    output = output.resolve()
    _report_tool_call_mode(report, parallel_tool_calls)
    report("Validating the current bundle...")
    with processing_phase(timing, "Current Bundle validation"):
        state = validate_bundle(output, for_mutation=True)
    report("Reading Knowledge Sources...")
    with processing_phase(timing, "Knowledge Source reading"):
        sources = extract_sources(source)
    evidence_count = sum(len(item.evidence) for item in sources)
    report(f"Loaded {len(sources)} Knowledge Sources with {evidence_count} evidence units.")
    identity = generation_identity(
        model=model,
        base_url=base_url,
        language=language,
        max_agent_steps=max_agent_steps,
        parallel_tool_calls=parallel_tool_calls,
        concept_concurrency=state.generation.concept_concurrency,
    )
    source_hash = source_set_hash(sources)
    regeneration_report: dict[str, object] | None = None
    if regenerate_all:
        authorization_output = regeneration_output or output
        regeneration_report = load_pending_staleness_report(authorization_output)
        if regeneration_report["live_bundle_hash"] != bundle_hash(
            authorization_output, include_state=True
        ):
            raise ValidationFailure("Regeneration Impact Report no longer matches the live Bundle")
        if regeneration_report["source_set_hash"] != source_hash:
            raise ValidationFailure("Regeneration Impact Report no longer matches the source set")
        try:
            report_generation = GenerationIdentity.model_validate(regeneration_report["generation"])
        except (TypeError, ValueError) as exc:
            raise ValidationFailure(
                "Invalid Generation Identity in Regeneration Impact Report"
            ) from exc
        if not _same_generation_request(report_generation, identity):
            raise ValidationFailure(
                "Regeneration Impact Report no longer matches the requested Generation Identity"
            )
    else:
        stale_concepts = detect_staleness(output, state, sources)
        planning_stale = source_hash != state.source_set_hash
        generation_stale = (
            sorted(
                concept_id
                for concept_id, concept_state in state.concepts.items()
                if concept_state.ownership == "agent" and not concept_state.deleted
            )
            if not _same_generation_request(identity, state.generation)
            else []
        )
        if stale_concepts or planning_stale or generation_stale:
            raise StalenessDetected(
                str(
                    write_staleness_report(
                        output,
                        stale_concepts,
                        planning_stale=planning_stale,
                        generation_stale=generation_stale,
                        live_bundle_hash=bundle_hash(output, include_state=True),
                        source_set_hash=source_hash,
                        generation=identity,
                    )
                )
            )
    report("Evaluating deterministic no-change conditions...")
    with processing_phase(timing, "No-change evaluation"):
        current = public_concepts(output)
        source_unchanged = source_hash == state.source_set_hash
        generation_unchanged = _same_generation_request(identity, state.generation)
        no_changes = (
            source_unchanged and generation_unchanged and bundle_hash(output) == state.bundle_hash
        )
    if no_changes:
        report("The source set and bundle are unchanged.")
        return False

    ownership = _register_human_concepts(current, state)
    deleted = {
        concept_id: item.deletion_candidate_hash or ""
        for concept_id, item in state.concepts.items()
        if item.deleted
    }
    if source_unchanged and generation_unchanged and not regenerate_all:
        report("Reusing the previous agent baseline; source evidence is unchanged.")
        with processing_phase(timing, "Agent Baseline reuse"):
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
            output=output,
            progress=progress,
            timing=timing,
        )
        identity.output_language = output_language

    report("Merging agent candidates and detecting Reconciliation Conflicts...")
    with processing_phase(timing, "Agent candidate merge and conflict detection"):
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
                                "A human deleted an agent-owned Concept that still has source "
                                "support."
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
                        _override_matches(override, conflict, human_raw)
                        for override in state.overrides
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
        with processing_phase(timing, "Candidate Bundle writing and validation"):
            _write_bundle(
                staging,
                concepts=published,
                baselines=baselines,
                ownership=ownership,
                sources=sources,
                generation=identity,
                previous_state=state,
                action="Update",
                log_detail=(
                    f"Reconciled {len(published)} Concepts from {len(sources)} PDF sources."
                ),
                deleted=deleted,
            )
        if conflicts:
            report("Conflicts require human reconciliation; preserving the current bundle.")
            with processing_phase(timing, "Reconciliation artifact writing"):
                _write_reconciliation(output, staging, state, sources, identity, conflicts)
            raise ReconciliationRequired(str(output.parent / f"{output.name}.reconciliation.md"))
        report("Publishing the bundle atomically...")
        with processing_phase(timing, "Atomic publication"):
            if regeneration_report is not None and resolve_regeneration:
                with prepared_staleness_resolution(
                    regeneration_output or output, regeneration_report
                ):
                    publish_staging(staging, output)
            else:
                publish_staging(staging, output)
    return True


def _expand_pages(values: list[str]) -> list[int]:
    """Expand rendered PDF Source page ranges for reconciliation evidence."""

    from .okf import expand_ranges

    return expand_ranges(values)


def _write_reconciliation(
    output: Path,
    staging: Path,
    prior_state: ForgeState,
    sources: list[KnowledgeSource],
    identity: GenerationIdentity,
    conflicts: list[Conflict],
) -> None:
    """Generate pending, manual, manifest, and report artifacts for conflicts."""

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
    """Format a conflict value as an indented, human-readable report block."""

    if value is None:
        return "_(deleted or absent)_"
    return "\n".join(f"    {line}" for line in value.splitlines()) or "_(empty)_"


def _set_conflict_value(raw: str, conflict: Conflict, value: str | None) -> str:
    """Replace or remove one frontmatter or structural Markdown conflict block."""

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
    """Apply a complete resolution file and atomically publish the pending Bundle."""

    source, output = resolve_disjoint_trees(source, output)
    reject_tracing()
    with output_lock(output.resolve()):
        _reconcile_locked(source=source, output=output, resolution_path=resolution_path)


def _reconcile_locked(*, source: Path, output: Path, resolution_path: Path) -> None:
    """Validate a reconciliation snapshot, apply choices, and publish its resolved Bundle."""

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
    """Record a human verification event for the current version of one Concept."""

    source, output = resolve_disjoint_trees(source, output)
    reject_tracing()
    with output_lock(output.resolve()):
        _verify_locked(source=source, output=output, concept_id=concept_id, actor=actor)


def _verify_locked(*, source: Path, output: Path, concept_id: str, actor: str) -> None:
    """Verify one Concept version while the caller holds the output lock."""

    if not actor.startswith("human:") or len(actor) <= len("human:"):
        raise ValidationFailure("--by must use the actor form human:<id>")
    if migration_available(output):
        with staged_bundle(output, copy_existing=True) as staging:
            migrate_bundle(staging)
            _verify_locked(source=source, output=staging, concept_id=concept_id, actor=actor)
            publish_staging(staging, output)
            return
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
