from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import ReasoningAgent
from .errors import ValidationFailure
from .models import ConceptDraft, ConceptPlan, KnowledgeSource
from .okf import render_concept, validate_concept
from .timing import ProcessingTimer, ProgressReporter, processing_phase


class WorkflowState(TypedDict, total=False):
    """State passed between planning, synthesis, and Concept validation nodes."""

    language: str
    existing_ids: list[str]
    plan: ConceptPlan
    drafts: list[ConceptDraft]
    concepts: dict[str, str]


def build_workflow(
    agent_factory: Callable[[], ReasoningAgent],
    sources: list[KnowledgeSource],
    actor: str,
    concept_concurrency: int = 1,
    progress: Callable[[str], None] | None = None,
    timing: ProcessingTimer | None = None,
):
    """Build the fixed workflow that plans, synthesizes, and validates Concepts."""

    source_map = {source.id: source for source in sources}
    reporter = (
        timing.reporter if timing is not None else ProgressReporter(progress or (lambda _: None))
    )

    def report(message: str) -> None:
        """Send one workflow progress message through the configured reporter."""

        reporter.report(message)

    def plan(state: WorkflowState) -> dict[str, object]:
        """Plan the Concepts and validate the requested output language."""

        report("Planning concepts with the reasoning agent...")
        with processing_phase(timing, "Concept planning"):
            try:
                result = agent_factory().plan(state["language"], state.get("existing_ids", []))
            except ValidationFailure as exc:
                raise ValidationFailure(f"Concept planning failed: {exc}") from exc
        if result.language.casefold() == "auto":
            raise ValidationFailure("Agent must resolve auto to one concrete Bundle language")
        if state["language"].casefold() != "auto" and result.language != state["language"]:
            raise ValidationFailure(
                f"Agent returned language {result.language!r}, expected {state['language']!r}"
            )
        slugs = [concept.slug for concept in result.concepts]
        if len(slugs) != len(set(slugs)):
            raise ValidationFailure("Agent planned duplicate Concept slugs")
        reserved_slugs = {
            concept_id.removeprefix("concepts/") for concept_id in state.get("existing_ids", [])
        }
        collisions = sorted(set(slugs) & reserved_slugs)
        if collisions:
            raise ValidationFailure(
                "Agent planned Concepts that already exist: " + ", ".join(collisions)
            )
        report(f"Planned {len(result.concepts)} concepts in {result.language}.")
        return {"plan": result, "language": result.language}

    def synthesize(state: WorkflowState) -> dict[str, object]:
        """Synthesize all planned Concepts with bounded concurrency."""

        planned = state["plan"].concepts
        total = len(planned)

        def synthesize_one(current: int) -> ConceptDraft:
            """Synthesize one Concept and wrap failures with its Concept ID."""

            concept = planned[current - 1]
            report(f"Synthesizing concept {current}/{len(planned)}: {concept.slug}")
            with processing_phase(timing, f"Concept synthesis {current}/{total} ({concept.slug})"):
                try:
                    return agent_factory().synthesize(concept, state["language"])
                except ValidationFailure as exc:
                    raise ValidationFailure(
                        f"Concept synthesis failed for concepts/{concept.slug}: {exc}"
                    ) from exc

        drafts: dict[int, ConceptDraft] = {}
        with ThreadPoolExecutor(max_workers=concept_concurrency) as executor:
            next_current = 1
            futures: dict[Future[ConceptDraft], int] = {}
            while next_current <= min(concept_concurrency, total):
                futures[executor.submit(synthesize_one, next_current)] = next_current
                next_current += 1
            try:
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        drafts[futures.pop(future)] = future.result()
                    while len(futures) < concept_concurrency and next_current <= total:
                        futures[executor.submit(synthesize_one, next_current)] = next_current
                        next_current += 1
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        return {"drafts": [drafts[current] for current in range(1, total + 1)]}

    def render_and_validate(state: WorkflowState) -> dict[str, object]:
        """Render every draft and reject the workflow if any Concept is invalid."""

        report(f"Rendering and validating {len(state['drafts'])} concepts...")

        def render_valid_concepts() -> dict[str, str]:
            """Render drafts in plan order and aggregate all validation errors."""

            concepts: dict[str, str] = {}
            errors: list[str] = []
            page_counts = {source.id: len(source.evidence) for source in sources}
            for draft in state["drafts"]:
                concept_id = f"concepts/{draft.slug}"
                raw = render_concept(draft, source_map, actor)
                errors.extend(validate_concept(raw, concept_id, page_counts))
                concepts[concept_id] = raw
            if errors:
                raise ValidationFailure("Generated Concepts are invalid:\n- " + "\n- ".join(errors))
            return concepts

        with processing_phase(timing, "Concept rendering and validation"):
            concepts = render_valid_concepts()
        report("Agent-generated concepts passed validation.")
        return {"concepts": concepts}

    graph = StateGraph(WorkflowState)
    graph.add_node("plan", plan)
    graph.add_node("synthesize", synthesize)
    graph.add_node("render_and_validate", render_and_validate)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "synthesize")
    graph.add_edge("synthesize", "render_and_validate")
    graph.add_edge("render_and_validate", END)
    return graph.compile()
