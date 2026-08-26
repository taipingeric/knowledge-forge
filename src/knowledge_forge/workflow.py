from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
from time import monotonic
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import ReasoningAgent, TokenUsage
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
    usage_lock = Lock()
    workflow_usage = TokenUsage()
    node_usage: dict[str, TokenUsage] = {}
    workflow_started_at: float | None = None
    workflow_reported = False

    def report(message: str) -> None:
        """Send one workflow progress message through the configured reporter."""

        reporter.report(message)

    def agent_usage(agent: object | None) -> TokenUsage | None:
        """Read usage from a reasoning agent without constraining workflow test doubles."""

        usage = getattr(agent, "token_usage", None)
        return usage if isinstance(usage, TokenUsage) else None

    def record_usage(node: str, usage: TokenUsage | None) -> None:
        """Aggregate usage from one agent into a node and workflow ledger."""

        nonlocal workflow_usage
        if usage is None:
            return
        with usage_lock:
            node_usage[node] = node_usage.get(node, TokenUsage()).add(usage)
            workflow_usage = workflow_usage.add(usage)

    def usage_text(usage: TokenUsage | None) -> str:
        """Format node usage without inventing token counts unavailable from the provider."""

        if usage is None or (
            usage.calls == 0
            and usage.input_tokens is None
            and usage.output_tokens is None
            and usage.total_tokens is None
        ):
            return (
                "model_calls=unavailable; input_tokens=unavailable; "
                "output_tokens=unavailable; total_tokens=unavailable"
            )
        return (
            f"model_calls={usage.calls}; "
            f"input_tokens={format_count(usage.input_tokens, usage.input_unknown_calls)}; "
            f"output_tokens={format_count(usage.output_tokens, usage.output_unknown_calls)}; "
            f"total_tokens={format_count(usage.total_tokens, usage.total_unknown_calls)}"
        )

    def format_count(value: int | None, unknown_calls: int | None) -> str:
        """Format one optional aggregate token count."""

        if value is None:
            return "unavailable"
        return f"{value} (partial)" if unknown_calls else str(value)

    def report_node(
        name: str, started_at: float, usage: TokenUsage | None, succeeded: bool
    ) -> None:
        """Report one LangGraph node's duration and model usage."""

        status = "completed" if succeeded else "failed"
        report(
            f"LangGraph node {name} {status} in {monotonic() - started_at:.3f}s; "
            f"{usage_text(usage)}."
        )

    def start_workflow() -> None:
        """Start the workflow timer at the first LangGraph node execution."""

        nonlocal workflow_started_at
        if workflow_started_at is None:
            workflow_started_at = monotonic()

    def report_workflow(succeeded: bool) -> None:
        """Report total workflow duration and aggregate model usage once."""

        nonlocal workflow_reported
        if workflow_reported:
            return
        workflow_reported = True
        status = "completed" if succeeded else "failed"
        started_at = workflow_started_at if workflow_started_at is not None else monotonic()
        with usage_lock:
            usage = workflow_usage
        report(
            f"LangGraph workflow {status} in {monotonic() - started_at:.3f}s; {usage_text(usage)}."
        )

    def plan(state: WorkflowState) -> dict[str, object]:
        """Plan the Concepts and validate the requested output language."""

        start_workflow()
        started_at = monotonic()
        agent: object | None = None
        succeeded = False
        try:
            report("Planning concepts with the reasoning agent...")
            agent = agent_factory()
            with processing_phase(timing, "Concept planning"):
                try:
                    result = agent.plan(state["language"], state.get("existing_ids", []))
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
            succeeded = True
            return {"plan": result, "language": result.language}
        finally:
            usage = agent_usage(agent)
            record_usage("plan", usage)
            report_node("plan", started_at, usage, succeeded)
            if not succeeded:
                report_workflow(False)

    def synthesize(state: WorkflowState) -> dict[str, object]:
        """Synthesize all planned Concepts with bounded concurrency."""

        planned = state["plan"].concepts
        total = len(planned)
        started_at = monotonic()
        succeeded = False

        def synthesize_one(current: int) -> ConceptDraft:
            """Synthesize one Concept and wrap failures with its Concept ID."""

            concept = planned[current - 1]
            agent: object | None = None
            task_started_at = monotonic()
            task_succeeded = False
            report(f"Synthesizing concept {current}/{len(planned)}: {concept.slug}")
            try:
                agent = agent_factory()
                with processing_phase(
                    timing, f"Concept synthesis {current}/{total} ({concept.slug})"
                ):
                    try:
                        draft = agent.synthesize(concept, state["language"])
                    except ValidationFailure as exc:
                        raise ValidationFailure(
                            f"Concept synthesis failed for concepts/{concept.slug}: {exc}"
                        ) from exc
                task_succeeded = True
                return draft
            finally:
                usage = agent_usage(agent)
                record_usage("synthesize", usage)
                report_node(f"synthesize[{concept.slug}]", task_started_at, usage, task_succeeded)

        drafts: dict[int, ConceptDraft] = {}
        try:
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
                succeeded = True
                return {"drafts": [drafts[current] for current in range(1, total + 1)]}
        finally:
            with usage_lock:
                aggregate_usage = node_usage.get("synthesize")
            report_node("synthesize", started_at, aggregate_usage, succeeded)
            if not succeeded:
                report_workflow(False)

    def render_and_validate(state: WorkflowState) -> dict[str, object]:
        """Render every draft and reject the workflow if any Concept is invalid."""

        started_at = monotonic()
        succeeded = False
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

        try:
            with processing_phase(timing, "Concept rendering and validation"):
                concepts = render_valid_concepts()
            report("Agent-generated concepts passed validation.")
            succeeded = True
            return {"concepts": concepts}
        finally:
            zero_usage = TokenUsage(calls=0, input_tokens=0, output_tokens=0, total_tokens=0)
            report_node("render_and_validate", started_at, zero_usage, succeeded)
            report_workflow(succeeded)

    graph = StateGraph(WorkflowState)
    graph.add_node("plan", plan)
    graph.add_node("synthesize", synthesize)
    graph.add_node("render_and_validate", render_and_validate)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "synthesize")
    graph.add_edge("synthesize", "render_and_validate")
    graph.add_edge("render_and_validate", END)
    return graph.compile()
