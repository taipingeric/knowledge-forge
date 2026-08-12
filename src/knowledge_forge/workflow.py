from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import ReasoningAgent
from .errors import ValidationFailure
from .models import ConceptDraft, ConceptPlan, PDFSource
from .okf import render_concept, validate_concept


class WorkflowState(TypedDict, total=False):
    language: str
    existing_ids: list[str]
    plan: ConceptPlan
    drafts: list[ConceptDraft]
    concepts: dict[str, str]


def build_workflow(agent: ReasoningAgent, sources: list[PDFSource], actor: str):
    source_map = {source.id: source for source in sources}

    def plan(state: WorkflowState) -> dict[str, object]:
        result = agent.plan(state["language"], state.get("existing_ids", []))
        if result.language.casefold() == "auto":
            raise ValidationFailure("Agent must resolve auto to one concrete Bundle language")
        if state["language"].casefold() != "auto" and result.language != state["language"]:
            raise ValidationFailure(
                f"Agent returned language {result.language!r}, expected {state['language']!r}"
            )
        slugs = [concept.slug for concept in result.concepts]
        if len(slugs) != len(set(slugs)):
            raise ValidationFailure("Agent planned duplicate Concept slugs")
        return {"plan": result, "language": result.language}

    def synthesize(state: WorkflowState) -> dict[str, object]:
        drafts = [
            agent.synthesize(concept, state["language"]) for concept in state["plan"].concepts
        ]
        return {"drafts": drafts}

    def render_and_validate(state: WorkflowState) -> dict[str, object]:
        concepts: dict[str, str] = {}
        errors: list[str] = []
        page_counts = {source.id: len(source.pages) for source in sources}
        for draft in state["drafts"]:
            concept_id = f"concepts/{draft.slug}"
            raw = render_concept(draft, source_map, actor)
            errors.extend(validate_concept(raw, concept_id, page_counts))
            concepts[concept_id] = raw
        if errors:
            raise ValidationFailure("Generated Concepts are invalid:\n- " + "\n- ".join(errors))
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
