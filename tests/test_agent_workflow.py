from __future__ import annotations

from pathlib import Path

import pytest
from langchain.messages import AIMessage

from knowledge_forge import agent as agent_module
from knowledge_forge.agent import ReasoningAgent
from knowledge_forge.errors import ValidationFailure
from knowledge_forge.models import (
    ConceptDraft,
    ConceptPlan,
    Evidence,
    PDFSource,
    PlannedConcept,
    SourcePage,
)
from knowledge_forge.sources import PageIndex, logical_resource, sha256_text
from knowledge_forge.workflow import build_workflow


def pdf() -> PDFSource:
    return PDFSource(
        id="policy.pdf",
        resource=logical_resource("policy.pdf"),
        content_sha256=sha256_text("one"),
        pages=[SourcePage(number=1, text="Refunds take seven days.")],
    )


class FakeCompiledAgent:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def invoke(self, inputs: dict[str, object], _: dict[str, object]) -> dict[str, object]:
        self.prompts.append(str(inputs))
        return {
            "messages": [AIMessage(content="structured")],
            "structured_response": self.responses.pop(0),
        }


def test_agent_repairs_semantically_invalid_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = {
        "language": "auto",
        "concepts": [
            {
                "slug": "refund-policy",
                "title": "Refund policy",
                "type": "Policy",
                "description": "Rules.",
                "search_queries": ["refund"],
            }
        ],
    }
    valid = {**invalid, "language": "English"}
    fake = FakeCompiledAgent([invalid, valid])
    monkeypatch.setattr(agent_module, "create_agent", lambda **_: fake)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url="https://models.example/v1",
            max_steps=5,
        )
        plan = agent.plan("auto", [])
    assert plan.language == "English"
    assert agent.steps == 2
    assert "Repair the previous invalid result" in fake.prompts[1]


def test_agent_budget_stops_repairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = {"language": "auto", "concepts": []}
    fake = FakeCompiledAgent([invalid])
    monkeypatch.setattr(agent_module, "create_agent", lambda **_: fake)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url=None,
            max_steps=1,
        )
        with pytest.raises(ValidationFailure, match="budget"):
            agent.plan("auto", [])


class FakeReasoningAgent:
    def plan(self, language: str, existing_ids: list[str]) -> ConceptPlan:
        return ConceptPlan(
            language="English",
            concepts=[
                PlannedConcept(
                    slug="refund-policy",
                    title="Refund policy",
                    type="Policy",
                    description="Rules.",
                    search_queries=["refund"],
                )
            ],
        )

    def synthesize(self, concept: PlannedConcept, language: str) -> ConceptDraft:
        return ConceptDraft(
            slug=concept.slug,
            title=concept.title,
            type=concept.type,
            description=concept.description,
            body=(
                "# Rule\n\nSeven days.[^policy.pdf@p1]\n\n[^policy.pdf@p1]: Refund policy, page 1"
            ),
            evidence=[Evidence(source_id="policy.pdf", pages=[1])],
        )


def test_langgraph_workflow_returns_valid_concepts() -> None:
    graph = build_workflow(FakeReasoningAgent(), [pdf()], "knowledge-forge/fake")
    result = graph.invoke({"language": "auto", "existing_ids": []})
    assert result["language"] == "English"
    assert "concepts/refund-policy" in result["concepts"]
