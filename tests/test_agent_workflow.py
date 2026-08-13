from __future__ import annotations

import json
from pathlib import Path

import httpx
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


class SourceContractFakeAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, inputs: dict[str, object], _: dict[str, object]) -> dict[str, object]:
        prompt = str(inputs)
        self.prompts.append(prompt)
        source_id = "policy.pdf" if "Valid source IDs and page bounds" in prompt else "_none"
        return {
            "messages": [AIMessage(content="structured")],
            "structured_response": {
                "slug": "deadlocks",
                "title": "Deadlocks",
                "type": "Concept",
                "description": "Deadlock behavior.",
                "body": (
                    "# Deadlocks\n\nTransactions can deadlock.[^policy.pdf@p1]\n\n"
                    "[^policy.pdf@p1]: Policy, page 1"
                ),
                "evidence": [{"source_id": source_id, "pages": [1]}],
            },
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


def test_synthesis_rejects_source_sentinels_and_supplies_valid_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = SourceContractFakeAgent()
    monkeypatch.setattr(agent_module, "create_agent", lambda **_: fake)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url=None,
            max_steps=5,
        )
        draft = agent.synthesize(
            PlannedConcept(
                slug="deadlocks",
                title="Deadlocks",
                type="Concept",
                description="Deadlock behavior.",
                search_queries=["deadlocks"],
            ),
            "English",
        )

    assert draft.evidence == [Evidence(source_id="policy.pdf", pages=[1])]
    assert "Valid source IDs and page bounds" in fake.prompts[0]
    assert "_none" not in fake.prompts[0]


def test_synthesis_repair_names_valid_source_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = {
        "slug": "deadlocks",
        "title": "Deadlocks",
        "type": "Concept",
        "description": "Deadlock behavior.",
        "body": "# Deadlocks\n\nUnsupported draft.",
        "evidence": [{"source_id": "_none", "pages": [1]}],
    }
    valid = {
        **invalid,
        "body": (
            "# Deadlocks\n\nSupported draft.[^policy.pdf@p1]\n\n[^policy.pdf@p1]: Policy, page 1"
        ),
        "evidence": [{"source_id": "policy.pdf", "pages": [1]}],
    }
    fake = FakeCompiledAgent([invalid, valid])
    monkeypatch.setattr(agent_module, "create_agent", lambda **_: fake)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url=None,
            max_steps=5,
        )
        draft = agent.synthesize(
            PlannedConcept(
                slug="deadlocks",
                title="Deadlocks",
                type="Concept",
                description="Deadlock behavior.",
                search_queries=["deadlocks"],
            ),
            "English",
        )

    assert draft.evidence[0].source_id == "policy.pdf"
    assert "valid source IDs are" in fake.prompts[1]
    assert "policy.pdf" in fake.prompts[1]
    assert "Placeholder or sentinel source IDs are forbidden" in fake.prompts[1]


def test_synthesis_repairs_invalid_citation_source_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = {
        "slug": "deadlocks",
        "title": "Deadlocks",
        "type": "Concept",
        "description": "Deadlock behavior.",
        "body": "# Deadlocks\n\nTransactions can deadlock.[^hpm@p1]\n\n[^hpm@p1]: HPM, page 1",
        "evidence": [{"source_id": "policy.pdf", "pages": [1]}],
    }
    valid = {
        **invalid,
        "body": (
            "# Deadlocks\n\nTransactions can deadlock.[^policy.pdf@p1]\n\n"
            "[^policy.pdf@p1]: Policy, page 1"
        ),
    }
    fake = FakeCompiledAgent([invalid, valid])
    monkeypatch.setattr(agent_module, "create_agent", lambda **_: fake)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url=None,
            max_steps=5,
        )
        draft = agent.synthesize(
            PlannedConcept(
                slug="deadlocks",
                title="Deadlocks",
                type="Concept",
                description="Deadlock behavior.",
                search_queries=["deadlocks"],
            ),
            "English",
        )

    assert "[^policy.pdf@p1]" in draft.body
    assert "citation references missing source hpm" in fake.prompts[1]
    assert "never abbreviate a filename or replace it with initials" in fake.prompts[0]


def test_reasoning_agent_forces_responses_api_without_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_model(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_model)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url="https://models.example/v1",
            max_steps=5,
        )

    assert captured["use_responses_api"] is True
    assert captured["store"] is False
    assert captured["model_kwargs"] == {"parallel_tool_calls": True}


def _responses_payload(output: list[dict[str, object]], response_id: str) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "fake",
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": 0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }


def _function_call(name: str, call_id: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def test_agent_serializes_tool_calls_when_bedrock_gateway_ignores_parallel_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        inputs = payload["input"]
        function_calls = [item for item in inputs if item.get("type") == "function_call"]
        function_outputs = [item for item in inputs if item.get("type") == "function_call_output"]
        if not function_calls:
            calls = [
                _function_call("search_pages", "call_search", {"query": "refunds"}),
                _function_call(
                    "read_pages", "call_read", {"source_id": "policy.pdf", "pages": [1]}
                ),
            ]
            return httpx.Response(200, json=_responses_payload(calls, f"resp_{len(requests)}"))
        if len(function_calls) > 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "[kiro] Bedrock error message: Expected toolResult blocks at "
                            "messages.2.content for the following Ids: tooluse_test"
                        ),
                        "type": "provider_api_error",
                    }
                },
            )
        assert function_outputs[0]["call_id"] == function_calls[0]["call_id"]
        plan = {
            "language": "English",
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
        return httpx.Response(
            200,
            json=_responses_payload(
                [_function_call("ConceptPlan", "call_plan", plan)], f"resp_{len(requests)}"
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    real_model = agent_module.ChatOpenAI

    def fake_model(**kwargs: object) -> object:
        return real_model(**kwargs, http_client=client)

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_model)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url="https://models.example/v1",
            max_steps=10,
            parallel_tool_calls=False,
        )
        plan = agent.plan("auto", [])

    assert plan.concepts[0].slug == "refund-policy"
    assert all(request["parallel_tool_calls"] is False for request in requests)
    replay = requests[1]["input"]
    assert [item["call_id"] for item in replay if item.get("type") == "function_call"] == [
        "call_search"
    ]
    assert [item["call_id"] for item in replay if item.get("type") == "function_call_output"] == [
        "call_search"
    ]


def test_agent_replays_every_parallel_tool_call_and_matching_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        inputs = payload["input"]
        function_calls = [item for item in inputs if item.get("type") == "function_call"]
        if not function_calls:
            return httpx.Response(
                200,
                json=_responses_payload(
                    [
                        _function_call("search_pages", "call_search", {"query": "refunds"}),
                        _function_call(
                            "read_pages", "call_read", {"source_id": "policy.pdf", "pages": [1]}
                        ),
                    ],
                    "resp_parallel_tools",
                ),
            )

        plan = {
            "language": "English",
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
        return httpx.Response(
            200,
            json=_responses_payload(
                [_function_call("ConceptPlan", "call_plan", plan)], "resp_parallel_plan"
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    real_model = agent_module.ChatOpenAI

    def fake_model(**kwargs: object) -> object:
        return real_model(**kwargs, http_client=client)

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_model)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url="https://models.example/v1",
            max_steps=10,
        )
        plan = agent.plan("auto", [])

    assert plan.concepts[0].slug == "refund-policy"
    assert all(request["parallel_tool_calls"] is True for request in requests)
    replay = requests[1]["input"]
    assert [item["call_id"] for item in replay if item.get("type") == "function_call"] == [
        "call_search",
        "call_read",
    ]
    assert [item["call_id"] for item in replay if item.get("type") == "function_call_output"] == [
        "call_search",
        "call_read",
    ]


def test_agent_recovers_from_invalid_search_query_with_langchain_middleware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_responses_payload(
                    [
                        _function_call(
                            "search_pages",
                            "call_invalid_search",
                            {"query": "storage engine API low-level functions"},
                        )
                    ],
                    "resp_invalid_search",
                ),
            )

        function_outputs = [
            item for item in payload["input"] if item.get("type") == "function_call_output"
        ]
        latest_output = function_outputs[-1]
        if len(requests) == 2:
            assert latest_output["call_id"] == "call_invalid_search"
            assert "Retry with plain words" in str(latest_output["output"])
            return httpx.Response(
                200,
                json=_responses_payload(
                    [
                        _function_call(
                            "search_pages",
                            "call_repaired_search",
                            {"query": "storage engine API low level functions"},
                        )
                    ],
                    "resp_repaired_search",
                ),
            )

        assert latest_output["call_id"] == "call_repaired_search"
        plan = {
            "language": "English",
            "concepts": [
                {
                    "slug": "storage-engine-api",
                    "title": "Storage engine API",
                    "type": "Concept",
                    "description": "Low-level storage functions.",
                    "search_queries": ["storage engine API low level functions"],
                }
            ],
        }
        return httpx.Response(
            200,
            json=_responses_payload(
                [_function_call("ConceptPlan", "call_plan", plan)], "resp_plan"
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    real_model = agent_module.ChatOpenAI

    def fake_model(**kwargs: object) -> object:
        return real_model(**kwargs, http_client=client)

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_model)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url="https://models.example/v1",
            max_steps=10,
        )
        plan = agent.plan("auto", [])

    assert plan.concepts[0].slug == "storage-engine-api"
    assert len(requests) == 3


def test_agent_does_not_retry_provider_api_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "endpoint rejected the tool sequence",
                    "type": "provider_api_error",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    real_model = agent_module.ChatOpenAI

    def fake_model(**kwargs: object) -> object:
        return real_model(**kwargs, http_client=client)

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_model)
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([pdf()])
        agent = ReasoningAgent(
            index=index,
            sources=[pdf()],
            model="fake",
            api_key="secret",
            base_url="https://models.example/v1",
            max_steps=10,
        )
        with pytest.raises(ValidationFailure, match="Model request failed"):
            agent.plan("auto", [])

    assert request_count == 1


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


class IsolatedSessionAgent:
    def __init__(self, session: int, calls: list[tuple[int, str]]) -> None:
        self.session = session
        self.calls = calls
        self.remaining_steps = 1

    def consume_full_budget(self, task: str) -> None:
        if self.remaining_steps == 0:
            raise ValidationFailure("Agent step budget exceeded (1 model call)")
        self.remaining_steps -= 1
        self.calls.append((self.session, task))

    def plan(self, language: str, existing_ids: list[str]) -> ConceptPlan:
        self.consume_full_budget("plan")
        return ConceptPlan(
            language="English",
            concepts=[
                PlannedConcept(
                    slug=slug,
                    title=slug.replace("-", " ").title(),
                    type="Concept",
                    description=f"Rules for {slug}.",
                    search_queries=[slug],
                )
                for slug in ("alpha", "beta", "gamma")
            ],
        )

    def synthesize(self, concept: PlannedConcept, language: str) -> ConceptDraft:
        self.consume_full_budget(concept.slug)
        return ConceptDraft(
            slug=concept.slug,
            title=concept.title,
            type=concept.type,
            description=concept.description,
            body=(
                f"# Rule\n\n{concept.title}.[^policy.pdf@p1]\n\n"
                "[^policy.pdf@p1]: Refund policy, page 1"
            ),
            evidence=[Evidence(source_id="policy.pdf", pages=[1])],
        )


def test_every_workflow_task_can_use_a_full_isolated_reasoning_budget() -> None:
    sessions: list[IsolatedSessionAgent] = []
    calls: list[tuple[int, str]] = []

    def agent_factory() -> IsolatedSessionAgent:
        agent = IsolatedSessionAgent(len(sessions) + 1, calls)
        sessions.append(agent)
        return agent

    graph = build_workflow(agent_factory, [pdf()], "knowledge-forge/fake")
    result = graph.invoke({"language": "auto", "existing_ids": []})

    assert calls == [(1, "plan"), (2, "alpha"), (3, "beta"), (4, "gamma")]
    assert list(result["concepts"]) == [
        "concepts/alpha",
        "concepts/beta",
        "concepts/gamma",
    ]


def test_workflow_identifies_the_concept_whose_reasoning_budget_failed() -> None:
    calls: list[tuple[int, str]] = []

    class FailingSessionAgent(IsolatedSessionAgent):
        def synthesize(self, concept: PlannedConcept, language: str) -> ConceptDraft:
            if concept.slug == "beta":
                raise ValidationFailure("Agent step budget exceeded (2 model calls)")
            return super().synthesize(concept, language)

    session = 0

    def agent_factory() -> FailingSessionAgent:
        nonlocal session
        session += 1
        return FailingSessionAgent(session, calls)

    graph = build_workflow(agent_factory, [pdf()], "knowledge-forge/fake")

    with pytest.raises(
        ValidationFailure,
        match=r"Concept synthesis failed for concepts/beta: Agent step budget exceeded",
    ):
        graph.invoke({"language": "auto", "existing_ids": []})

    assert calls == [(1, "plan"), (2, "alpha")]


def test_workflow_identifies_a_planning_reasoning_budget_failure() -> None:
    class FailingPlanningAgent:
        def plan(self, language: str, existing_ids: list[str]) -> ConceptPlan:
            raise ValidationFailure("Agent step budget exceeded (2 model calls)")

    graph = build_workflow(FailingPlanningAgent, [pdf()], "knowledge-forge/fake")

    with pytest.raises(
        ValidationFailure,
        match=r"Concept planning failed: Agent step budget exceeded",
    ):
        graph.invoke({"language": "auto", "existing_ids": []})


def test_langgraph_workflow_returns_valid_concepts() -> None:
    progress: list[str] = []
    graph = build_workflow(
        FakeReasoningAgent, [pdf()], "knowledge-forge/fake", progress=progress.append
    )
    result = graph.invoke({"language": "auto", "existing_ids": []})
    assert result["language"] == "English"
    assert "concepts/refund-policy" in result["concepts"]
    assert progress == [
        "Planning concepts with the reasoning agent...",
        "Planned 1 concepts in English.",
        "Synthesizing concept 1/1: refund-policy",
        "Rendering and validating 1 concepts...",
        "Agent-generated concepts passed validation.",
    ]
