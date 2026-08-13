from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "reproduce_parallel_tool_call_error.py")
)


def test_reproducer_matches_every_parallel_call_id_with_an_output() -> None:
    calls = [
        {
            "type": "function_call",
            "name": "lookup_alpha",
            "call_id": "tooluse_alpha",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "name": "lookup_beta",
            "call_id": "tooluse_beta",
            "arguments": "{}",
        },
    ]

    outputs = SCRIPT["tool_outputs"](calls)

    assert [item["call_id"] for item in outputs] == ["tooluse_alpha", "tooluse_beta"]
    assert all(item["type"] == "function_call_output" for item in outputs)


def test_reproducer_uses_one_search_tool_with_a_required_query() -> None:
    tools = SCRIPT["build_tools"]()

    assert [tool["name"] for tool in tools] == ["search_pages"]
    assert tools[0]["parameters"]["required"] == ["query"]


def test_reproducer_prompt_requests_repeated_calls_to_the_same_tool() -> None:
    prompt = SCRIPT["build_prompt"](5)

    assert "search_pages exactly 5 times" in prompt
    assert "alpha, beta, gamma, delta, epsilon" in prompt


def test_reproducer_uses_an_explicit_api_key_placeholder() -> None:
    assert SCRIPT["API_KEY"] == "API_KEY"


def test_output_shape_reports_types_without_message_text() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "do not leak me"}],
            }
        ]
    }

    assert SCRIPT["output_shape"](response) == [
        {
            "type": "message",
            "status": "completed",
            "content_types": ["output_text"],
        }
    ]


def test_response_tool_ids_reports_every_function_call_id() -> None:
    response = {
        "output": [
            {"type": "function_call", "call_id": "tooluse_alpha"},
            {"type": "message", "content": []},
            {"type": "function_call", "call_id": "tooluse_beta"},
        ]
    }

    assert SCRIPT["response_tool_ids"](response) == ["tooluse_alpha", "tooluse_beta"]


def test_print_response_tool_ids_aligns_one_id_per_line(capsys) -> None:
    response = {
        "output": [
            *[
                {"type": "function_call", "call_id": f"tooluse_{position}"}
                for position in range(1, 11)
            ]
        ]
    }

    SCRIPT["print_response_tool_ids"](response)

    assert capsys.readouterr().out.splitlines() == [
        "Response tool call IDs (10):",
        "  [ 1] tooluse_1",
        "  [ 2] tooluse_2",
        "  [ 3] tooluse_3",
        "  [ 4] tooluse_4",
        "  [ 5] tooluse_5",
        "  [ 6] tooluse_6",
        "  [ 7] tooluse_7",
        "  [ 8] tooluse_8",
        "  [ 9] tooluse_9",
        "  [10] tooluse_10",
    ]


def test_print_response_tool_ids_reports_an_empty_list_when_response_has_no_calls(
    capsys,
) -> None:
    SCRIPT["print_response_tool_ids"]({"error": {"message": "failed"}})

    assert capsys.readouterr().out.splitlines() == ["Response tool call IDs (0):", "  <none>"]


def test_print_tool_ids_uses_the_same_comparable_layout_for_each_round(capsys) -> None:
    tool_ids = ["tooluse_alpha", "tooluse_beta"]

    SCRIPT["print_tool_ids"]("Round 1 function call IDs", tool_ids)
    SCRIPT["print_tool_ids"]("Round 2 tool result IDs", tool_ids)

    assert capsys.readouterr().out.splitlines() == [
        "Round 1 function call IDs (2):",
        "  [1] tooluse_alpha",
        "  [2] tooluse_beta",
        "Round 2 tool result IDs (2):",
        "  [1] tooluse_alpha",
        "  [2] tooluse_beta",
    ]
