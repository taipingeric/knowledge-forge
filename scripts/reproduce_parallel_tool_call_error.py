#!/usr/bin/env python3
"""Reproduce Kiro's parallel Responses tool-result conversion error.

The script intentionally asks the model for several tool calls in one response,
then replays every response output item plus one correctly matched result per
call. It never prints the API key or request headers.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://agw.playground.straitsx.ai/kiro/openai/v1/responses"
# Replace this placeholder before running the standalone reproducer.
API_KEY = "API_KEY"


QUERIES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "search_pages",
            "description": "Search indexed document pages for one query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The single term to search for.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    ]


def build_prompt(count: int) -> str:
    queries = ", ".join(QUERIES[:count])
    return (
        f"Search separately for every one of these terms: {queries}. Call search_pages "
        f"exactly {count} times in the same response, once per term. Do not combine terms "
        "and do not answer until every tool result has been returned."
    )


class ResponseFailure(Exception):
    """An HTTP response from the gateway was not successful."""

    def __init__(self, status: int, body: dict[str, Any] | str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


def _parse_body(raw: bytes) -> dict[str, Any] | str:
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed if isinstance(parsed, dict) else text


def response_tool_ids(body: dict[str, Any] | str) -> list[str]:
    """Return every function-call ID exposed by one Responses API response."""
    if not isinstance(body, dict):
        return []
    output = body.get("output")
    if not isinstance(output, list):
        return []
    return [
        call_id
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
        if isinstance(call_id := item.get("call_id"), str) and call_id
    ]


def print_tool_ids(label: str, tool_ids: list[str]) -> None:
    """Print indexed tool IDs in a stable layout for line-by-line comparison."""
    print(f"{label} ({len(tool_ids)}):", flush=True)
    if not tool_ids:
        print("  <none>", flush=True)
        return
    width = len(str(len(tool_ids)))
    for position, tool_id in enumerate(tool_ids, start=1):
        print(f"  [{position:>{width}}] {tool_id}", flush=True)


def print_response_tool_ids(body: dict[str, Any] | str) -> None:
    """Log tool IDs for every HTTP response without exposing arguments or credentials."""
    print_tool_ids("Response tool call IDs", response_tool_ids(body))


def post_response(payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoint
            body = _parse_body(response.read())
            # Print IDs at the HTTP seam so no successful response path can omit them.
            print_response_tool_ids(body)
            if not isinstance(body, dict):
                raise ResponseFailure(response.status, body)
            return body
    except HTTPError as exc:
        body = _parse_body(exc.read())
        # Error responses are responses too; report any tool IDs they happen to contain.
        print_response_tool_ids(body)
        raise ResponseFailure(exc.code, body) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach endpoint: {exc.reason}") from exc


def function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        raise RuntimeError("First response has no output item list")
    return [
        item for item in output if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def output_shape(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    summary: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            summary.append({"type": type(item).__name__})
            continue
        content = item.get("content")
        content_types = (
            [part.get("type", "<missing>") for part in content if isinstance(part, dict)]
            if isinstance(content, list)
            else []
        )
        summary.append(
            {
                "type": item.get("type", "<missing>"),
                "status": item.get("status", "<missing>"),
                "content_types": content_types,
            }
        )
    return summary


def tool_outputs(calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for call in calls:
        call_id = call.get("call_id")
        name = call.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError(f"Function call has no call_id: {name!r}")
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"value": f"result for {name}"}),
            }
        )
    return outputs


def error_message(body: dict[str, Any] | str) -> str:
    if not isinstance(body, dict):
        return body
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Reproduce the Kiro Responses-to-Bedrock failure for parallel function calls.")
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Responses model name.",
    )
    parser.add_argument(
        "--tool-count",
        type=int,
        choices=range(2, 9),
        default=5,
        metavar="2-8",
        help="Number of parallel calls to request; defaults to 5.",
    )
    parser.add_argument(
        "--max-round-1-attempts",
        type=int,
        choices=range(1, 11),
        default=3,
        metavar="1-10",
        help="Retry Round 1 when the model returns too few calls; defaults to 3.",
    )
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = API_KEY.strip()
    if not api_key or api_key == "API_KEY":
        print(
            "ERROR: Replace the API_KEY placeholder at the top of this script.",
            file=sys.stderr,
        )
        return 1

    tools = build_tools()
    user_item = {"role": "user", "content": build_prompt(args.tool_count)}
    first_payload = {
        "model": args.model,
        "input": [user_item],
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "store": False,
    }

    print(f"POST {ENDPOINT}", flush=True)
    first: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    for attempt in range(1, args.max_round_1_attempts + 1):
        print(
            f"Round 1 attempt {attempt}/{args.max_round_1_attempts}: requesting "
            f"{args.tool_count} parallel function calls...",
            flush=True,
        )
        try:
            first = post_response(first_payload, api_key, args.timeout)
        except (ResponseFailure, RuntimeError) as exc:
            print(f"ERROR: First request failed: {exc}", file=sys.stderr)
            return 1

        calls = function_calls(first)
        print_tool_ids(
            "Round 1 function call IDs",
            [str(call.get("call_id", "<missing>")) for call in calls],
        )
        if len(calls) >= args.tool_count:
            break
        print(f"Round 1 output shape: {json.dumps(output_shape(first))}")
    else:
        print(
            f"NOT REPRODUCED: The model returned fewer than {args.tool_count} function calls "
            f"in all {args.max_round_1_attempts} attempts.",
            file=sys.stderr,
        )
        return 2

    response_output = first.get("output")
    assert isinstance(response_output, list)
    outputs = tool_outputs(calls)
    second_input = [user_item, *response_output, *outputs]
    second_payload = {
        "model": args.model,
        "input": second_input,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
    }
    print("Round 2: replaying every response output item and matching tool results.")
    print_tool_ids("Round 2 tool result IDs", [output["call_id"] for output in outputs])

    try:
        second = post_response(second_payload, api_key, args.timeout)
    except ResponseFailure as exc:
        message = error_message(exc.body)
        print(f"Round 2 HTTP status: {exc.status}")
        print(f"Round 2 error: {message}")
        if exc.status == 400 and "Expected toolResult blocks" in message:
            print("REPRODUCED: Kiro/Bedrock rejected correctly matched parallel tool results.")
            return 0
        print("FAILED WITH A DIFFERENT ERROR.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: Second request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Round 2 response status: {second.get('status', '<missing>')}")
    print("NOT REPRODUCED: The gateway accepted the parallel tool results.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
