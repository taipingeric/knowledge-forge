from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ValidationFailure
from .models import GenerationIdentity

TRACING_VARIABLES = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)


def resolve_disjoint_trees(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValidationFailure(
            "Resolved --source and --out directory trees must be disjoint: "
            f"source={source}, output={output}"
        )
    return source, output


def reject_tracing() -> None:
    enabled = [
        name
        for name in TRACING_VARIABLES
        if os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}
    ]
    if enabled:
        raise ValidationFailure(
            "Third-party tracing must be disabled because source text may contain sensitive data: "
            + ", ".join(enabled)
        )


def endpoint_identity(base_url: str | None) -> str:
    value = base_url or "https://api.openai.com/v1"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationFailure(f"Invalid OpenAI-compatible base URL: {value}")
    if parsed.username or parsed.password:
        raise ValidationFailure("OpenAI-compatible base URL must not contain credentials")
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def generation_identity(
    *,
    model: str,
    base_url: str | None,
    language: str,
    max_agent_steps: int,
    parallel_tool_calls: bool,
    concept_concurrency: int,
) -> GenerationIdentity:
    if not model.strip():
        raise ValidationFailure("Model must be set with --model or OPENAI_MODEL")
    if not language.strip():
        raise ValidationFailure("--language must not be empty")
    if max_agent_steps < 1:
        raise ValidationFailure("--max-agent-steps must be at least 1")
    if concept_concurrency < 1:
        raise ValidationFailure("--concept-concurrency must be at least 1")
    return GenerationIdentity(
        model=model,
        endpoint=endpoint_identity(base_url),
        language=language,
        max_agent_steps=max_agent_steps,
        parallel_tool_calls=parallel_tool_calls,
        concept_concurrency=concept_concurrency,
    )
