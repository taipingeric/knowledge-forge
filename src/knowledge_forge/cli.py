from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Annotated

import typer
from dotenv import load_dotenv

from .application import generate as generate_bundle
from .application import reconcile as reconcile_bundle
from .application import update as update_bundle
from .application import verify as verify_concept
from .errors import KnowledgeForgeError, ReconciliationRequired
from .sources import extract_sources
from .timing import ProcessingTimer
from .validation import validate_bundle

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Build human-and-agent maintained OKF 0.2 knowledge bundles from PDFs.",
)


def load_local_env() -> bool:
    """Load only .env in the invocation directory without overriding process env."""
    path = Path.cwd() / ".env"
    return load_dotenv(dotenv_path=path, override=False) if path.is_file() else False


def main() -> None:
    load_local_env()
    app()


SourceOption = Annotated[
    Path, typer.Option("--source", exists=True, file_okay=False, readable=True)
]
OutputOption = Annotated[Path, typer.Option("--out", file_okay=False)]
ModelOption = Annotated[str, typer.Option("--model", envvar="OPENAI_MODEL")]
ApiKeyOption = Annotated[str, typer.Option(envvar="OPENAI_API_KEY", hide_input=True)]
BaseUrlOption = Annotated[str | None, typer.Option("--base-url", envvar="OPENAI_BASE_URL")]
LanguageOption = Annotated[str, typer.Option("--language")]
StepOption = Annotated[
    int,
    typer.Option(
        "--max-agent-steps",
        min=1,
        help="Maximum model calls for each planning or Concept synthesis task.",
    ),
]
NoParallelToolCallsOption = Annotated[
    bool,
    typer.Option(
        "--no-parallel-tool-calls",
        help="Use serial tool calls for gateways that cannot replay parallel results.",
    ),
]
ConceptConcurrencyOption = Annotated[
    int,
    typer.Option(
        "--concept-concurrency",
        min=1,
        help="Maximum Concept synthesis tasks to run concurrently.",
    ),
]


def _show_progress(message: str) -> None:
    typer.echo(f"[knowledge-forge] {message}", err=True)


def _run(operation) -> None:
    try:
        operation()
    except ReconciliationRequired as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(3) from exc
    except KnowledgeForgeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc


@app.command()
def generate(
    source: SourceOption,
    out: OutputOption,
    model: ModelOption,
    api_key: ApiKeyOption,
    base_url: BaseUrlOption = None,
    language: LanguageOption = "auto",
    max_agent_steps: StepOption = 50,
    no_parallel_tool_calls: NoParallelToolCallsOption = False,
    concept_concurrency: ConceptConcurrencyOption = 4,
) -> None:
    """Create a new OKF Bundle from the complete PDF source set."""
    timing = ProcessingTimer(_show_progress, monotonic)

    def operation() -> None:
        generate_bundle(
            source=source,
            output=out,
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            max_agent_steps=max_agent_steps,
            parallel_tool_calls=not no_parallel_tool_calls,
            concept_concurrency=concept_concurrency,
            progress=_show_progress,
            timing=timing,
        )
        typer.echo(f"Generated OKF Bundle: {out.resolve()}")

    try:
        _run(operation)
    finally:
        timing.report_total()


@app.command()
def update(
    source: SourceOption,
    out: OutputOption,
    model: ModelOption,
    api_key: ApiKeyOption,
    base_url: BaseUrlOption = None,
    language: LanguageOption = "auto",
    max_agent_steps: StepOption = 50,
) -> None:
    """Reconcile a Bundle against its complete authoritative PDF source set."""

    def operation() -> None:
        changed = update_bundle(
            source=source,
            output=out,
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            max_agent_steps=max_agent_steps,
            progress=_show_progress,
        )
        typer.echo(f"Updated OKF Bundle: {out.resolve()}" if changed else "No changes")

    _run(operation)


@app.command()
def reconcile(
    source: SourceOption,
    out: OutputOption,
    resolution: Annotated[Path, typer.Option("--resolution", exists=True, dir_okay=False)],
) -> None:
    """Apply reviewed conflict resolutions to a pending candidate."""

    def operation() -> None:
        reconcile_bundle(source=source, output=out, resolution_path=resolution)
        typer.echo(f"Reconciled OKF Bundle: {out.resolve()}")

    _run(operation)


@app.command()
def verify(
    source: SourceOption,
    out: OutputOption,
    concept: Annotated[str, typer.Option("--concept")],
    by: Annotated[str, typer.Option("--by")],
) -> None:
    """Append a human verification event for the current Concept version."""

    def operation() -> None:
        verify_concept(source=source, output=out, concept_id=concept, actor=by)
        typer.echo(f"Verified {concept} as {by}")

    _run(operation)


@app.command("validate")
def validate_command(
    out: OutputOption,
    source: Annotated[
        Path | None,
        typer.Option("--source", exists=True, file_okay=False, readable=True),
    ] = None,
) -> None:
    """Deterministically validate the Bundle, provenance, and private state."""

    def operation() -> None:
        sources = extract_sources(source) if source else None
        validate_bundle(out.resolve(), sources)
        typer.echo("Bundle is valid")

    _run(operation)
