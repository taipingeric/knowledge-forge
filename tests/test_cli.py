import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knowledge_forge import application, cli
from knowledge_forge.models import (
    ConceptDraft,
    ConceptPlan,
    Evidence,
    PDFSource,
    PlannedConcept,
    SourcePage,
)
from knowledge_forge.sources import logical_resource, sha256_text


def test_validate_command_is_read_only(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    called: list[Path] = []
    monkeypatch.setattr(cli, "validate_bundle", lambda path, sources: called.append(path))
    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "Bundle is valid"
    assert called == [output.resolve()]


def test_load_local_env_reads_only_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (tmp_path / ".env").write_text("OPENAI_MODEL=parent-model\n")
    monkeypatch.chdir(child)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert cli.load_local_env() is False
    assert os.getenv("OPENAI_MODEL") is None

    (child / ".env").write_text("OPENAI_MODEL=local-model\n")
    assert cli.load_local_env() is True
    assert os.getenv("OPENAI_MODEL") == "local-model"


def test_load_local_env_does_not_override_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")

    assert cli.load_local_env() is True
    assert os.getenv("OPENAI_API_KEY") == "from-process"


def test_console_entrypoint_loads_dotenv_before_starting_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    observed: list[str | None] = []
    monkeypatch.setattr(cli, "app", lambda: observed.append(os.getenv("OPENAI_MODEL")))

    cli.main()

    assert observed == ["dotenv-model"]


def test_generate_shows_progress_on_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"

    def fake_generate(**kwargs: object) -> None:
        progress = kwargs["progress"]
        assert callable(progress)
        progress("Planning concepts with the reasoning agent...")

    monkeypatch.setattr(cli, "generate_bundle", fake_generate)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(cli, "monotonic", clock.__next__)
    result = CliRunner().invoke(
        cli.app,
        [
            "generate",
            "--source",
            str(source),
            "--out",
            str(output),
            "--model",
            "fake-model",
            "--api-key",
            "secret",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == f"Generated OKF Bundle: {output.resolve()}"
    assert result.stderr.splitlines() == [
        "[knowledge-forge] Planning concepts with the reasoning agent...",
        "[knowledge-forge] Total processing time: 1.000s.",
    ]


class TickingClock:
    def __init__(self) -> None:
        self.current = -1.0

    def __call__(self) -> float:
        self.current += 1.0
        return self.current


class FakeTimingAgent:
    def __init__(self, **_: object) -> None:
        pass

    def plan(self, language: str, existing_ids: list[str]) -> ConceptPlan:
        return ConceptPlan(
            language="English",
            concepts=[
                PlannedConcept(
                    slug="refund-policy",
                    title="Refund policy",
                    type="Policy",
                    description="Rules for refunds.",
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


def test_generate_reports_processing_time_without_changing_command_output_or_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"
    pdf = PDFSource(
        id="policy.pdf",
        resource=logical_resource("policy.pdf"),
        content_sha256=sha256_text("pdf"),
        pages=[SourcePage(number=1, text="Refunds take seven days.")],
    )
    monkeypatch.setattr(application, "extract_sources", lambda _: [pdf])
    monkeypatch.setattr(application, "ReasoningAgent", FakeTimingAgent)
    monkeypatch.setattr(cli, "monotonic", TickingClock(), raising=False)

    result = CliRunner().invoke(
        cli.app,
        [
            "generate",
            "--source",
            str(source),
            "--out",
            str(output),
            "--model",
            "fake-model",
            "--api-key",
            "secret",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == f"Generated OKF Bundle: {output.resolve()}"
    assert result.stderr.splitlines() == [
        "[knowledge-forge] Reading PDF sources...",
        "[knowledge-forge] PDF Source reading completed in 1.000s.",
        "[knowledge-forge] Loaded 1 PDFs with 1 pages.",
        "[knowledge-forge] Indexing 1 pages from 1 PDFs...",
        "[knowledge-forge] PDF indexing completed in 1.000s.",
        "[knowledge-forge] Planning concepts with the reasoning agent...",
        "[knowledge-forge] Concept planning completed in 1.000s.",
        "[knowledge-forge] Planned 1 concepts in English.",
        "[knowledge-forge] Synthesizing concept 1/1: refund-policy",
        "[knowledge-forge] Concept synthesis 1/1 (refund-policy) completed in 1.000s.",
        "[knowledge-forge] Rendering and validating 1 concepts...",
        "[knowledge-forge] Concept rendering and validation completed in 1.000s.",
        "[knowledge-forge] Agent-generated concepts passed validation.",
        "[knowledge-forge] Writing and validating the candidate bundle...",
        "[knowledge-forge] Candidate Bundle writing and validation completed in 1.000s.",
        "[knowledge-forge] Publishing the bundle atomically...",
        "[knowledge-forge] Atomic publication completed in 1.000s.",
        "[knowledge-forge] Total processing time: 15.000s.",
    ]
    bundle_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()
    )
    assert "processing time" not in bundle_text.casefold()


def test_failed_generate_reports_total_time_and_preserves_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"

    def fail_reading(_: Path) -> list[PDFSource]:
        raise cli.KnowledgeForgeError("PDF Source failed")

    monkeypatch.setattr(application, "extract_sources", fail_reading)
    monkeypatch.setattr(cli, "monotonic", TickingClock(), raising=False)

    result = CliRunner().invoke(
        cli.app,
        [
            "generate",
            "--source",
            str(source),
            "--out",
            str(output),
            "--model",
            "fake-model",
            "--api-key",
            "secret",
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "[knowledge-forge] Reading PDF sources...",
        "Error: PDF Source failed",
        "[knowledge-forge] Total processing time: 2.000s.",
    ]
