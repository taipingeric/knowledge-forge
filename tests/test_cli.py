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


def test_validate_command_accepts_a_state_free_portable_bundle_read_only(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    concept = output / "guides" / "refunds.md"
    concept.parent.mkdir(parents=True)
    concept.write_text("---\ntype: Guide\n---\n\n# Refunds\n")
    before = concept.read_bytes()

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS (portable OKF 0.2)"
    assert concept.read_bytes() == before
    assert not (output / ".knowledge-forge").exists()


def test_validate_command_portable_warns_about_managed_state_without_reading_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "knowledge"
    concept = output / "guide.md"
    concept.parent.mkdir(parents=True)
    concept.write_text("---\ntype: Guide\n---\n\n# Guide\n")
    private_state = output / ".knowledge-forge" / "state.json"
    private_state.parent.mkdir()
    private_state.write_text("not a Knowledge Forge state")
    before = sorted(
        (path.relative_to(output), path.read_bytes())
        for path in output.rglob("*")
        if path.is_file()
    )

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS (portable OKF 0.2)"
    assert "managed state detected — run with --managed for full validation" in result.stderr
    assert sorted(
        (path.relative_to(output), path.read_bytes())
        for path in output.rglob("*")
        if path.is_file()
    ) == before


def test_validate_command_managed_validates_portable_then_managed_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    calls: list[tuple[str, Path, bool | None]] = []

    monkeypatch.setattr(
        cli,
        "validate_portable_bundle",
        lambda bundle: calls.append(("portable", bundle, None)),
    )
    monkeypatch.setattr(
        cli,
        "validate_bundle",
        lambda bundle, sources=None, *, check_live_hash=False, **_: calls.append(
            ("managed", bundle, check_live_hash)
        ),
    )

    result = CliRunner().invoke(cli.app, ["validate", "--managed", "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS (managed Knowledge Forge Bundle)"
    assert calls == [("portable", output.resolve(), None), ("managed", output.resolve(), True)]


def test_validate_command_managed_with_source_checks_authoritative_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pdfs"
    output = tmp_path / "knowledge"
    source.mkdir()
    output.mkdir()
    (output / "guide.md").write_text("---\ntype: Guide\n---\n\n# Guide\n")
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cli,
        "validate_portable_bundle",
        lambda bundle: calls.append(("portable", bundle)),
    )
    monkeypatch.setattr(cli, "extract_sources", lambda root: ["authoritative-source"])
    monkeypatch.setattr(
        cli,
        "validate_bundle",
        lambda bundle, sources=None, *, check_live_hash=False, **_: calls.append(
            ("managed", (bundle, sources, check_live_hash))
        ),
    )

    result = CliRunner().invoke(
        cli.app,
        ["validate", "--managed", "--source", str(source), "--out", str(output)],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS (managed Knowledge Forge Bundle)"
    assert calls == [
        ("portable", output.resolve()),
        ("managed", (output.resolve(), ["authoritative-source"], True)),
    ]


def test_validate_command_rejects_a_missing_bundle_directory(tmp_path: Path) -> None:
    output = tmp_path / "missing"

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 2
    assert "Bundle directory does not exist" in result.stderr


@pytest.mark.parametrize(
    "raw",
    [
        "\ufeff---\ntype: Guide\n---\n# Guide\n",
        "---  \r\ntype: Guide\r\n--- \r\n# Guide\r\n",
    ],
)
def test_validate_command_accepts_portable_frontmatter_delimiters(
    tmp_path: Path, raw: str
) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    (output / "guide.md").write_text(raw, encoding="utf-8")

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0


@pytest.mark.parametrize(
    "raw",
    [
        "# No frontmatter\n",
        "---\ntype: [broken\n---\n",
        "---\n- type\n- Guide\n---\n",
        "---\ntype: '   '\n---\n",
        "---\ntype: 42\n---\n",
    ],
)
def test_validate_command_rejects_non_conformant_concepts(tmp_path: Path, raw: str) -> None:
    output = tmp_path / "knowledge"
    concept = output / "nested" / "broken.md"
    concept.parent.mkdir(parents=True)
    concept.write_text(raw)

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 2
    assert "Error: Portable OKF 0.2 validation failed" in result.stderr
    assert "nested/broken" in result.stderr


def test_validate_command_accepts_reserved_files_unknown_types_and_extensions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "knowledge"
    (output / "section").mkdir(parents=True)
    (output / "index.md").write_text(
        '---\nokf_version: "0.2"\n---\n\n# Knowledge\n'
    )
    (output / "section" / "index.md").write_text("# Section\n")
    (output / "section" / "log.md").write_text("# Log\n\n## 2026-08-17\n- Added.\n")
    (output / "section" / "unusual.md").write_text(
        "---\ntype: Acme Internal Widget\nx-acme-routing:\n  queue: blue\n---\n\n# Widget\n"
    )

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS (portable OKF 0.2)"


@pytest.mark.parametrize(
    ("raw_index", "expected_error"),
    [
        ('---\nokf_version: "0.1"\n---\n# Index\n', "unsupported okf_version"),
        ('\ufeff---\nokf_version: "0.1"\n---\n# Index\n', "unsupported okf_version"),
        ("---\nokf_version: 0.2\n---\n# Index\n", "must be the string"),
        ("---\ntitle: Index\n---\n# Index\n", "only declare okf_version"),
    ],
)
def test_validate_command_rejects_invalid_root_version_markers(
    tmp_path: Path, raw_index: str, expected_error: str
) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    (output / "index.md").write_text(raw_index)

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 2
    assert expected_error in result.stderr


@pytest.mark.parametrize(
    ("relative", "raw", "expected_error"),
    [
        ("nested/index.md", "---\nokf_version: '0.2'\n---\n# Index\n", "reserved index.md"),
        ("nested/index.md", "---\ntype: [broken\n---\n# Index\n", "reserved index.md"),
        ("nested/log.md", "---\ntype: Log\n---\n# Log\n", "reserved log.md"),
        ("log.md", "# Log\n\n## August 17, 2026\n- Added.\n", "YYYY-MM-DD"),
        ("log.md", "# Log\n\n## 2026-02-30\n- Added.\n", "valid calendar date"),
    ],
)
def test_validate_command_rejects_malformed_reserved_files(
    tmp_path: Path, relative: str, raw: str, expected_error: str
) -> None:
    output = tmp_path / "knowledge"
    path = output / relative
    path.parent.mkdir(parents=True)
    path.write_text(raw)

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 2
    assert expected_error in result.stderr


def test_validate_command_accepts_well_formed_optional_v02_fields(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    (output / "revenue.md").write_text(
        """---
type: Attested Computation
title: Revenue
description: Sanctioned revenue calculation.
resource: urn:acme:metric:revenue
tags: [finance, revenue]
status: stable
stale_after: 2026-12-31
generated: {by: process:finance, at: '2026-08-17T12:30:00Z'}
verified:
  - {by: human:reviewer, at: '2026-08-17T13:00:00+00:00'}
usage_window: {from: 2026-08-01, to: 2026-08-31}
sources:
  - id: revenue-policy
    resource: https://example.com/revenue-policy
    title: Revenue policy
    author: team:finance
    usage_count: 12
    last_modified: 2026-08-16
runtime: bigquery
parameters:
  - {name: year, type: integer, required: true}
computation: references/revenue.sql
executor:
  resource: references/run-revenue.md
  receipt: [job_id, executed_sql, result]
attester:
  resource: references/attest-revenue.py
x-acme-owner: finance-platform
---

# Computation

Revenue follows the approved policy.[^revenue-policy]

[^revenue-policy]: Revenue policy
"""
    )

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS (portable OKF 0.2)"


def test_validate_command_accepts_v02_datetime_fields(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    (output / "metric.md").write_text(
        """---
type: Metric
stale_after: 2026-12-31T00:00:00Z
usage_window: {from: 2026-08-01T00:00:00Z, to: 2026-08-31T23:59:59+00:00}
sources:
  - id: policy
    resource: https://example.com/policy
    last_modified: 2026-08-16T12:30:00+00:00
---

The metric follows the policy.[^policy]

[^policy]: Policy
"""
    )

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0


def test_validate_command_accepts_inline_attested_computation(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    (output / "metric.md").write_text(
        """---
type: Attested Computation
runtime: python
---

# Computation

```python
return 42
```
"""
    )

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("concept_type", "frontmatter", "body", "expected_error"),
    [
        ("Reference", "sources: source.md\n", "", "sources must be a list"),
        (
            "Reference",
            "sources:\n  - {id: policy}\n",
            "",
            "sources[0].resource must be a non-empty string",
        ),
        (
            "Reference",
            "usage_window: {from: 2026-08-31, to: 2026-08-01}\n",
            "",
            "usage_window.from must not be after usage_window.to",
        ),
        ("Reference", "generated: []\n", "", "generated must be a mapping"),
        (
            "Reference",
            "verified: {by: human:reviewer, at: yesterday}\n",
            "",
            "verified[0].at must be an ISO 8601 datetime",
        ),
        ("Reference", "status: retired\n", "", "status must be one of"),
        ("Reference", "stale_after: 30d\n", "", "stale_after must be a YYYY-MM-DD date"),
        (
            "Reference",
            "sources:\n  - {id: policy, resource: policy.md}\n",
            "Claim.[^missing]\n\n[^missing]: Missing\n",
            "citation [^missing] does not match any sources[].id",
        ),
        ("Attested Computation", "", "", "runtime is required"),
        (
            "Attested Computation",
            "runtime: python\n",
            "",
            "must provide either computation",
        ),
        (
            "Attested Computation",
            "runtime: python\ncomputation: references/metric.sql\n",
            "# Computation\n\n```sql\nSELECT 1\n```\n",
            "must provide computation in a file or an inline fence",
        ),
        (
            "Attested Computation",
            "runtime: python\ncomputation: references/metric.sql\n",
            "# Computation\n\n```sql\nSELECT 1\n```\n\n```sql\nSELECT 2\n```\n",
            "must provide computation in a file or an inline fence",
        ),
        (
            "Attested Computation",
            "runtime: python\nparameters:\n  - {name: year, type: integer, required: required}\n",
            "",
            "parameters[0].required must be a boolean",
        ),
    ],
)
def test_validate_command_rejects_malformed_optional_v02_fields(
    tmp_path: Path,
    concept_type: str,
    frontmatter: str,
    body: str,
    expected_error: str,
) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    (output / "concept.md").write_text(
        f"---\ntype: {concept_type}\n{frontmatter}---\n\n{body}"
    )

    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])

    assert result.exit_code == 2
    assert expected_error in result.stderr


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
    parallel_modes: list[bool] = []
    concept_concurrencies: list[int] = []

    def fake_generate(**kwargs: object) -> None:
        parallel_modes.append(bool(kwargs["parallel_tool_calls"]))
        concept_concurrencies.append(int(kwargs["concept_concurrency"]))
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
    assert parallel_modes == [True]
    assert concept_concurrencies == [4]


def test_generate_accepts_non_parallel_tool_call_compatibility_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"
    parallel_modes: list[bool] = []

    def fake_generate(**kwargs: object) -> None:
        parallel_modes.append(bool(kwargs["parallel_tool_calls"]))

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
            "--no-parallel-tool-calls",
        ],
    )

    assert result.exit_code == 0
    assert parallel_modes == [False]


@pytest.mark.parametrize(
    ("extra_args", "expected_parallel"),
    [([], True), (["--no-parallel-tool-calls"], False)],
)
def test_update_selects_tool_call_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_parallel: bool,
) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"
    observed: list[bool] = []

    def fake_update(**kwargs: object) -> bool:
        parallel = bool(kwargs["parallel_tool_calls"])
        observed.append(parallel)
        progress = kwargs["progress"]
        assert callable(progress)
        progress(
            "Tool-call mode: parallel."
            if parallel
            else "Tool-call mode: non-parallel compatibility."
        )
        return False

    monkeypatch.setattr(cli, "update_bundle", fake_update)
    monkeypatch.setattr(cli, "monotonic", iter([0.0, 1.0]).__next__)
    result = CliRunner().invoke(
        cli.app,
        [
            "update",
            "--source",
            str(source),
            "--out",
            str(output),
            "--model",
            "fake-model",
            "--api-key",
            "secret",
            *extra_args,
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "No changes"
    expected_mode = (
        "[knowledge-forge] Tool-call mode: parallel."
        if expected_parallel
        else "[knowledge-forge] Tool-call mode: non-parallel compatibility."
    )
    assert result.stderr.splitlines() == [
        expected_mode,
        "[knowledge-forge] Total processing time: 1.000s.",
    ]
    assert observed == [expected_parallel]


@pytest.mark.parametrize(
    ("failure", "expected_exit", "expected_error"),
    [
        (None, 0, None),
        (cli.ReconciliationRequired("report.md"), 3, "report.md"),
        (cli.KnowledgeForgeError("update failed"), 2, "Error: update failed"),
    ],
)
def test_update_reports_total_time_and_preserves_output_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    expected_exit: int,
    expected_error: str | None,
) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"

    def fake_update(**kwargs: object) -> bool:
        timing = kwargs["timing"]
        assert isinstance(timing, cli.ProcessingTimer)
        with timing.phase("Current Bundle validation"):
            pass
        if failure is not None:
            raise failure
        return False

    monkeypatch.setattr(cli, "update_bundle", fake_update)
    monkeypatch.setattr(cli, "monotonic", iter([0.0, 1.0, 2.0, 3.0]).__next__)
    result = CliRunner().invoke(
        cli.app,
        [
            "update",
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

    assert result.exit_code == expected_exit
    assert result.stdout.strip() == ("No changes" if failure is None else "")
    stderr = result.stderr.splitlines()
    assert "[knowledge-forge] Current Bundle validation completed in 1.000s." in stderr
    assert stderr[-1] == "[knowledge-forge] Total processing time: 3.000s."
    if expected_error is not None:
        assert expected_error in stderr[-2]


def test_generate_accepts_sequential_concept_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()
    output = tmp_path / "knowledge"
    configuration: list[tuple[int, bool]] = []

    def fake_generate(**kwargs: object) -> None:
        configuration.append(
            (int(kwargs["concept_concurrency"]), bool(kwargs["parallel_tool_calls"]))
        )

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
            "--concept-concurrency",
            "1",
            "--no-parallel-tool-calls",
        ],
    )

    assert result.exit_code == 0
    assert configuration == [(1, False)]


def test_generate_rejects_non_positive_concept_concurrency(tmp_path: Path) -> None:
    source = tmp_path / "pdfs"
    source.mkdir()

    result = CliRunner().invoke(
        cli.app,
        [
            "generate",
            "--source",
            str(source),
            "--out",
            str(tmp_path / "knowledge"),
            "--model",
            "fake-model",
            "--api-key",
            "secret",
            "--concept-concurrency",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert "x>=1" in result.stderr


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
        "[knowledge-forge] Tool-call mode: parallel.",
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
        "[knowledge-forge] Tool-call mode: parallel.",
        "[knowledge-forge] Reading PDF sources...",
        "Error: PDF Source failed",
        "[knowledge-forge] Total processing time: 2.000s.",
    ]
