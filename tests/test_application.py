from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from knowledge_forge import application
from knowledge_forge.errors import ReconciliationRequired, ValidationFailure
from knowledge_forge.models import ConceptDraft, Evidence, PDFSource, SourcePage
from knowledge_forge.okf import dump_markdown, parse_markdown, render_concept
from knowledge_forge.sources import logical_resource, sha256_text
from knowledge_forge.state import bundle_hash, load_state
from knowledge_forge.validation import validate_bundle


def pdf(version: str = "one") -> PDFSource:
    return PDFSource(
        id="policy.pdf",
        resource=logical_resource("policy.pdf"),
        content_sha256=sha256_text(version),
        pages=[SourcePage(number=1, text=f"Refund policy {version}")],
    )


def concept(source: PDFSource, rule: str, notes: str = "Original") -> str:
    return render_concept(
        ConceptDraft(
            slug="refund-policy",
            title="Refund policy",
            type="Policy",
            description="Rules for refunds.",
            body=(
                f"# Rule\n\n{rule}[^policy.pdf@p1]\n\n# Notes\n\n{notes}\n\n"
                "[^policy.pdf@p1]: Refund policy, page 1\n"
            ),
            evidence=[Evidence(source_id=source.id, pages=[1])],
        ),
        {source.id: source},
        "knowledge-forge/fake",
    )


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    current_source = [pdf()]
    generated = [concept(current_source[0], "Seven days")]

    monkeypatch.setattr(application, "extract_sources", lambda _: current_source)
    monkeypatch.setattr(
        application,
        "_run_agent",
        lambda **_: ({"concepts/refund-policy": generated[0]}, "English"),
    )
    return current_source, generated


def generate_bundle(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    output = tmp_path / "knowledge"
    application.generate(
        source=source_dir,
        output=output,
        model="fake-model",
        api_key="secret",
        base_url="https://models.example/v1",
        language="auto",
        max_agent_steps=50,
    )
    return source_dir, output


def test_generate_and_deterministic_noop(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    source_dir, output = generate_bundle(tmp_path)
    assert load_state(output).generation.parallel_tool_calls is True
    before = bundle_hash(output, include_state=True)
    assert (
        application.update(
            source=source_dir,
            output=output,
            model="fake-model",
            api_key="secret",
            base_url="https://models.example/v1",
            language="auto",
            max_agent_steps=50,
        )
        is False
    )
    assert bundle_hash(output, include_state=True) == before
    validate_bundle(output)


def test_generate_records_and_reports_non_parallel_compatibility_mode(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    output = tmp_path / "knowledge"
    progress: list[str] = []

    application.generate(
        source=source_dir,
        output=output,
        model="fake-model",
        api_key="secret",
        base_url="https://models.example/v1",
        language="auto",
        max_agent_steps=50,
        parallel_tool_calls=False,
        progress=progress.append,
    )

    assert progress[0] == "Tool-call mode: non-parallel compatibility."
    assert load_state(output).generation.parallel_tool_calls is False


def test_nonoverlapping_human_edit_is_preserved_without_agent_call(
    tmp_path: Path,
    fake_runtime: tuple[list[PDFSource], list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir, output = generate_bundle(tmp_path)
    path = output / "concepts/refund-policy.md"
    raw = path.read_text().replace("Original", "Curated by a human")
    raw = raw.replace("title: Refund policy", "title: Curated refund policy")
    path.write_text(raw)
    monkeypatch.setattr(
        application,
        "_run_agent",
        lambda **_: pytest.fail("unchanged sources must not call the agent"),
    )
    assert application.update(
        source=source_dir,
        output=output,
        model="fake-model",
        api_key="secret",
        base_url="https://models.example/v1",
        language="auto",
        max_agent_steps=50,
    )
    assert "Curated by a human" in path.read_text()
    assert "Curated refund policy" in (output / "index.md").read_text()
    validate_bundle(output)


def test_conflict_leaves_live_bundle_untouched_and_can_use_source(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    sources, generated = fake_runtime
    source_dir, output = generate_bundle(tmp_path)
    live = output / "concepts/refund-policy.md"
    live.write_text(live.read_text().replace("Seven days", "Fourteen days"))
    live_before = live.read_text()
    sources[0] = pdf("two")
    generated[0] = concept(sources[0], "Five days")

    with pytest.raises(ReconciliationRequired):
        application.update(
            source=source_dir,
            output=output,
            model="fake-model",
            api_key="secret",
            base_url="https://models.example/v1",
            language="auto",
            max_agent_steps=50,
        )
    assert live.read_text() == live_before
    work = tmp_path / "knowledge.reconciliation"
    report = (tmp_path / "knowledge.reconciliation.md").read_text()
    assert "Fourteen days" in report
    assert "Five days" in report
    assert "`policy.pdf` pages 1" in report
    resolution = yaml.safe_load((work / "resolution.yaml").read_text())
    resolution["resolutions"][0]["action"] = "use-source"
    (work / "resolution.yaml").write_text(yaml.safe_dump(resolution, sort_keys=False))
    application.reconcile(
        source=source_dir,
        output=output,
        resolution_path=work / "resolution.yaml",
    )
    assert "Five days" in live.read_text()
    assert not work.exists()
    assert "Status: resolved" in (tmp_path / "knowledge.reconciliation.md").read_text()
    validate_bundle(output, sources)


def test_verify_is_version_bound(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    sources, _ = fake_runtime
    source_dir, output = generate_bundle(tmp_path)
    concept_path = output / "concepts/refund-policy.md"
    concept_path.write_text(
        concept_path.read_text().replace("title: Refund policy", "title: Reviewed policy")
    )
    application.verify(
        source=source_dir,
        output=output,
        concept_id="concepts/refund-policy.md",
        actor="human:reviewer",
    )
    raw = (output / "concepts/refund-policy.md").read_text()
    metadata, _ = parse_markdown(raw)
    assert metadata["verified"][0]["by"] == "human:reviewer"
    assert "Reviewed policy" in (output / "index.md").read_text()
    assert load_state(output).verification_history[-1].by == "human:reviewer"
    validate_bundle(output, sources)

    concept_path.write_text(concept_path.read_text().replace("Seven days", "Seven business days"))
    application.verify(
        source=source_dir,
        output=output,
        concept_id="concepts/refund-policy",
        actor="human:second-reviewer",
    )
    metadata, _ = parse_markdown(concept_path.read_text())
    assert [item["by"] for item in metadata["verified"]] == ["human:second-reviewer"]
    assert len(load_state(output).verification_history) == 2


def test_managed_provenance_tampering_fails(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    _, output = generate_bundle(tmp_path)
    path = output / "concepts/refund-policy.md"
    metadata, body = parse_markdown(path.read_text())
    metadata["sources"][0]["content_sha256"] = "0" * 64
    path.write_text(dump_markdown(metadata, body))
    with pytest.raises(ValidationFailure, match="provenance"):
        validate_bundle(output)


def test_manual_verified_metadata_tampering_fails(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    _, output = generate_bundle(tmp_path)
    path = output / "concepts/refund-policy.md"
    metadata, body = parse_markdown(path.read_text())
    metadata["verified"] = {"by": "human:someone", "at": "2026-01-01T00:00:00Z"}
    path.write_text(dump_markdown(metadata, body))
    with pytest.raises(ValidationFailure, match="provenance"):
        validate_bundle(output)


def test_private_state_tampering_fails(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    _, output = generate_bundle(tmp_path)
    path = output / ".knowledge-forge/state.json"
    path.write_text(path.read_text().replace('"language": "auto"', '"language": "fr"'))
    with pytest.raises(ValidationFailure, match="state integrity"):
        validate_bundle(output)


def test_human_deletion_can_be_kept_as_a_tombstone(
    tmp_path: Path, fake_runtime: tuple[list[PDFSource], list[str]]
) -> None:
    source_dir, output = generate_bundle(tmp_path)
    concept_path = output / "concepts/refund-policy.md"
    concept_path.unlink()

    with pytest.raises(ReconciliationRequired):
        application.update(
            source=source_dir,
            output=output,
            model="fake-model",
            api_key="secret",
            base_url="https://models.example/v1",
            language="auto",
            max_agent_steps=50,
        )
    work = tmp_path / "knowledge.reconciliation"
    application.reconcile(
        source=source_dir,
        output=output,
        resolution_path=work / "resolution.yaml",
    )
    assert not concept_path.exists()
    state = load_state(output)
    assert state.concepts["concepts/refund-policy"].deleted
    validate_bundle(output)

    assert application.update(
        source=source_dir,
        output=output,
        model="a-different-model",
        api_key="secret",
        base_url="https://models.example/v1",
        language="auto",
        max_agent_steps=50,
    )
    assert not concept_path.exists()
