from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from knowledge_forge.errors import ValidationFailure
from knowledge_forge.models import (
    ConceptDraft,
    ConceptState,
    Evidence,
    EvidenceUnit,
    ForgeState,
    GenerationIdentity,
    MarkdownBlockLocator,
    MarkdownSource,
    PDFPageLocator,
    PDFSource,
    SourcePage,
)
from knowledge_forge.okf import render_concept
from knowledge_forge.sources import (
    EvidenceIndex,
    PageIndex,
    extract_sources,
    logical_resource,
    sha256_text,
)
from knowledge_forge.staleness import detect_staleness


def test_logical_resource_is_portable() -> None:
    assert logical_resource("policies/退款 規則.pdf") == (
        "urn:knowledge-forge:pdf:policies%2F%E9%80%80%E6%AC%BE%20%E8%A6%8F%E5%89%87.pdf"
    )


def test_extract_sources_discovers_structural_markdown_evidence(tmp_path: Path) -> None:
    (tmp_path / "handbook.md").write_text(
        "---\ntitle: Opaque metadata\n---\n<!-- keep this -->\nintro\n"
        "# Policy\nBody\n```md\n# Not a heading\n```\n## Scope\nApplies\n# Policy\nSecond\n"
    )

    sources = extract_sources(tmp_path)

    assert len(sources) == 1
    markdown = sources[0]
    assert isinstance(markdown, MarkdownSource)
    assert markdown.resource == logical_resource("handbook.md", "markdown")
    root, first_policy, scope, second_policy = markdown.evidence
    assert root.text.startswith("---")
    assert "# Not a heading" in first_policy.text
    assert isinstance(scope.locator, MarkdownBlockLocator)
    assert scope.locator.heading_path == ["Policy", "Scope"]
    assert first_policy.locator.occurrence == 1
    assert second_policy.locator.occurrence == 2
    assert root.line_start == 1
    assert root.line_end == 5


def test_staleness_detects_changed_markdown_block_but_ignores_line_shifts(tmp_path: Path) -> None:
    source = MarkdownSource(
        id="handbook.md",
        resource=logical_resource("handbook.md", "markdown"),
        content_sha256=sha256_text("first"),
        evidence=[
            EvidenceUnit(
                locator=MarkdownBlockLocator(
                    heading_path=["Policy"], occurrence=1, content_sha256=sha256_text("rule")
                ),
                text="rule",
                line_start=2,
                line_end=3,
            )
        ],
    )
    raw = render_concept(
        ConceptDraft(
            slug="policy",
            title="Policy",
            type="Policy",
            description="A policy.",
            body="# Policy\n\nRule.[^handbook.md#markdown_block:%5B%5B%22Policy%22%5D%2C1%5D]",
            evidence=[Evidence(source_id="handbook.md", locators=[source.evidence[0].locator])],
        ),
        {source.id: source},
        "knowledge-forge/test",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "policy.md").write_text(raw)
    state = ForgeState(
        generation=GenerationIdentity(
            model="test", endpoint="https://models.example", language="auto", max_agent_steps=1
        ),
        source_set_hash="source-set",
        sources={},
        bundle_hash="bundle",
        tool_files={},
        concepts={"policy": ConceptState(ownership="agent")},
    )

    shifted = source.model_copy(deep=True)
    shifted.evidence[0].line_start = 10
    shifted.evidence[0].line_end = 11
    assert detect_staleness(bundle, state, [shifted]) == []

    changed = source.model_copy(deep=True)
    changed.evidence[0].text = "changed rule"
    changed.evidence[0].locator.content_sha256 = sha256_text("changed rule")
    assert detect_staleness(bundle, state, [changed])[0]["source_id"] == "handbook.md"


@pytest.mark.parametrize(
    "content, message", [(b"", "Markdown source is empty"), (b"\xff", "not valid UTF-8")]
)
def test_extract_sources_rejects_invalid_markdown_before_agent(
    tmp_path: Path, content: bytes, message: str
) -> None:
    (tmp_path / "evidence.md").write_bytes(content)

    with pytest.raises(ValidationFailure, match=message):
        extract_sources(tmp_path)


def test_extract_sources_rejects_marked_okf_root(tmp_path: Path) -> None:
    (tmp_path / "index.md").write_text('---\nokf_version: "0.2"\n---\n# Imported\n')

    with pytest.raises(ValidationFailure, match="not-yet-supported import"):
        extract_sources(tmp_path)


def test_pdf_source_exposes_typed_evidence_through_generalized_contract() -> None:
    source = PDFSource(
        id="handbook.pdf",
        resource=logical_resource("handbook.pdf"),
        content_sha256=sha256_text("pdf"),
        pages=[SourcePage(number=1, text="Refunds take seven business days.")],
    )

    assert source.kind == "pdf"
    assert source.source_identity == "handbook.pdf"
    assert source.evidence == [
        EvidenceUnit(locator=PDFPageLocator(page=1), text="Refunds take seven business days.")
    ]


def test_evidence_index_returns_typed_pdf_locators_from_worker_threads(tmp_path: Path) -> None:
    source = PDFSource(
        id="handbook.pdf",
        resource=logical_resource("handbook.pdf"),
        content_sha256=sha256_text("pdf"),
        pages=[SourcePage(number=1, text="Refunds take seven business days.")],
    )
    with EvidenceIndex(tmp_path / "evidence.sqlite") as index:
        index.add([source])
        with ThreadPoolExecutor(max_workers=2) as executor:
            search, read = [
                future.result()
                for future in (
                    executor.submit(index.search, "refunds"),
                    executor.submit(index.read, "handbook.pdf", [PDFPageLocator(page=1)]),
                )
            ]

    assert search[0]["locator"] == {"kind": "pdf_page", "page": 1}
    assert read[0]["locator"] == {"kind": "pdf_page", "page": 1}
    assert read[0]["text"].startswith("Refunds")


def test_page_index_searches_and_reads(tmp_path: Path) -> None:
    from knowledge_forge.models import PDFSource, SourcePage

    source = PDFSource(
        id="handbook.pdf",
        resource=logical_resource("handbook.pdf"),
        content_sha256=sha256_text("pdf"),
        pages=[SourcePage(number=1, text="Refunds take seven business days.")],
    )
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([source])
        assert index.search("refunds")[0]["page"] == 1
        assert index.read("handbook.pdf", [1])[0]["text"].startswith("Refunds")


def test_page_index_rejects_invalid_fts_query(tmp_path: Path) -> None:
    with PageIndex(tmp_path / "pages.sqlite") as index, pytest.raises(ValidationFailure):
        index.search('"unterminated')


def test_page_index_can_be_queried_from_agent_worker_thread(tmp_path: Path) -> None:
    from knowledge_forge.models import PDFSource, SourcePage

    source = PDFSource(
        id="handbook.pdf",
        resource=logical_resource("handbook.pdf"),
        content_sha256=sha256_text("pdf"),
        pages=[SourcePage(number=1, text="Refunds take seven business days.")],
    )
    with PageIndex(tmp_path / "pages.sqlite") as index:
        index.add([source])
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                future
                for _ in range(4)
                for future in (
                    executor.submit(index.search, "refunds"),
                    executor.submit(index.read, "handbook.pdf", [1]),
                )
            ]
            results = [future.result() for future in futures]

    assert all(result[0]["page"] == 1 for result in results)
