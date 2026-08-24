from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from knowledge_forge.errors import ValidationFailure
from knowledge_forge.models import EvidenceUnit, PDFPageLocator, PDFSource, SourcePage
from knowledge_forge.sources import EvidenceIndex, PageIndex, logical_resource, sha256_text


def test_logical_resource_is_portable() -> None:
    assert logical_resource("policies/退款 規則.pdf") == (
        "urn:knowledge-forge:pdf:policies%2F%E9%80%80%E6%AC%BE%20%E8%A6%8F%E5%89%87.pdf"
    )


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
