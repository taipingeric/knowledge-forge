from pathlib import Path

import pytest

from knowledge_forge.errors import ValidationFailure
from knowledge_forge.sources import PageIndex, logical_resource, sha256_text


def test_logical_resource_is_portable() -> None:
    assert logical_resource("policies/退款 規則.pdf") == (
        "urn:knowledge-forge:pdf:policies%2F%E9%80%80%E6%AC%BE%20%E8%A6%8F%E5%89%87.pdf"
    )


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
