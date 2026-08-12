from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from knowledge_forge.errors import ValidationFailure
from knowledge_forge.sources import discover_pdfs, extract_sources


def write_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    safe = text.replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def test_extract_source_preserves_relative_identity_and_page_text(tmp_path: Path) -> None:
    nested = tmp_path / "Policies"
    nested.mkdir()
    write_text_pdf(nested / "Refund.PDF", "Refunds take seven days")
    sources = extract_sources(tmp_path)
    assert sources[0].id == "Policies/Refund.PDF"
    assert sources[0].pages[0].text == "Refunds take seven days"


def test_discovery_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.pdf"
    write_text_pdf(target, "Text")
    (tmp_path / "link.pdf").symlink_to(target)
    with pytest.raises(ValidationFailure, match="Symlinks"):
        discover_pdfs(tmp_path)


def test_blank_pdf_fails_atomically(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (tmp_path / "blank.pdf").open("wb") as handle:
        writer.write(handle)
    with pytest.raises(ValidationFailure, match="no extractable text"):
        extract_sources(tmp_path)


def test_text_pdf_may_contain_an_intentionally_blank_page(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    write_text_pdf(path, "Text on page one")
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)
    sources = extract_sources(tmp_path)
    assert len(sources[0].pages) == 2
    assert sources[0].pages[1].text == ""
