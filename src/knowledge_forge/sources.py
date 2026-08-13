from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader

from .errors import SearchQueryFailure, ValidationFailure
from .models import PDFSource, SourcePage


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def logical_resource(source_id: str) -> str:
    return f"urn:knowledge-forge:pdf:{quote(source_id, safe='')}"


def discover_pdfs(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ValidationFailure(f"Source directory does not exist: {root}")

    found: list[Path] = []
    identities: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValidationFailure(f"Symlinks are not allowed in the source tree: {path}")
        if not path.is_file() or path.suffix.casefold() != ".pdf":
            continue
        relative = path.relative_to(root).as_posix()
        if any(character in relative for character in "[]") or any(
            ord(character) < 32 for character in relative
        ):
            raise ValidationFailure(
                f"PDF source path contains characters unsupported by citation labels: {relative!r}"
            )
        normalized = unicodedata.normalize("NFC", relative).casefold()
        if normalized in identities:
            raise ValidationFailure(
                f"PDF source identity collision: {identities[normalized]} and {path}"
            )
        identities[normalized] = path
        found.append(path)
    if not found:
        raise ValidationFailure(f"No PDF files found under: {root}")
    return found


def extract_sources(root: Path) -> list[PDFSource]:
    root = root.resolve()
    failures: list[str] = []
    sources: list[PDFSource] = []
    for path in discover_pdfs(root):
        source_id = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError("encrypted PDFs are not supported")
            pages = [
                SourcePage(number=index, text=(page.extract_text() or "").strip())
                for index, page in enumerate(reader.pages, start=1)
            ]
            if not pages or not any(page.text for page in pages):
                raise ValueError("PDF has no extractable text layer")
            content = path.read_bytes()
            sources.append(
                PDFSource(
                    id=source_id,
                    resource=logical_resource(source_id),
                    content_sha256=sha256_bytes(content),
                    pages=pages,
                )
            )
        except Exception as exc:  # pypdf exposes several parser-specific errors
            failures.append(f"{source_id}: {exc}")
    if failures:
        raise ValidationFailure("Invalid PDF source set:\n- " + "\n- ".join(failures))
    return sources


def source_set_hash(sources: list[PDFSource]) -> str:
    material = "\n".join(f"{item.id}\0{item.content_sha256}" for item in sources)
    return sha256_text(material)


class PageIndex:
    """Ephemeral page-oriented full-text index."""

    def __init__(self, database: Path) -> None:
        self._database = database
        with self._connect() as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE pages USING fts5(source_id UNINDEXED, page UNINDEXED, text)"
            )

    def _connect(self) -> sqlite3.Connection:
        """Create a connection owned by the calling thread."""
        return sqlite3.connect(self._database)

    def add(self, sources: list[PDFSource]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO pages(source_id, page, text) VALUES (?, ?, ?)",
                [
                    (source.id, page.number, page.text)
                    for source in sources
                    for page in source.pages
                ],
            )

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        limit = min(max(limit, 1), 25)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT source_id, page, snippet(pages, 2, '[', ']', ' … ', 32) "
                    "FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT ?",
                    (query, limit),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            raise SearchQueryFailure(f"Invalid full-text search query: {query!r}") from exc
        return [{"source_id": row[0], "page": int(row[1]), "snippet": row[2]} for row in rows]

    def read(self, source_id: str, pages: list[int]) -> list[dict[str, object]]:
        if not pages:
            return []
        placeholders = ",".join("?" for _ in pages)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT source_id, page, text FROM pages "  # noqa: S608 - placeholders below
                f"WHERE source_id = ? AND page IN ({placeholders}) ORDER BY page",
                (source_id, *pages),
            ).fetchall()
        return [{"source_id": row[0], "page": int(row[1]), "text": row[2]} for row in rows]

    def close(self) -> None:
        """Retained for the context-manager API; connections are operation-scoped."""

    def __enter__(self) -> PageIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
