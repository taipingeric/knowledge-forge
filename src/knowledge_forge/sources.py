from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader

from .errors import SearchQueryFailure, ValidationFailure
from .models import (
    EvidenceLocator,
    EvidenceUnit,
    KnowledgeSource,
    MarkdownBlockLocator,
    MarkdownSource,
    PDFPageLocator,
    PDFSource,
    SourcePage,
)


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of raw source bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 encoded text."""

    return sha256_bytes(value.encode("utf-8"))


def logical_resource(source_id: str, kind: str = "pdf") -> str:
    """Build the stable logical resource URI for a source-root-relative identity."""

    return f"urn:knowledge-forge:{kind}:{quote(source_id, safe='')}"


def discover_pdfs(root: Path) -> list[Path]:
    """Discover PDFs while rejecting symlinks, collisions, and unsafe identities."""

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


def extract_sources(root: Path) -> list[KnowledgeSource]:
    """Extract every supported source, failing atomically if any file is invalid."""

    root = root.resolve()
    failures: list[str] = []
    sources: list[KnowledgeSource] = []
    paths = _discover_source_files(root)
    marked_roots = _marked_okf_roots(root, paths)
    for path in paths:
        source_id = unicodedata.normalize("NFC", path.relative_to(root).as_posix())
        try:
            if any(parent == path.parent or parent in path.parents for parent in marked_roots):
                raise ValueError("not-yet-supported import boundary")
            if path.suffix.casefold() == ".md":
                raw = path.read_bytes()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("Markdown source is not valid UTF-8") from exc
                if not text.strip():
                    raise ValueError("Markdown source is empty")
                sources.append(
                    MarkdownSource(
                        id=source_id,
                        resource=logical_resource(source_id, "markdown"),
                        content_sha256=sha256_bytes(raw),
                        evidence=_markdown_evidence(text),
                    )
                )
                continue
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
        raise ValidationFailure("Invalid Knowledge Source set:\n- " + "\n- ".join(failures))
    return sources


def _discover_source_files(root: Path) -> list[Path]:
    """Discover supported PDF and Markdown sources with normalized identity checks."""

    if not root.is_dir():
        raise ValidationFailure(f"Source directory does not exist: {root}")
    found: list[Path] = []
    identities: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValidationFailure(f"Symlinks are not allowed in the source tree: {path}")
        if not path.is_file() or path.suffix.casefold() not in {".pdf", ".md"}:
            continue
        relative = path.relative_to(root).as_posix()
        normalized = unicodedata.normalize("NFC", relative).casefold()
        if normalized in identities:
            raise ValidationFailure(
                f"Knowledge Source identity collision: {identities[normalized]} and {path}"
            )
        identities[normalized] = path
        found.append(path)
    if not found:
        raise ValidationFailure(f"No Knowledge Source files found under: {root}")
    return found


def _marked_okf_roots(root: Path, paths: list[Path]) -> set[Path]:
    """Find recognized portable OKF roots that must not be imported recursively."""

    roots: set[Path] = set()
    for path in paths:
        if path.name != "index.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "okf_version:" in text and "0.2" in text:
            roots.add(path.parent)
    return roots


def _markdown_evidence(text: str) -> list[EvidenceUnit]:
    """Split ordinary Markdown into durable heading blocks and line hints."""

    lines = text.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        stripped = line.rstrip("\r\n")
        if match := re.match(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", stripped):
            headings.append((index, len(match.group(1)), match.group(2)))
    starts = [0] + [item[0] for item in headings]
    blocks: list[EvidenceUnit] = []
    paths: list[tuple[int, str]] = []
    occurrences: dict[tuple[tuple[str, ...], int], int] = {}
    for block_index, start in enumerate(starts):
        if block_index == 0:
            end = headings[0][0] if headings else len(lines)
            path: list[str] = []
            occurrence = 1
        else:
            heading_index, level, title = headings[block_index - 1]
            next_same_or_higher = next(
                (item[0] for item in headings[block_index:] if item[1] <= level), len(lines)
            )
            end = next_same_or_higher
            paths = [item for item in paths if item[0] < level]
            paths.append((level, title))
            path = [item[1] for item in paths]
            key = (tuple(path), level)
            occurrences[key] = occurrences.get(key, 0) + 1
            occurrence = occurrences[key]
        block_text = "".join(lines[start:end])
        locator = MarkdownBlockLocator(
            heading_path=path,
            occurrence=occurrence,
            content_sha256=sha256_text(block_text),
        )
        blocks.append(
            EvidenceUnit(locator=locator, text=block_text, line_start=start + 1, line_end=end)
        )
    return blocks


def source_set_hash(sources: list[KnowledgeSource]) -> str:
    """Hash source identities and content digests in authoritative order."""

    material = "\n".join(f"{item.id}\0{item.content_sha256}" for item in sources)
    return sha256_text(material)


class EvidenceIndex:
    """Ephemeral full-text index for typed Knowledge Source evidence."""

    def __init__(self, database: Path) -> None:
        self._database = database
        with self._connect() as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE evidence USING fts5("
                "source_id UNINDEXED, locator UNINDEXED, text)"
            )

    def _connect(self) -> sqlite3.Connection:
        """Create a connection owned by the calling thread."""
        return sqlite3.connect(self._database)

    def add(self, sources: list[KnowledgeSource]) -> None:
        """Index typed evidence locators and text in the ephemeral SQLite FTS table."""

        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO evidence(source_id, locator, text) VALUES (?, ?, ?)",
                [
                    (
                        source.id,
                        json.dumps(
                            unit.locator.model_dump(mode="json"),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        unit.text,
                    )
                    for source in sources
                    for unit in source.evidence
                ],
            )

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        """Search evidence text and return bounded snippets with typed locators."""

        limit = min(max(limit, 1), 25)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT source_id, locator, snippet(evidence, 2, '[', ']', ' … ', 32) "
                    "FROM evidence WHERE evidence MATCH ? ORDER BY rank LIMIT ?",
                    (query, limit),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            raise SearchQueryFailure(f"Invalid full-text search query: {query!r}") from exc
        return [
            {"source_id": row[0], "locator": json.loads(row[1]), "snippet": row[2]} for row in rows
        ]

    def read(
        self, source_id: str, locators: list[EvidenceLocator | int]
    ) -> list[dict[str, object]]:
        """Read selected evidence, accepting legacy PDF-source locator values."""

        if not locators:
            return []
        typed_locators = [
            locator if isinstance(locator, PDFPageLocator) else PDFPageLocator(page=locator)
            for locator in locators
        ]
        serialized = [
            json.dumps(locator.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            for locator in typed_locators
        ]
        placeholders = ",".join("?" for _ in serialized)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT source_id, locator, text FROM evidence "  # noqa: S608 - placeholders below
                f"WHERE source_id = ? AND locator IN ({placeholders}) ORDER BY locator",
                (source_id, *serialized),
            ).fetchall()
        return [
            {"source_id": row[0], "locator": json.loads(row[1]), "text": row[2]} for row in rows
        ]

    def close(self) -> None:
        """Retained for the context-manager API; connections are operation-scoped."""

    def __enter__(self) -> EvidenceIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PageIndex(EvidenceIndex):
    """Compatibility view of the typed evidence index for legacy page tools."""

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        """Return typed-index search results in the legacy page-shaped format."""

        return [
            {
                "source_id": item["source_id"],
                "page": item["locator"]["page"],
                "snippet": item["snippet"],
            }
            for item in super().search(query, limit)
        ]

    def read(self, source_id: str, pages: list[int | PDFPageLocator]) -> list[dict[str, object]]:
        """Read legacy page locators through the compatibility interface for older tools."""

        rows = super().read(source_id, pages)
        return [
            {"source_id": item["source_id"], "page": item["locator"]["page"], "text": item["text"]}
            for item in rows
        ]
