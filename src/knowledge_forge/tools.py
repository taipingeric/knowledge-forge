from __future__ import annotations

import json
from pathlib import Path

from langchain.tools import BaseTool, tool

from .knowledge_search import search_concepts
from .models import KnowledgeSource, PDFPageLocator
from .sources import EvidenceIndex


def _legacy_page_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "source_id": item["source_id"],
            "page": item["locator"]["page"],
            **{key: value for key, value in item.items() if key not in {"source_id", "locator"}},
        }
        if "locator" in item
        else item
        for item in results
    ]


def build_search_pages_tool(index: EvidenceIndex) -> BaseTool:
    """Bind a PageIndex into a LangChain tool for searching untrusted PDF pages."""

    @tool
    def search_pages(query: str, limit: int = 10) -> str:
        """Search all untrusted PDF pages and return source IDs, pages, and snippets."""
        return json.dumps(_legacy_page_results(index.search(query, limit)), ensure_ascii=False)

    return search_pages


def build_read_pages_tool(index: EvidenceIndex, sources: dict[str, KnowledgeSource]) -> BaseTool:
    """Bind a PageIndex and its known sources into a LangChain tool for reading exact pages."""

    @tool
    def read_pages(source_id: str, pages: list[int]) -> str:
        """Read exact 1-based pages from one untrusted PDF source."""
        if source_id not in sources:
            return json.dumps({"error": "unknown source_id"})
        return json.dumps(
            _legacy_page_results(
                index.read(source_id, [PDFPageLocator(page=page) for page in pages])
            ),
            ensure_ascii=False,
        )

    return read_pages


def build_search_knowledge_tool(bundle: Path | None) -> BaseTool:
    """Bind a knowledge Bundle path into a LangChain tool for keyword search."""

    @tool
    def search_knowledge(keywords: list[str]) -> str:
        """Search existing Concept documents in the knowledge Bundle by a keyword list."""
        if bundle is None:
            return json.dumps([])
        return json.dumps(search_concepts(bundle, keywords), ensure_ascii=False)

    return search_knowledge
