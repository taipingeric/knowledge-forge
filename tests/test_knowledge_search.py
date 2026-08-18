from pathlib import Path

from knowledge_forge.knowledge_search import search_concepts


def _write_concept(bundle: Path, slug: str, title: str, body: str) -> None:
    path = bundle / "concepts" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntype: Concept\ntitle: {title}\ndescription: d\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_search_matches_on_title(tmp_path: Path) -> None:
    _write_concept(tmp_path, "deadlocks", "Deadlocks", "Transactions can conflict.")
    _write_concept(tmp_path, "mvcc", "MVCC", "Snapshots avoid blocking readers.")

    matches = search_concepts(tmp_path, ["deadlock"])

    assert [match["concept_id"] for match in matches] == ["concepts/deadlocks"]
    assert matches[0]["title"] == "Deadlocks"


def test_search_matches_on_body(tmp_path: Path) -> None:
    _write_concept(tmp_path, "mvcc", "MVCC", "Multi-version concurrency control uses snapshots.")

    matches = search_concepts(tmp_path, ["snapshots"])

    assert [match["concept_id"] for match in matches] == ["concepts/mvcc"]
    assert "snapshots" in matches[0]["snippet"].casefold()


def test_search_returns_no_matches_for_unrelated_keywords(tmp_path: Path) -> None:
    _write_concept(tmp_path, "mvcc", "MVCC", "Multi-version concurrency control uses snapshots.")

    assert search_concepts(tmp_path, ["replication"]) == []


def test_search_over_empty_or_missing_knowledge_folder(tmp_path: Path) -> None:
    assert search_concepts(tmp_path, ["anything"]) == []

    missing = tmp_path / "does-not-exist"
    assert search_concepts(missing, ["anything"]) == []


def test_search_ignores_blank_keywords_without_matching_everything(tmp_path: Path) -> None:
    _write_concept(tmp_path, "mvcc", "MVCC", "Multi-version concurrency control uses snapshots.")

    assert search_concepts(tmp_path, ["  ", ""]) == []


def test_search_snippet_radius_is_configurable(tmp_path: Path) -> None:
    body = "Before context words here. Snapshots avoid blocking readers. After context words here."
    _write_concept(tmp_path, "mvcc", "MVCC", body)

    narrow = search_concepts(tmp_path, ["snapshots"], snippet_radius=5)
    wide = search_concepts(tmp_path, ["snapshots"], snippet_radius=80)

    assert len(narrow[0]["snippet"]) < len(wide[0]["snippet"])
    assert "Before context" not in narrow[0]["snippet"]
    assert "Before context" in wide[0]["snippet"]
