from knowledge_forge.merge import merge_concept
from knowledge_forge.okf import dump_markdown


def concept(title: str, rule: str, notes: str) -> str:
    return dump_markdown(
        {"type": "Policy", "title": title, "generated": {"by": "tool"}, "sources": []},
        f"# Rule\n\n{rule}\n\n# Notes\n\n{notes}\n",
    )


def test_non_overlapping_heading_changes_merge() -> None:
    baseline = concept("Refunds", "Seven days", "Original")
    human = concept("Refunds", "Seven business days", "Original")
    candidate = concept("Refunds", "Seven days", "Updated evidence")
    result = merge_concept("concepts/refunds", baseline, human, candidate)
    assert not result.conflicts
    assert "Seven business days" in result.markdown
    assert "Updated evidence" in result.markdown


def test_same_heading_change_conflicts() -> None:
    baseline = concept("Refunds", "Seven days", "Original")
    human = concept("Refunds", "Fourteen days", "Original")
    candidate = concept("Refunds", "Five days", "Original")
    result = merge_concept("concepts/refunds", baseline, human, candidate)
    assert [item.block_id for item in result.conflicts] == ["body:rule"]
    assert "Fourteen days" in result.markdown
