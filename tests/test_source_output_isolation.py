from pathlib import Path

from typer.testing import CliRunner

from knowledge_forge import cli


def test_validate_with_source_rejects_resolved_overlap_before_discovery(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "workspace"
    (output / "nested").mkdir(parents=True)
    marker = output / "unchanged.txt"
    marker.write_text("preserve live files\n")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "validate",
            "--source",
            "workspace/nested/..",
            "--out",
            "workspace",
        ],
    )

    assert result.exit_code == 2
    assert "disjoint" in result.stderr
    assert marker.read_text() == "preserve live files\n"
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == [
        "workspace",
        "workspace/nested",
        "workspace/unchanged.txt",
    ]
