from pathlib import Path

from typer.testing import CliRunner

from knowledge_forge import cli


def test_validate_command_is_read_only(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "knowledge"
    output.mkdir()
    called: list[Path] = []
    monkeypatch.setattr(cli, "validate_bundle", lambda path, sources: called.append(path))
    result = CliRunner().invoke(cli.app, ["validate", "--out", str(output)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "Bundle is valid"
    assert called == [output.resolve()]
