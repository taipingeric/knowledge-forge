import os
from pathlib import Path

import pytest
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


def test_load_local_env_reads_only_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (tmp_path / ".env").write_text("OPENAI_MODEL=parent-model\n")
    monkeypatch.chdir(child)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert cli.load_local_env() is False
    assert os.getenv("OPENAI_MODEL") is None

    (child / ".env").write_text("OPENAI_MODEL=local-model\n")
    assert cli.load_local_env() is True
    assert os.getenv("OPENAI_MODEL") == "local-model"


def test_load_local_env_does_not_override_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")

    assert cli.load_local_env() is True
    assert os.getenv("OPENAI_API_KEY") == "from-process"


def test_console_entrypoint_loads_dotenv_before_starting_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    observed: list[str | None] = []
    monkeypatch.setattr(cli, "app", lambda: observed.append(os.getenv("OPENAI_MODEL")))

    cli.main()

    assert observed == ["dotenv-model"]
