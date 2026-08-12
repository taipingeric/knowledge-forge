import json
from pathlib import Path

import pytest

from knowledge_forge.errors import ValidationFailure
from knowledge_forge.publish import output_lock


def test_live_lock_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    with (
        output_lock(output),
        pytest.raises(ValidationFailure, match="holds the lock"),
        output_lock(output),
    ):
        pass


def test_stale_lock_and_interrupted_swap_are_recovered(tmp_path: Path) -> None:
    output = tmp_path / "knowledge"
    backup = tmp_path / ".knowledge.backup"
    staging = tmp_path / ".knowledge.staging-crashed"
    backup.mkdir()
    (backup / "old.txt").write_text("old")
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    (tmp_path / ".knowledge.knowledge-forge.lock").write_text("pid=99999999\n")
    (tmp_path / ".knowledge.transaction.json").write_text(
        json.dumps({"output": str(output), "staging": str(staging), "backup": str(backup)})
    )

    with output_lock(output):
        assert (output / "old.txt").read_text() == "old"
        assert not staging.exists()
        assert not (tmp_path / ".knowledge.transaction.json").exists()
