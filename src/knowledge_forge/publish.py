from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import ValidationFailure


def _lock_owner_is_alive(lock: Path) -> bool:
    """Return whether the process recorded in a mutation lock is still alive."""

    try:
        line = lock.read_text(encoding="utf-8").strip()
        pid = int(line.removeprefix("pid="))
        os.kill(pid, 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _journal_path(output: Path) -> Path:
    """Return the transaction journal path used for publication recovery."""

    return output.parent / f".{output.name}.transaction.json"


def _recover_interrupted_publication(output: Path) -> None:
    """Recover or reject an interrupted atomic publication from its journal."""

    journal = _journal_path(output)
    if not journal.exists():
        return
    try:
        transaction = json.loads(journal.read_text(encoding="utf-8"))
        staging = Path(transaction["staging"])
        backup = Path(transaction["backup"])
    except Exception as exc:
        raise ValidationFailure(f"Invalid publication recovery journal: {journal}") from exc

    if output.exists():
        if backup.exists():
            shutil.rmtree(backup)
    elif backup.exists():
        backup.rename(output)
    else:
        raise ValidationFailure(
            f"Cannot recover interrupted publication; output and backup are missing: {journal}"
        )
    if staging.exists():
        shutil.rmtree(staging)
    journal.unlink()


@contextmanager
def output_lock(output: Path) -> Iterator[None]:
    """Serialize Bundle mutations and recover interrupted publication before yielding."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.knowledge-forge.lock"
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError as exc:
            if attempt == 0 and not _lock_owner_is_alive(lock):
                lock.unlink(missing_ok=True)
                continue
            raise ValidationFailure(
                f"Another Knowledge Forge mutation holds the lock: {lock}"
            ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        _recover_interrupted_publication(output)
        yield
    finally:
        lock.unlink(missing_ok=True)


@contextmanager
def staged_bundle(output: Path, *, copy_existing: bool) -> Iterator[Path]:
    """Create a temporary sibling staging tree and remove it if the operation fails."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        if copy_existing:
            if not output.is_dir():
                raise ValidationFailure(f"Bundle does not exist: {output}")
            shutil.copytree(output, temporary, dirs_exist_ok=True)
        yield temporary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def publish_staging(staging: Path, output: Path) -> None:
    """Publish a staged Bundle with journaled rename and rollback protection."""

    output = output.resolve()
    backup = output.parent / f".{output.name}.backup"
    if backup.exists():
        raise ValidationFailure(
            f"Recovery backup already exists; inspect it before continuing: {backup}"
        )
    journal = _journal_path(output)
    journal.write_text(
        json.dumps(
            {"output": str(output), "staging": str(staging), "backup": str(backup)},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    moved_old = False
    try:
        if output.exists():
            output.rename(backup)
            moved_old = True
        staging.rename(output)
    except Exception:
        if moved_old and backup.exists() and not output.exists():
            backup.rename(output)
        if output.exists() or not moved_old:
            journal.unlink(missing_ok=True)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    journal.unlink()
