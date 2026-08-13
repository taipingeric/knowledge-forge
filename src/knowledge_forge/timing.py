from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext


class ProcessingTimer:
    """Report completed processing phases using a monotonic clock."""

    def __init__(self, report: Callable[[str], None], clock: Callable[[], float]) -> None:
        self._report = report
        self._clock = clock
        self._started_at = clock()

    @contextmanager
    def phase(self, label: str) -> Iterator[None]:
        started_at = self._clock()
        try:
            yield
        except Exception:
            raise
        else:
            self._report(f"{label} completed in {self._clock() - started_at:.3f}s.")

    def report_total(self) -> None:
        self._report(f"Total processing time: {self._clock() - self._started_at:.3f}s.")


def processing_phase(timer: ProcessingTimer | None, label: str):
    return timer.phase(label) if timer is not None else nullcontext()
