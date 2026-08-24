from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from threading import Lock


class ProgressReporter:
    """Deliver progress messages serially across concurrent processing tasks."""

    def __init__(self, report: Callable[[str], None]) -> None:
        self._report = report
        self._lock = Lock()

    def report(self, message: str) -> None:
        """Emit one message while holding the lock that serializes concurrent reports."""

        with self._lock:
            self._report(message)


class ProcessingTimer:
    """Report completed processing phases using a monotonic clock."""

    def __init__(
        self, report: Callable[[str], None] | ProgressReporter, clock: Callable[[], float]
    ) -> None:
        self.reporter = report if isinstance(report, ProgressReporter) else ProgressReporter(report)
        self._clock = clock
        self._started_at = clock()

    @contextmanager
    def phase(self, label: str) -> Iterator[None]:
        """Time a successful phase and report its duration after the phase completes."""

        started_at = self._clock()
        try:
            yield
        except Exception:
            raise
        else:
            self.reporter.report(f"{label} completed in {self._clock() - started_at:.3f}s.")

    def report_total(self) -> None:
        """Report elapsed time since this timer was created."""

        self.reporter.report(f"Total processing time: {self._clock() - self._started_at:.3f}s.")


def processing_phase(timer: ProcessingTimer | None, label: str):
    """Use a timed phase when configured, otherwise provide a no-op context manager."""

    return timer.phase(label) if timer is not None else nullcontext()
