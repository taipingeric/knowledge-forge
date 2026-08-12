class KnowledgeForgeError(Exception):
    """Expected user-facing failure."""


class ValidationFailure(KnowledgeForgeError):
    """Bundle, source, or state validation failed."""


class ReconciliationRequired(KnowledgeForgeError):
    """Human input is required before an update can be published."""

    def __init__(self, report_path: str) -> None:
        self.report_path = report_path
        super().__init__(f"Reconciliation required: {report_path}")
