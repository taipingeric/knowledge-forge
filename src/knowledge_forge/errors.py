class KnowledgeForgeError(Exception):
    """Base exception for expected, user-facing Knowledge Forge failures."""


class ValidationFailure(KnowledgeForgeError):
    """A Bundle, source, state, or model result failed a required constraint."""


class SearchQueryFailure(ValidationFailure):
    """A full-text query could not be parsed by the temporary search index."""


class ReconciliationRequired(KnowledgeForgeError):
    """Human input is required before an update can publish its candidate Bundle."""

    def __init__(self, report_path: str) -> None:
        self.report_path = report_path
        super().__init__(f"Reconciliation required: {report_path}")
