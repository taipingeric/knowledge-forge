import pytest

from knowledge_forge.errors import ValidationFailure
from knowledge_forge.security import endpoint_identity, reject_tracing


def test_endpoint_identity_drops_query_and_credentials() -> None:
    assert endpoint_identity("https://example.test/v1/?token=secret") == "https://example.test/v1"


def test_tracing_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(ValidationFailure, match="tracing"):
        reject_tracing()
