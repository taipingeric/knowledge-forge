from knowledge_forge.models import ConceptDraft, Evidence, PDFSource, SourcePage
from knowledge_forge.okf import (
    compact_ranges,
    expand_ranges,
    parse_markdown,
    render_concept,
    validate_concept,
)
from knowledge_forge.sources import logical_resource, sha256_text


def source() -> PDFSource:
    return PDFSource(
        id="policies/refunds.pdf",
        resource=logical_resource("policies/refunds.pdf"),
        content_sha256=sha256_text("pdf"),
        pages=[SourcePage(number=index, text=f"Page {index}") for index in range(1, 6)],
    )


def test_page_ranges_round_trip() -> None:
    assert compact_ranges([1, 2, 3, 5]) == ["1-3", "5"]
    assert expand_ranges(["1-3", "5"]) == [1, 2, 3, 5]


def test_rendered_concept_has_okf_provenance() -> None:
    pdf = source()
    draft = ConceptDraft(
        slug="refund-policy",
        title="Refund Policy",
        type="Policy",
        description="The rules for refunds.",
        body=(
            "# Rule\n\nRefunds take seven days.[^policies/refunds.pdf@p2]\n\n"
            "[^policies/refunds.pdf@p2]: Refund policy, page 2"
        ),
        evidence=[Evidence(source_id=pdf.id, pages=[2])],
    )
    raw = render_concept(draft, {pdf.id: pdf}, "knowledge-forge/test")
    metadata, _ = parse_markdown(raw)
    assert metadata["sources"][0]["resource"] == logical_resource(pdf.id)
    assert validate_concept(raw, "concepts/refund-policy", {pdf.id: 5}) == []


def test_citation_source_id_may_contain_at_sign() -> None:
    pdf = source().model_copy(update={"id": "policies/team@example.pdf"})
    pdf.resource = logical_resource(pdf.id)
    draft = ConceptDraft(
        slug="refund-policy",
        title="Refund Policy",
        type="Policy",
        description="The rules for refunds.",
        body=(
            "# Rule\n\nSeven days.[^policies/team@example.pdf@p2]\n\n"
            "[^policies/team@example.pdf@p2]: Policy, page 2"
        ),
        evidence=[Evidence(source_id=pdf.id, pages=[2])],
    )
    raw = render_concept(draft, {pdf.id: pdf}, "knowledge-forge/test")
    assert validate_concept(raw, "concepts/refund-policy", {pdf.id: 5}) == []


def test_repeated_invalid_citation_is_reported_once() -> None:
    pdf = source()
    draft = ConceptDraft(
        slug="refund-policy",
        title="Refund Policy",
        type="Policy",
        description="The rules for refunds.",
        body=("# Rule\n\nFirst claim.[^hpm@p2] Second claim.[^hpm@p2]\n\n[^hpm@p2]: HPM, page 2"),
        evidence=[Evidence(source_id=pdf.id, pages=[2])],
    )
    raw = render_concept(draft, {pdf.id: pdf}, "knowledge-forge/test")

    errors = validate_concept(raw, "concepts/refund-policy", {pdf.id: 5})

    assert errors.count("concepts/refund-policy: citation references missing source hpm") == 1
