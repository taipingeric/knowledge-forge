from knowledge_forge.models import ConceptDraft, Evidence, PDFSource, SourcePage
from knowledge_forge.okf import (
    compact_ranges,
    dump_markdown,
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
            "# Rule\n\nRefunds take seven days.[^policies/refunds.pdf#pdf_page:2]\n\n"
            "[^policies/refunds.pdf#pdf_page:2]: Refund policy, page 2"
        ),
        evidence=[Evidence(source_id=pdf.id, pages=[2])],
    )
    raw = render_concept(draft, {pdf.id: pdf}, "knowledge-forge/test")
    metadata, _ = parse_markdown(raw)
    assert metadata["sources"][0]["resource"] == logical_resource(pdf.id)
    assert metadata["sources"][0]["id"] == "policies/refunds.pdf#pdf_page:2"
    assert metadata["sources"][0]["locator"] == {"kind": "pdf_page", "page": 2}
    assert len(metadata["sources"][0]["locator_sha256"]) == 64
    assert validate_concept(raw, "concepts/refund-policy", {pdf.id: 5}) == []


def test_source_reference_id_escapes_source_identity_characters() -> None:
    pdf = source().model_copy(update={"id": "policies/team@example.pdf"})
    pdf.resource = logical_resource(pdf.id)
    draft = ConceptDraft(
        slug="refund-policy",
        title="Refund Policy",
        type="Policy",
        description="The rules for refunds.",
        body=(
            "# Rule\n\nSeven days.[^policies/team%40example.pdf#pdf_page:2]\n\n"
            "[^policies/team%40example.pdf#pdf_page:2]: Policy, page 2"
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

    assert (
        errors.count("concepts/refund-policy: citation references missing source reference hpm@p2")
        == 1
    )


def test_validation_rejects_missing_definition_and_out_of_bounds_page() -> None:
    pdf = source()
    draft = ConceptDraft(
        slug="refund-policy",
        title="Refund Policy",
        type="Policy",
        description="The rules for refunds.",
        body="# Rule\n\nSeven days.[^policies/refunds.pdf#pdf_page:2]",
        evidence=[Evidence(source_id=pdf.id, pages=[2])],
    )
    raw = render_concept(draft, {pdf.id: pdf}, "knowledge-forge/test")

    errors = validate_concept(raw, "concepts/refund-policy", {pdf.id: 1})

    assert (
        "concepts/refund-policy: citation "
        "policies/refunds.pdf#pdf_page:2 has no footnote definition" in errors
    )
    assert (
        "concepts/refund-policy: page outside source bounds for policies/refunds.pdf#pdf_page:2"
        in errors
    )


def test_validation_rejects_a_source_reference_outside_known_evidence() -> None:
    pdf = source()
    draft = ConceptDraft(
        slug="refund-policy",
        title="Refund Policy",
        type="Policy",
        description="The rules for refunds.",
        body=(
            "# Rule\n\nSeven days.[^unknown.pdf#pdf_page:1]\n\n"
            "[^unknown.pdf#pdf_page:1]: Unknown policy, page 1"
        ),
        evidence=[Evidence(source_id=pdf.id, pages=[2])],
    )
    metadata, body = parse_markdown(render_concept(draft, {pdf.id: pdf}, "knowledge-forge/test"))
    metadata["sources"][0]["id"] = "unknown.pdf#pdf_page:1"
    metadata["sources"][0]["resource"] = logical_resource("unknown.pdf")
    metadata["sources"][0]["locator"] = {"kind": "pdf_page", "page": 1}
    metadata["sources"][0]["locator_sha256"] = sha256_text('{"kind":"pdf_page","page":1}')

    errors = validate_concept(dump_markdown(metadata, body), "concepts/refund-policy", {pdf.id: 5})

    assert "concepts/refund-policy: source reference is outside referenced evidence" in errors
