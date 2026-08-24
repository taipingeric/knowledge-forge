from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKFLOW_VERSION = "3"
STATE_VERSION = 1
CONCEPT_TYPES = ("Concept", "Definition", "Policy", "Procedure", "FAQ")


class SourceKind(StrEnum):
    PDF = "pdf"


class PDFPageLocator(BaseModel):
    kind: Literal["pdf_page"] = "pdf_page"
    page: int = Field(ge=1)


EvidenceLocator = PDFPageLocator


class EvidenceUnit(BaseModel):
    locator: EvidenceLocator
    text: str


class KnowledgeSource(BaseModel):
    kind: SourceKind
    id: str
    resource: str
    content_sha256: str
    evidence: list[EvidenceUnit]

    @property
    def source_identity(self) -> str:
        """Return the stable source-root-relative identity."""
        return self.id


class SourcePage(BaseModel):
    number: int = Field(ge=1)
    text: str


PDFPage = SourcePage


class PDFSource(KnowledgeSource):
    kind: Literal[SourceKind.PDF] = SourceKind.PDF
    pages: list[SourcePage]
    evidence: list[EvidenceUnit] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_evidence(self) -> PDFSource:
        self.evidence = [
            EvidenceUnit(locator=PDFPageLocator(page=page.number), text=page.text)
            for page in self.pages
        ]
        return self


class Evidence(BaseModel):
    source_id: str
    pages: list[int] = Field(min_length=1)

    @field_validator("pages")
    @classmethod
    def normalize_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("page numbers are 1-based")
        return sorted(set(value))


class PlannedConcept(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str
    type: Literal["Concept", "Definition", "Policy", "Procedure", "FAQ"]
    description: str
    search_queries: list[str] = Field(min_length=1, max_length=8)


class ConceptPlan(BaseModel):
    language: str = Field(min_length=2)
    concepts: list[PlannedConcept] = Field(min_length=1)


class ConceptDraft(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str
    type: Literal["Concept", "Definition", "Policy", "Procedure", "FAQ"]
    description: str
    tags: list[str] = Field(default_factory=list)
    body: str
    evidence: list[Evidence] = Field(min_length=1)


class BaselineSnapshot(BaseModel):
    concept_id: str
    raw_markdown: str
    sha256: str


class ConditionalOverride(BaseModel):
    concept_id: str
    block_id: str
    human_hash: str
    evidence_hash: str


class VerificationEvent(BaseModel):
    concept_id: str
    by: str
    at: datetime
    version_hash: str


class ConceptState(BaseModel):
    ownership: Literal["agent", "human"]
    deleted: bool = False
    deletion_candidate_hash: str | None = None
    baseline_hash: str | None = None
    source_dependencies: dict[str, str] = Field(default_factory=dict)
    managed_fields_hash: str | None = None


class SourceState(BaseModel):
    content_sha256: str
    page_count: int = Field(ge=1)


class GenerationIdentity(BaseModel):
    workflow_version: str = WORKFLOW_VERSION
    model: str
    endpoint: str
    language: str
    output_language: str | None = None
    max_agent_steps: int
    parallel_tool_calls: bool = False
    concept_concurrency: int = Field(default=1, ge=1)


class ForgeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_version: int = STATE_VERSION
    integrity_hash: str = ""
    generation: GenerationIdentity
    source_set_hash: str
    sources: dict[str, SourceState]
    bundle_hash: str
    tool_files: dict[str, str]
    concepts: dict[str, ConceptState]
    overrides: list[ConditionalOverride] = Field(default_factory=list)
    verification_history: list[VerificationEvent] = Field(default_factory=list)


class Conflict(BaseModel):
    id: str
    concept_id: str
    block_id: str
    baseline: str | None = None
    human: str | None = None
    candidate: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str


class ReconciliationManifest(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    output_path: str
    live_bundle_hash: str
    source_set_hash: str
    candidate_hash: str
    generation: GenerationIdentity
    conflicts: list[Conflict]


class Resolution(BaseModel):
    conflict_id: str
    action: Literal["keep-human", "use-source", "manual"]
    artifact: str | None = None


class ResolutionFile(BaseModel):
    resolutions: list[Resolution]
