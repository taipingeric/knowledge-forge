# Knowledge Forge

A shared knowledge wiki that turns approved team source material into structured, reviewable, and maintainable knowledge for people and agents.

## Language

**Team Knowledge Wiki**:
The shared body of organizational knowledge synthesized from sources approved by a team and represented as an OKF Bundle.
_Avoid_: Repository wiki, personal wiki, general-purpose wiki engine

**Knowledge Source**:
Material approved by the team as evidence from which the Team Knowledge Wiki may be synthesized.
_Avoid_: Input, context, connector

**Input Corpus**:
The complete authoritative set of supported source material recursively discovered under the source root for one Knowledge Forge operation.
_Avoid_: PDF directory, changed files, incremental input

**PDF Source**:
A PDF document approved as a Knowledge Source for the Team Knowledge Wiki. It is evidence for concepts rather than a wiki page to be copied verbatim.
_Avoid_: PDF page, uploaded file, source page

**Markdown Source**:
An ordinary Markdown document approved as Knowledge Source evidence that does not qualify as an Imported Concept.
_Avoid_: Markdown input, OKF Concept, existing wiki page

**OKF Bundle**:
A directory tree of Markdown Concept Documents conforming to Google Open Knowledge Format 0.2. A root `index.md` and `okf_version` marker are optional for portable conformance.
_Avoid_: Wiki export, document dump

**Concept Document**:
A Markdown document in an OKF Bundle that represents one knowledge concept and declares a non-empty `type` in YAML frontmatter.
_Avoid_: Wiki page, article

**Imported Concept**:
An existing OKF Concept discovered in the Input Corpus and incorporated as a knowledge artifact without re-synthesis, while preserving its identity, lifecycle, provenance, and citations.
_Avoid_: Markdown Source, copied page, agent-generated Concept

**Recognized Import Bundle**:
An OKF Bundle in the Input Corpus that passes Portable OKF Validation and whose root `index.md` explicitly declares `okf_version: "0.2"`, signaling that all of its Concept Documents are intended for automatic import.
_Avoid_: Any conformant Bundle, Markdown directory, source folder

**Markdown Evidence Block**:
A parser-delimited heading section identified by its full heading path, same-level occurrence, and raw-content hash; a synthetic root represents content before the first heading, and line ranges are only human navigation hints.
_Avoid_: Page, line range, chunk

**Source Identity**:
The stable source-root-relative identity of a Knowledge Source, independent of its source kind, runtime filesystem location, and content version.
_Avoid_: File path, content hash

**Source Reference ID**:
A Concept-local identifier for one evidence reference whose exact value joins a claim footnote to its `sources` entry and typed locator.
_Avoid_: Source Identity, citation label extension

**Provenance Chain**:
A sequence of evidence references that connects a generated claim through an Imported Concept to its preserved upstream sources without claiming that Knowledge Forge directly observed them.
_Avoid_: Flattened provenance, copied citations

**Agent Baseline**:
The exact previous agent-authored version of a Concept Document used as the common ancestor in reconciliation.
_Avoid_: Backup, current version

**Imported Baseline**:
The exact previous source-provided version of an Imported Concept used as the common ancestor when reconciling a later import with human curation.
_Avoid_: Agent Baseline, copied source, current version

**Human Delta**:
The difference between the current Concept Document and its Agent Baseline or Imported Baseline that represents authoritative human curation.
_Avoid_: Manual patch, local edit

**Human-owned Concept**:
A Concept Document created by a person and persistently excluded from ordinary agent rewriting or deletion.
_Avoid_: Unmanaged document, custom page

**Reconciliation Conflict**:
A concurrent human and producing-agent or imported-source change to the same structural block, or another ownership conflict that cannot be merged deterministically.
_Avoid_: Validation error, model failure

**Conditional Override**:
An explicit decision to retain a human-authored block while both that block and the conflicting source evidence remain unchanged.
_Avoid_: Permanent lock, ignored conflict

**Generation Identity**:
The Generation Policy Version, model endpoint, model name, language, per-task step budget, tool-call mode, and Concept concurrency that identify how an agent candidate was produced.
_Avoid_: Run ID, model configuration

**State Schema Version**:
The version of the private managed-state representation and its deterministic migration contract.
_Avoid_: Workflow Version, Generation Policy Version

**Workflow Version**:
The version of Knowledge Forge orchestration behavior that does not by itself determine whether existing agent-authored knowledge must be regenerated.
_Avoid_: State Schema Version, Generation Policy Version

**Generation Policy Version**:
The version of prompts, tool contracts, planning rules, and other reasoning semantics that can materially change an agent candidate.
_Avoid_: State Schema Version, Workflow Version

**Regeneration Impact Report**:
An external human- and machine-readable report that distinguishes Concepts with changed referenced evidence, stale planning coverage, and Agent-owned Concepts invalidated by Generation Identity changes.
_Avoid_: Reconciliation report, `stale_after`, lifecycle status

**Portable OKF Validation**:
Assessment of whether any OKF Bundle conforms to Open Knowledge Format 0.2 without requiring Knowledge Forge private state.
_Avoid_: Managed validation, full validation

**Managed Bundle Validation**:
Additional assessment of a Knowledge Forge-managed OKF Bundle's private state, provenance, hashes, manifests, and filesystem consistency.
_Avoid_: OKF conformance, strict validation
