# Knowledge Forge

A shared knowledge wiki that turns approved team source material into structured, reviewable, and maintainable knowledge for people and agents.

## Language

**Team Knowledge Wiki**:
The shared body of organizational knowledge synthesized from sources approved by a team and represented as an OKF Bundle.
_Avoid_: Repository wiki, personal wiki, general-purpose wiki engine

**Knowledge Source**:
Material approved by the team as evidence from which the Team Knowledge Wiki may be synthesized.
_Avoid_: Input, context, connector

**PDF Source**:
A PDF document approved as a Knowledge Source for the Team Knowledge Wiki. It is evidence for concepts rather than a wiki page to be copied verbatim.
_Avoid_: PDF page, uploaded file, source page

**OKF Bundle**:
A versioned directory tree of Markdown concept documents conforming to Google Open Knowledge Format 0.2, with a root `index.md` that declares `okf_version: "0.2"`.
_Avoid_: Wiki export, document dump

**Concept Document**:
A Markdown document in an OKF Bundle that represents one knowledge concept and declares a non-empty `type` in YAML frontmatter.
_Avoid_: Wiki page, article

**Source Identity**:
The stable source-root-relative identity of a PDF Source, independent of its runtime filesystem location and content version.
_Avoid_: File path, content hash

**Agent Baseline**:
The exact previous agent-authored version of a Concept Document used as the common ancestor in reconciliation.
_Avoid_: Backup, current version

**Human Delta**:
The difference between the current Concept Document and its Agent Baseline that represents authoritative human curation.
_Avoid_: Manual patch, local edit

**Human-owned Concept**:
A Concept Document created by a person and persistently excluded from ordinary agent rewriting or deletion.
_Avoid_: Unmanaged document, custom page

**Reconciliation Conflict**:
A concurrent human and agent change to the same structural block, or another ownership conflict that cannot be merged deterministically.
_Avoid_: Validation error, model failure

**Conditional Override**:
An explicit decision to retain a human-authored block while both that block and the conflicting source evidence remain unchanged.
_Avoid_: Permanent lock, ignored conflict

**Generation Identity**:
The workflow version, model endpoint, model name, language, per-task step budget, tool-call mode, and Concept concurrency that identify how an agent candidate was produced.
_Avoid_: Run ID, model configuration
