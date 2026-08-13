# Knowledge Forge

English | [繁體中文](README-zh.md)

Knowledge Forge turns team-approved PDF documents into a wiki that conforms to Google Open Knowledge Format 0.2. LangGraph orchestrates a fixed workflow with one LangChain reasoning-agent role for cross-document concept planning and synthesis. Each planning or Concept synthesis task runs in an isolated reasoning session.

Agents and people jointly maintain the resulting Bundle. A later update never silently overwrites human edits. A deterministic three-way merge combines non-overlapping changes. A conflict in the same structural block leaves the live Bundle unchanged and creates an auditable reconciliation workspace.

## MVP scope

- Input: recursively discovered PDFs with complete text layers under a specified directory.
- Output: an OKF 0.2 Markdown Bundle. The MVP does not include a Web UI, chat, RAG API, or source PDFs.
- Model: an OpenAI-compatible endpoint that supports tool calling through the Responses API. Chat Completions is not used.
- Retrieval: a temporary SQLite FTS5 page index created for each run and deleted afterward.
- Concept types: `Concept`, `Definition`, `Policy`, `Procedure`, and `FAQ`.

Image-only scanned PDFs, encrypted PDFs, damaged files, documents with no extractable text, symlinks, and normalized path collisions cause the complete operation to fail atomically. A PDF with extractable text can contain intentionally blank pages.

PDF content is untrusted data. The agent has no shell, network, or arbitrary file-writing tools. Extracted PDF text is sent to the configured model endpoint. Knowledge Forge does not write PDFs, extracted full text, or API keys into the Bundle. Model requests always use the Responses API with `store: false`. Knowledge Forge refuses to run when LangSmith or LangChain tracing is enabled, which prevents additional transmission of source content.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required:

```bash
uv sync
```

The model name must be explicit. API keys are not written into the Bundle or state:

```bash
cp .env.example .env
```

Edit `.env` in the current working directory:

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=...
# Optional:
OPENAI_BASE_URL=https://models.example/v1
```

Knowledge Forge reads only the `.env` in the command's working directory. It does not search parent directories. CLI arguments and existing process environment variables take precedence over `.env`, so local files do not override values injected by CI, containers, or secret managers.

A custom `OPENAI_BASE_URL` must support the OpenAI Responses API (`/v1/responses`), function calling, and structured tool output. A service that implements only `/v1/chat/completions` cannot be used. Knowledge Forge allows parallel tool calls by default and preserves every model-issued call with its matching result during Responses replay.

You can also use the process environment directly:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
# Optional: export OPENAI_BASE_URL=https://models.example/v1
```

## CLI

The model and endpoint for `generate` and `update` can also be specified on the command line:

```bash
uv run knowledge-forge generate \
  --source ./pdfs \
  --out ./knowledge \
  --model <model> \
  --base-url https://models.example/v1
```

Although the CLI accepts `--api-key`, prefer `.env`, `OPENAI_API_KEY`, or a production secret manager so credentials do not remain in shell history. Git ignores `.env`; never put a real secret in `.env.example`.

Create a new Bundle. `--out` must not exist or must be an empty directory:

```bash
uv run knowledge-forge generate \
  --source ./pdfs \
  --out ./knowledge \
  --language auto \
  --max-agent-steps 50 \
  --concept-concurrency 4
```

For `generate` and agent-backed `update`, `--max-agent-steps` is the maximum number of model calls for each reasoning task. Concept planning receives one independent budget, and every Concept synthesis receives a fresh budget. A complex Concept therefore cannot consume the model calls available to later Concepts. If a task exceeds its limit, the error identifies planning or the affected Concept and the candidate Bundle is not published.

After planning, `generate` synthesizes up to four independent Concept Documents concurrently by default. Set `--concept-concurrency 1` to run synthesis sequentially, or select another positive limit to balance throughput against provider rate limits. Tasks may complete in any order, but the Bundle is assembled in ConceptPlan order and is published only when every Concept succeeds. The selected concurrency is part of Generation Identity.

Concept concurrency is separate from parallel tool calls. `--concept-concurrency` controls how many Concept synthesis tasks run at once. Within each task, parallel tool-call mode controls whether one model turn may request several `search_pages` or `read_pages` calls.

For a Responses-to-Bedrock gateway that cannot replay multiple tool results, explicitly select non-parallel compatibility mode:

```bash
uv run knowledge-forge generate \
  --source ./pdfs \
  --out ./knowledge \
  --no-parallel-tool-calls
```

Compatibility mode sends `parallel_tool_calls: false`. If the model or gateway ignores that request and still returns multiple calls, LangChain middleware deterministically keeps one call and its matching result for that turn. Provider API errors fail immediately; Knowledge Forge never switches modes automatically. The selected mode is shown on stderr and stored in the Generation Identity.

`generate` and `update` show the current phase on stderr. After the concept count is known, they also show synthesis progress:

```text
[knowledge-forge] Tool-call mode: parallel.
[knowledge-forge] Reading PDF sources...
[knowledge-forge] Loaded 3 PDFs with 84 pages.
[knowledge-forge] Indexing 84 pages from 3 PDFs...
[knowledge-forge] Planning concepts with the reasoning agent...
[knowledge-forge] Planned 6 concepts in Traditional Chinese.
[knowledge-forge] Synthesizing concept 1/6: refund-policy
```

Each completed phase includes its duration, and the command ends with a total-duration summary:

```text
[knowledge-forge] PDF Source reading completed in 0.214s.
[knowledge-forge] PDF indexing completed in 0.038s.
[knowledge-forge] Concept planning completed in 12.481s.
[knowledge-forge] Concept synthesis 1/6 (refund-policy) completed in 8.327s.
[knowledge-forge] Concept rendering and validation completed in 0.006s.
[knowledge-forge] Candidate Bundle writing and validation completed in 0.014s.
[knowledge-forge] Atomic publication completed in 0.002s.
[knowledge-forge] Total processing time: 61.842s.
```

On failure, completed phases and total elapsed time are still reported while the existing error and exit code are preserved. These messages do not contain full PDF text, model responses, or API keys. Timing and progress are operational stderr output only; they do not enter the OKF Bundle, Agent Baseline, Generation Identity, or private state. The final command result remains on stdout for use in shell pipelines.

Update from the complete authoritative PDF set. `--source` is not the set of files changed in this run. It is the complete set of PDFs that the Team Knowledge Wiki must currently use:

```bash
uv run knowledge-forge update \
  --source ./pdfs \
  --out ../knowledge-base
```

When the PDF Sources, Bundle, state, and Generation Identity are all unchanged, `update` does not call the model or write files. People can edit Concept body text and the `type`, `title`, `description`, `tags`, and `status` fields. Do not directly edit `generated`, `sources`, hashes, page mappings, `verified`, `index.md`, `log.md`, or `.knowledge-forge/`.

A valid document that a person adds under `concepts/` is registered as a permanent Human-owned Concept during the next mutation. A normal `update` does not rewrite, delete, or adopt it. The MVP does not yet provide an ownership adoption command.

Run deterministic validation without calling a model or changing files:

```bash
uv run knowledge-forge validate --out ./knowledge --source ./pdfs
```

Without `--source`, validation checks the Bundle's internal provenance format, citations, Agent Baselines, and private state. With `--source`, it also checks the complete PDF set, actual SHA-256 values, and page boundaries.

Add human verification to the current Concept version:

```bash
uv run knowledge-forge verify \
  --source ./pdfs \
  --out ./knowledge \
  --concept concepts/refund-policy.md \
  --by human:reviewer-id
```

When a Concept's semantic content, curated metadata, or evidence changes, active `verified` data is removed. Historical events remain in audit state.

## Reconciliation

When a merge cannot be completed safely, `update` returns exit code 3, preserves the live Bundle, and creates:

```text
knowledge.reconciliation.md
knowledge.reconciliation/
  manifest.json
  resolution.yaml
  pending/
  manual/
```

Select one action for each conflict in `resolution.yaml`:

- `keep-human`: create a Conditional Override bound to the human block and evidence hash.
- `use-source`: use the existing candidate block without calling the model again.
- `manual`: edit the Concept in `manual/` and point `artifact` to that file.

Then run:

```bash
uv run knowledge-forge reconcile \
  --source ./pdfs \
  --out ./knowledge \
  --resolution ./knowledge.reconciliation/resolution.yaml
```

Knowledge Forge rejects a stale resolution if the live Bundle, PDF Source set, pending candidate, or Generation Identity has changed. After success, the pending workspace is deleted and the resolved Markdown report remains for audit.

## Exit codes

- `0`: success, including deterministic `No changes` from `update`.
- `2`: PDF Source, Bundle, state, model output, or operation-argument validation failed.
- `3`: `update` requires human reconciliation; the live Bundle remains unchanged.

## Bundle and provenance

```text
knowledge/
  index.md
  log.md
  concepts/<stable-kebab-case-slug>.md
  .knowledge-forge/
    state.json
    baseline/<concept-id>.json
```

`.knowledge-forge/state.json` stores the workflow and Generation Identity, source dependencies, ownership, overrides, and verification audit. A deterministic checksum detects unexpected changes. `baseline/` wraps complete agent-authored Markdown in JSON so OKF consumers do not treat it as a public Concept Document. Both locations are tool-managed state and must not be edited manually.

Each source entry uses a durable logical URN and records the public content hash and page ranges:

```yaml
sources:
  - id: policies/refunds.pdf
    resource: urn:knowledge-forge:pdf:policies%2Frefunds.pdf
    content_sha256: <sha256>
    pages: [2-4, "7"]
```

Material, disputed, numeric, policy, or version-sensitive statements use page-level footnotes such as `[^policies/refunds.pdf@p3]`, with a matching footnote definition in the body.

## TODO

- [x] Make parallel tool calls configurable for `generate`. The default preserves and replays all parallel calls; `--no-parallel-tool-calls` selects deterministic single-call compatibility for affected gateways. Provider failures never trigger an automatic fallback.
- [x] Add processing-time statistics for `generate`. PDF reading, indexing, concept planning, each Concept synthesis, render and validation, candidate writing and validation, publication, and total duration are reported without changing deterministic artifacts.

## Documentation maintenance

`README.md` is the English document and `README-zh.md` is the Traditional Chinese document. Any change to product behavior, CLI usage, format, installation, security guidance, TODO items, or operational instructions must update both files in the same change. Keep their section order and examples equivalent, and keep the language links at the top of both files.

## Development

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

Architecture decisions are in [`docs/adr/`](docs/adr/). Domain language is in [`CONTEXT.md`](CONTEXT.md).
