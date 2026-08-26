# Knowledge Forge

[English](README.md) | 繁體中文

Knowledge Forge 將團隊核准的來源資料整理成符合 Google Open Knowledge Format 0.2 的 Wiki。它使用 LangGraph 編排固定 workflow，並以單一 LangChain reasoning-agent role 負責跨文件的 concept planning 與 synthesis；每個 planning 或 Concept synthesis task 都在隔離的 reasoning session 中執行。

輸出的 Bundle 同時由 agent 與人維護：人工修改不會在下一次更新時被靜默覆蓋。非重疊變更會以 deterministic three-way merge 合併；同一結構區塊的衝突則維持 live Bundle 不變，產生可稽核的 reconciliation workspace。

## Knowledge Sources

Input Corpus 可包含 PDF、一般非空的 UTF-8 Markdown 文件，以及可辨識的 OKF 0.2 import Bundle。一般 Markdown 是 opaque untrusted evidence：frontmatter、HTML、comments 與 body text 都會被索引，但不會執行 includes、抓取 URL、追蹤 links 或讀取 linked images。Markdown evidence 使用 durable heading path、same-level occurrence 與 content hash；顯示的 line ranges 僅供診斷。根 `index.md` 只宣告 `okf_version: "0.2"` 的目錄會被視為 import boundary，而不是一般 evidence。可辨識的 Bundle 會在匯入前先驗證，巢狀的可辨識 Bundle 則從最內層 boundary 匯入。

已辨識的 OKF 0.2 Bundle 會在沒有 model configuration 或 reasoning 的情況下匯入。其 Concept Documents 會保留精確的 UTF-8 Markdown bytes、identity、language、type、extension、provenance、citation、link 與 image syntax；Knowledge Forge 絕不翻譯或修改 imported content。

## MVP 範圍

- 輸入：指定目錄內遞迴發現、具有完整文字層的 PDF 與一般 Markdown；可辨識的 OKF 0.2 Bundle 會在宣告的 boundary 匯入。
- 輸出：OKF 0.2 Markdown Bundle；不包含 Web UI、聊天、RAG API 或原始 PDF。
- 模型：對 agent-backed 操作而言，透過 Responses API 呼叫支援 tool calling 的 OpenAI-compatible endpoint；不使用 Chat Completions。
- Retrieval：每次執行建立暫存 SQLite FTS5 evidence index，結束後刪除。
- Agent 產生的 Concept types：`Concept`、`Definition`、`Policy`、`Procedure`、`FAQ`；imported Concept 保留來源宣告的 type。

純掃描 PDF、加密 PDF、損壞文件、空白或非 UTF-8 Markdown、整份沒有可擷取文字、symlink、格式錯誤的 imported Bundle，以及正規化後路徑或 imported Concept ID 碰撞，都會讓整次操作原子失敗。具有文字內容的 PDF 可以包含刻意留白的頁面。

來源內容被視為不可信資料，agent 沒有 shell、network 或任意檔案寫入工具。需要 reasoning 時，擷取出的來源 evidence 會送往設定的模型 endpoint；只有 import 的操作不會呼叫模型。Knowledge Forge 不會把來源檔案、擷取全文或 API key 寫入 Bundle。模型請求固定使用 Responses API 並設定 `store: false`。為避免內容被額外傳送，偵測到 LangSmith／LangChain tracing 開啟時會拒絕執行。

## 來源與輸出隔離

所有需要來源的操作（`generate`、`update`、`reconcile`、`verify`，以及提供 `--source` 時的 `validate`）都要求解析後的 `--source` 與 `--out` 目錄樹彼此完全分離。兩個路徑相同、輸出位於來源之下，或來源位於輸出之下時，Knowledge Forge 都會拒絕執行。系統會先解析兩個路徑，因此 `sources/../sources` 之類的相對路徑寫法無法繞過檢查。

拒絕會發生在來源探索、模型設定驗證、模型呼叫、staging 或報告建立之前，因此 live Bundle 與 private state 會維持不存在或完全不變。請使用平行的目錄樹，例如：

```text
workspace/
  sources/
  knowledge/
```

來源探索採遞迴方式，目前沒有 manifest 或 ignore policy。`--source` 下的每個支援檔案都屬於權威 Input Corpus，因此必須維持乾淨的來源根目錄，或把團隊未核准作為 evidence 的資料實際移到目錄外。

## 安裝

需要 Python 3.12 與 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
```

需要 reasoning 的操作必須明確指定模型名稱；API key 不會寫入 Bundle 或 state：

```bash
cp .env.example .env
```

編輯目前工作目錄的 `.env`：

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=...
# 選用：
OPENAI_BASE_URL=https://models.example/v1
```

Knowledge Forge 只讀取執行命令時所在目錄的 `.env`，不會向父目錄搜尋。CLI 參數與既有 process environment 優先於 `.env`，因此 CI、container 或 secret manager 注入的設定不會被本機檔案覆蓋。

自訂 `OPENAI_BASE_URL` 必須相容 OpenAI Responses API（`/v1/responses`）、function calling 與 structured tool output；只有 `/v1/chat/completions` 的服務無法使用。Knowledge Forge 預設允許 parallel tool calls，並在 Responses replay 中保留每個模型發出的 call 及其對應結果。

也可以直接使用 process environment：

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
# 選用：export OPENAI_BASE_URL=https://models.example/v1
```

## CLI

agent-backed `generate` 與 `update` 的 model 和 endpoint 也可以在命令列明確傳入。若 source tree 只包含可辨識的 OKF import Bundle，則不需要 model configuration 或 reasoning：

```bash
uv run knowledge-forge generate \
  --source ./pdfs \
  --out ./knowledge \
  --model <model> \
  --base-url https://models.example/v1
```

雖然 CLI 也接受 `--api-key`，仍建議使用 `.env`、`OPENAI_API_KEY` 或 production secret manager，避免 credential 留在 shell history。`.env` 已被 Git 忽略；請勿把真實 secret 寫入 `.env.example`。

建立新的 Bundle；`--out` 必須不存在或為空目錄：

```bash
uv run knowledge-forge generate \
  --source ./pdfs \
  --out ./knowledge \
  --language auto \
  --max-agent-steps 50 \
  --concept-concurrency 4
```

對 `generate` 與需要 agent 的 `update` 而言，`--max-agent-steps` 是每個 reasoning task 可使用的 model calls 上限。Concept planning 取得一份獨立 budget，每個 Concept synthesis 也各自取得全新 budget，因此複雜 Concept 不會耗用後續 Concepts 可用的 model calls。若某個 task 超過上限，錯誤會指出是 planning 或哪個 Concept，且不會發布 candidate Bundle。

Planning 完成後，`generate` 預設最多同時 synthesis 四個互不依賴的 Concept Documents。使用 `--concept-concurrency 1` 可恢復依序 synthesis，也可指定其他正整數，在 throughput 與 provider rate limit 之間取捨。Tasks 可以用不同順序完成，但 Bundle 仍按 ConceptPlan 順序組裝，且只有全部 Concepts 成功後才發布。所選 concurrency 是 Generation Identity 的一部分。

Concept concurrency 與 parallel tool calls 是不同設定。`--concept-concurrency` 控制同時執行多少個 Concept synthesis tasks；每個 task 內的 parallel tool-call mode 則控制單次 model turn 能否要求多個 `search_pages` 或 `read_pages` calls。

若 Responses-to-Bedrock gateway 無法 replay 多個 tool results，請在 `generate` 或 `update` 明確選用 non-parallel compatibility mode：

```bash
uv run knowledge-forge generate \
  --source ./pdfs \
  --out ./knowledge \
  --no-parallel-tool-calls
```

更新既有 Bundle 時使用相同 option：

```bash
uv run knowledge-forge update \
  --source ./pdfs \
  --out ./knowledge \
  --no-parallel-tool-calls
```

Compatibility mode 會送出 `parallel_tool_calls: false`。若模型或 gateway 忽略該要求而仍回傳多個 calls，LangChain middleware 會在該輪以 deterministic 方式保留一個 call 及其對應結果。Provider API error 會立即失敗；Knowledge Forge 絕不自動切換模式。所選模式會顯示在 stderr，並寫入 Generation Identity。

`generate` 與 `update` 會在 stderr 顯示目前階段；concept 數量確定後，也會顯示 synthesis 進度：

```text
[knowledge-forge] Tool-call mode: parallel.
[knowledge-forge] Reading Knowledge Sources...
[knowledge-forge] Loaded 3 Knowledge Sources with 84 evidence units.
[knowledge-forge] Indexing 84 evidence units from 3 Knowledge Sources...
[knowledge-forge] Planning concepts with the reasoning agent...
[knowledge-forge] Planned 6 concepts in Traditional Chinese.
[knowledge-forge] Synthesizing concept 1/6: refund-policy
```

每個完成的階段都會顯示處理時間，命令結束時也會顯示總時間：

```text
[knowledge-forge] Knowledge Source reading completed in 0.214s.
[knowledge-forge] Knowledge Source indexing completed in 0.038s.
[knowledge-forge] Concept planning completed in 12.481s.
[knowledge-forge] Concept synthesis 1/6 (refund-policy) completed in 8.327s.
[knowledge-forge] Concept rendering and validation completed in 0.006s.
[knowledge-forge] Candidate Bundle writing and validation completed in 0.014s.
[knowledge-forge] Atomic publication completed in 0.002s.
[knowledge-forge] Total processing time: 61.842s.
```

執行失敗時，Knowledge Forge 仍會回報已完成階段及總經過時間，並保留原有錯誤訊息與 exit code。這些訊息不包含來源全文、模型回應或 API key。Timing 與 progress 僅輸出到 stderr，不會進入 OKF Bundle、Agent Baseline、Generation Identity、reconciliation artifacts 或 private state；最終命令結果仍輸出到 stdout，方便 shell pipeline 分流。

以完整、權威的目前 Knowledge Source 集合更新。`--source` 不是「本次變更的檔案」，而是 Team Knowledge Wiki 當下應採用的全部支援來源：

```bash
uv run knowledge-forge update \
  --source ./pdfs \
  --out ../knowledge-base
```

Knowledge Sources、Bundle、state 與 Generation Identity 全部未變時，`update` 不呼叫模型也不寫入任何檔案。Tool-call mode 是 Generation Identity 的一部分，因此變更模式時必須明確授權 regeneration，而不會回傳 `No changes`。人工可以修改 Concept 正文及 `type`、`title`、`description`、`tags`、`status`；不可直接修改 `generated`、`sources`、hash、locators、`verified`、`index.md`、`log.md` 或 `.knowledge-forge/`。

`update` 只回報實際執行的 phases。Deterministic `No changes` 會回報目前 Bundle validation、Knowledge Source reading、no-change evaluation 與總時間，不會出現 model phases。Source set 未變但有人工作出修改時，會回報 Agent Baseline reuse 與 merge，不會虛構 planning 或 synthesis。需要 regeneration 時，會分別回報 temporary indexing、planning 與每個 Concept synthesis。Agent candidate merge 與 Reconciliation Conflict detection 會合併成一個 phase 計時，因為 structural three-way merge 會在執行時同時偵測 conflicts。Candidate validation、需要時的 reconciliation artifact writing，以及確實發生時的 atomic publication 也都有 timing。Reconciliation 維持 exit code 3，staleness 維持 exit code 4，其他 operational failures 維持 exit code 2，全部都會回報已完成 phases 與總經過時間。

人工新增在 `concepts/` 下且符合命名規則的文件，會在下一次 mutation 中登記為永久 Human-owned Concept。一般 `update` 不會重寫、刪除或接管這些文件；MVP 尚未提供 ownership adoption command。

當 referenced evidence、planning coverage 或 Generation Identity 過時時，`update` 會維持 live Bundle 與 managed state 不變，寫入 Regeneration Impact Report，並以 exit code 4 結束：

```text
knowledge.staleness.md
knowledge.staleness/
  manifest.json
```

報告會區分 referenced evidence 已變更或消失的 Concepts、過時的 planning coverage，以及受 Generation Identity 變更影響的 Agent-owned Concepts。Imported 與 Human-owned Concepts 不會只因 generation policy 改變而被標示為 stale。請檢閱報告，然後使用相同的 source set 與 generation options，透過以下命令授權完整 replanning 與 synthesis：

```bash
uv run knowledge-forge update \
  --source ./sources \
  --out ./knowledge \
  --regenerate-all
```

此授權會綁定 live Bundle、source set 與要求的 Generation Identity。成功發布後，pending workspace 會被刪除，Markdown report 會保留為 resolved audit record。目前不支援只針對受影響 Concepts 的 incremental regeneration。

執行 deterministic validation，不呼叫模型或修改檔案：

```bash
uv run knowledge-forge validate --out ./knowledge
```

預設命令執行 Portable OKF Validation，成功時輸出 `PASS (portable OKF 0.2)`。任何符合 v0.2 的 Bundle 都可通過，不要求 `.knowledge-forge/` private state、root `index.md`、`okf_version` marker、`concepts/` namespace，或 Knowledge Forge 的受控 Concept types。每個非 reserved Markdown 檔案都必須有可解析的 YAML frontmatter，且包含非空字串 `type`；每一層目錄中的 `index.md` 與 `log.md` 都是 reserved files。若出現 v0.2 的 optional provenance、trust、freshness、lifecycle、citation 或 attestation fields，validator 會檢查其標準結構；producer extension fields 仍可使用。Validation 全程 read-only，失敗時會以 exit code 2 回報可採取行動的錯誤。

Portable validation 會忽略 Knowledge Forge private state。若存在 `.knowledge-forge/`，仍保留 portable 結果，並在 stderr 顯示 `managed state detected — run with --managed for full validation`。若要先驗證 portable Bundle，再檢查 state integrity、tool-managed files、provenance、hash、baseline、ownership 與 filesystem consistency，請使用 managed profile：

```bash
uv run knowledge-forge validate --managed --out ./knowledge
```

如要一併驗證完整 authoritative Knowledge Source set，請提供 `--source`。Managed validation 全程 read-only，成功時輸出 `PASS (managed Knowledge Forge Bundle)`。

為目前 Concept 版本加入人工 verification：

```bash
uv run knowledge-forge verify \
  --source ./pdfs \
  --out ./knowledge \
  --concept concepts/refund-policy.md \
  --by human:reviewer-id
```

Concept 的語意內容、策展 metadata 或相關 evidence 改變後，active `verified` 會被移除；歷史事件仍保留在 audit state。

## Reconciliation

無法安全合併時，`update` 回傳 exit code 3，保留 live Bundle，並建立：

```text
knowledge.reconciliation.md
knowledge.reconciliation/
  manifest.json
  resolution.yaml
  pending/
  manual/
```

在 `resolution.yaml` 為每個 conflict 選擇：

- `keep-human`：建立綁定 human block 與 evidence hash 的 Conditional Override。
- `use-source`：採用既有 candidate block，不重新呼叫模型。
- `manual`：編輯 `manual/` 內的 Concept，並讓 `artifact` 指向該檔案。

完成後執行：

```bash
uv run knowledge-forge reconcile \
  --source ./pdfs \
  --out ./knowledge \
  --resolution ./knowledge.reconciliation/resolution.yaml
```

任何 live Bundle、Knowledge Source set、pending candidate 或 Generation Identity 已變動，都會拒絕套用過期 resolution。Imported Concept withdrawal 或 ownership collision 也可能需要 reconciliation。成功後 pending workspace 會刪除，resolved Markdown report 保留供稽核。

## Exit codes

- `0`：成功；包含 `update` 的 deterministic `No changes`。
- `2`：Knowledge Source、Bundle、state、模型輸出或操作參數驗證失敗。
- `3`：`update` 需要人工 reconciliation；live Bundle 保持不變。
- `4`：`update` 偵測到 stale evidence、planning coverage 或 Agent-owned Concepts；提供 `--regenerate-all` 前 live Bundle 保持不變。

## Bundle 與 provenance

```text
knowledge/
  index.md
  log.md
  concepts/<bundle-relative-concept-id>.md
  .knowledge-forge/
    state.json
    baseline/<concept-id>.json
```

`.knowledge-forge/state.json` 保存 source dependencies、ownership、overrides、verification audit，以及刻意分離的三種版本。`state_version` 是 private 的 **State Schema Version**，負責 deterministic state migration；頂層的 `workflow_version` 描述 orchestration behavior，因此只改變它不需要重新產生 Concept。`generation.generation_policy_version` 加上 model、endpoint、language、step budget、tool-call mode 與 Concept concurrency 共同構成 **Generation Identity**；改變任一會產生不同的 candidate，必須重新產生。

目前 schema 為 v2。checksum 驗證過的 v1 PDF state 只會透過明確的 compatibility migration 載入；不支援或未知的 schema version 會以清楚錯誤失敗，絕不被默默重新解讀。compatibility loading 是 read-only；會發布 managed state 的 mutation 則會以 atomic 方式寫入 current schema。state serialization 與 checksum 都是 deterministic。`baseline/` 使用 JSON 包裝完整 agent Markdown，避免被 OKF 誤認為公開 Concept Document。這兩者都是 tool-managed state，不應人工編輯。

每個 Concept-local Source Reference 使用 durable logical URN，並分別記錄文件與 typed locator hash。對 PDF evidence 而言，其 `id` 由 Source Identity 與頁碼位址 deterministic 地推導而得：

```yaml
sources:
  - id: policies/refunds.pdf#pdf_page:3
    resource: urn:knowledge-forge:pdf:policies%2Frefunds.pdf
    content_sha256: <sha256>
    locator: {kind: pdf_page, page: 3}
    locator_sha256: <sha256>
```

重大、爭議、數字、政策或版本敏感陳述使用 footnote，其 label 必須完全等於 Source Reference ID，例如 `[^policies/refunds.pdf#pdf_page:3]`，並在正文加入同 label 的 footnote definition。

Markdown evidence 使用 structural locator 而不是 page number，記錄完整 heading path、same-level occurrence 與 raw block hash；顯示的 line ranges 只供導覽。Imported Concept reference 會保留 upstream source references 作為 provenance chain，匯入時不會被重寫。

## 對 Concept 做關鍵字搜尋

若要在自己的程式中檢查 Bundle 的 `knowledge/concepts/*.md`，比對 id、title、body 是否符合關鍵字：

```python
from knowledge_forge.knowledge_search import search_concepts

matches = search_concepts(bundle_path, ["deadlock", "mvcc"])
```

若要把同樣的搜尋功能包成 LangChain tool，接進自己的 agent（例如讓 LLM 在起草新 Concept 前，先檢查是否已存在相關 Concept）：

```python
from knowledge_forge.tools import build_search_knowledge_tool

search_knowledge = build_search_knowledge_tool(bundle_path)
agent = create_agent(model=model, tools=[search_knowledge, ...])
```

這個 tool 只接受純關鍵字清單 `keywords: list[str]`（不接受自由文字 query），回傳符合的 concept ID 與簡短片段（JSON 格式）。

## TODO

- [x] 讓 `generate` 與 `update` 可切換 parallel tool calls。預設模式保留並 replay 所有平行 calls；`--no-parallel-tool-calls` 為受影響的 gateways 選用 deterministic single-call compatibility。Provider failure 絕不觸發自動 fallback。
- [x] 加入 `generate` 處理時間統計。Knowledge Source 讀取、索引、concept planning、各 Concept synthesis、render/validation、candidate writing/validation、publication 與總時間都會顯示，且不改變 deterministic artifacts。
- [x] 將處理時間統計延伸到 `update` 的 no-change、baseline-reuse、regeneration、reconciliation 與 publication 路徑，且只回報真正執行的 phases。

## 文件同步維護規則

`README.md` 是英文文件，`README-zh.md` 是繁體中文文件。產品行為、CLI 用法、格式、安裝、安全說明、TODO 或操作流程有任何變更時，必須在同一個 change 中同步更新兩份文件。兩份文件必須維持等價的章節順序與範例，並保留頂部的語言切換連結。

## 開發

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

架構取捨記錄在 [`docs/adr/`](docs/adr/)，領域語言記錄在 [`CONTEXT.md`](CONTEXT.md)。
