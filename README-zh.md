# Knowledge Forge

[English](README.md) | 繁體中文

Knowledge Forge 將團隊核准的 PDF 文件整理成符合 Google Open Knowledge Format 0.2 的 Wiki。它使用 LangGraph 編排固定 workflow，並讓單一 LangChain reasoning agent 負責跨文件的 concept planning 與 synthesis。

輸出的 Bundle 同時由 agent 與人維護：人工修改不會在下一次更新時被靜默覆蓋。非重疊變更會以 deterministic three-way merge 合併；同一結構區塊的衝突則維持 live Bundle 不變，產生可稽核的 reconciliation workspace。

## MVP 範圍

- 輸入：指定目錄內遞迴發現、具有完整文字層的 PDF。
- 輸出：OKF 0.2 Markdown Bundle；不包含 Web UI、聊天、RAG API 或原始 PDF。
- 模型：透過 Responses API 呼叫支援 tool calling 的 OpenAI-compatible endpoint；不使用 Chat Completions。
- Retrieval：每次執行建立暫存 SQLite FTS5 page index，結束後刪除。
- Concept types：`Concept`、`Definition`、`Policy`、`Procedure`、`FAQ`。

純掃描 PDF、加密 PDF、損壞文件、整份沒有可擷取文字、symlink 及正規化後路徑碰撞，都會讓整次操作原子失敗。具有文字內容的 PDF 可以包含刻意留白的頁面。

PDF 內容被視為不可信資料，agent 沒有 shell、network 或任意檔案寫入工具。擷取出的 PDF 文字會送往設定的模型 endpoint；Knowledge Forge 不會把 PDF、擷取全文或 API key 寫入 Bundle。模型請求固定使用 Responses API 並設定 `store: false`。為避免內容被額外傳送，偵測到 LangSmith／LangChain tracing 開啟時會拒絕執行。

## 安裝

需要 Python 3.12 與 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
```

模型名稱必須明確指定，API key 不會寫入 Bundle 或 state：

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

自訂 `OPENAI_BASE_URL` 必須相容 OpenAI Responses API（`/v1/responses`）、function calling 與 structured tool output；只有 `/v1/chat/completions` 的服務無法使用。為相容 Responses-to-Bedrock gateways 的 tool-result 排序限制，Knowledge Forge 目前會要求停用 parallel tool calls；若 gateway 或下游模型忽略該旗標，LangChain middleware 也會在本機把每輪 calls 正規化成單一 call。

也可以直接使用 process environment：

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
# 選用：export OPENAI_BASE_URL=https://models.example/v1
```

## CLI

`generate` 與 `update` 的 model 和 endpoint 也可以在命令列明確傳入：

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
  --max-agent-steps 50
```

`generate` 與 `update` 會在 stderr 顯示目前階段；concept 數量確定後，也會顯示 synthesis 進度：

```text
[knowledge-forge] Reading PDF sources...
[knowledge-forge] Loaded 3 PDFs with 84 pages.
[knowledge-forge] Indexing 84 pages from 3 PDFs...
[knowledge-forge] Planning concepts with the reasoning agent...
[knowledge-forge] Planned 6 concepts in Traditional Chinese.
[knowledge-forge] Synthesizing concept 1/6: refund-policy
```

這些訊息不包含 PDF 全文、模型回應或 API key；最終結果仍輸出到 stdout，方便 shell pipeline 分流。

以完整、權威的目前 PDF 集合更新。`--source` 不是「本次變更的檔案」，而是 Team Knowledge Wiki 當下應採用的全部 PDF：

```bash
uv run knowledge-forge update \
  --source ./pdfs \
  --out ../knowledge-base
```

PDF Sources、Bundle、state 與 Generation Identity 全部未變時，`update` 不呼叫模型也不寫入任何檔案。人工可以修改 Concept 正文及 `type`、`title`、`description`、`tags`、`status`；不可直接修改 `generated`、`sources`、hash、page mapping、`verified`、`index.md`、`log.md` 或 `.knowledge-forge/`。

人工新增在 `concepts/` 下且符合命名規則的文件，會在下一次 mutation 中登記為永久 Human-owned Concept。一般 `update` 不會重寫、刪除或接管這些文件；MVP 尚未提供 ownership adoption command。

執行 deterministic validation，不呼叫模型或修改檔案：

```bash
uv run knowledge-forge validate --out ./knowledge --source ./pdfs
```

省略 `--source` 時，只驗證 Bundle、provenance 格式、citations、Agent Baseline 與 private state 的內部一致性；加上 `--source` 後，還會核對完整 PDF 集合、實際 SHA-256 與頁碼邊界。

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

任何 live Bundle、PDF Source set、pending candidate 或 Generation Identity 已變動，都會拒絕套用過期 resolution。成功後 pending workspace 會刪除，resolved Markdown report 保留供稽核。

## Exit codes

- `0`：成功；包含 `update` 的 deterministic `No changes`。
- `2`：PDF Source、Bundle、state、模型輸出或操作參數驗證失敗。
- `3`：`update` 需要人工 reconciliation；live Bundle 保持不變。

## Bundle 與 provenance

```text
knowledge/
  index.md
  log.md
  concepts/<stable-kebab-case-slug>.md
  .knowledge-forge/
    state.json
    baseline/<concept-id>.json
```

`.knowledge-forge/state.json` 保存 workflow 與 Generation Identity、source dependencies、ownership、overrides 與 verification audit，並以 deterministic checksum 偵測意外修改。`baseline/` 使用 JSON 包裝完整 agent Markdown，避免被 OKF 誤認為公開 Concept Document。這兩者都是 tool-managed state，不應人工編輯。

每個 source entry 使用 durable logical URN，並公開記錄內容 hash 與頁碼：

```yaml
sources:
  - id: policies/refunds.pdf
    resource: urn:knowledge-forge:pdf:policies%2Frefunds.pdf
    content_sha256: <sha256>
    pages: [2-4, "7"]
```

重大、爭議、數字、政策或版本敏感陳述使用 page-level footnote，例如 `[^policies/refunds.pdf@p3]`，並在正文加入同 label 的 footnote definition。

## TODO

- [ ] 支援可切換的 parallel tool calls。產品預設應使用平行呼叫，以降低 planning 與 synthesis 的等待時間；目前 Responses-to-Bedrock 測試 gateway 無法正確接收多個 `toolResult`，因此需加入 CLI 參數以選擇 non-parallel compatibility mode。預設平行模式在發布前必須通過 gateway replay regression test。
- [ ] 加入處理時間統計。記錄 PDF 讀取、索引、concept planning、各 Concept synthesis、render/validation、publication 與整體執行時間；時間資訊僅供觀測，不寫入 deterministic Bundle 或 generation state。

## 文件同步維護規則

`README.md` 是英文文件，`README-zh.md` 是繁體中文文件。產品行為、CLI 用法、格式、安裝、安全說明、TODO 或操作流程有任何變更時，必須在同一個 change 中同步更新兩份文件。兩份文件必須維持等價的章節順序與範例，並保留頂部的語言切換連結。

## 開發

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

架構取捨記錄在 [`docs/adr/`](docs/adr/)，領域語言記錄在 [`CONTEXT.md`](CONTEXT.md)。
