# Semantic Grep — Pi Extension Design (TypeScript + Neural Network)

> **One-liner:** `grep` by meaning, not letters. Local embedding model inside a `pi` extension finds code by intent: “where is payment retry?” → finds `stripeRetry.ts`, `refreshToken()` even with zero keyword overlap.

**Status:** Design only — no implementation. Targets `pi` coding agent ExtensionAPI (`pi-extensions.html` + `pi-tools.html`).

---

## 1. Goals & Non-Goals

### Goals
- Make pi *understand* codebase semantics, not just `bash grep`.
- 100% local, offline, private — no API key, no data exfiltration. Works in `tui` / `rpc` / `json` / `print` modes.
- Fast: index <2s for 300 files, query <50ms, survive `/fork`/`/resume`/`/reload`.
- Prove pi-extension mastery: `registerTool` + `on(session_start/tool_result)` + `details` branching + `registerEntryRenderer`.

### Non-Goals
- Not a vector DB SaaS. No Pinecone/Qdrant. In-memory cosine only for MVP (<5k blocks).
- Not training/fine-tuning. Uses frozen `all-MiniLM-L6-v2` (22MB). Learning loop is future.
- Not replacing `read` — augments it; returns top 3 file:line hits, pi then calls `read`.

---

## 2. End-User Experience

### Installation
```bash
# global (all projects)
cp -r semantic-grep ~/.pi/agent/extensions/
cd ~/.pi/agent/extensions/semantic-grep && npm install
# or project-local (requires project trust)
cp -r semantic-grep .pi/extensions/ && cd .pi/extensions/semantic-grep && npm install
pi
```

### First Run
```
pi
[semantic-grep] Loading model all-MiniLM-L6-v2 (22MB) ... cached
[semantic-grep] Indexed 342 files → 2,140 blocks (1.23s). Ready.
[semantic-grep] Tip: /semantic-search <query> or just ask "where is X?"
```

### Daily Use — Three Entry Points

1. **Natural language (automatic):** Pi's LLM sees tool description and calls it:
   > You: where do we handle auth retry?
   > pi → `semantic_search({query:"auth retry logic", topK:3})` →
   > `1. src/payments/stripeRetry.ts:42 (0.89) — exponential backoff`
   > pi: *Found it. Want me to explain?*

2. **Slash command (explicit):**
   > `/semantic-search where do we validate email?`
   > `/semantic-status` → `2140 blocks • last sync 2m ago • model: MiniLM • cache 22MB`
   > `/semantic-reindex` → force rebuild after big refactor

3. **Widget (ambient):**
   Bottom bar: `🧠 Semantic: 2140 blocks • synced 2m ago` via `ctx.ui.setWidget`. Hidden in `print/json` mode.

### Before vs After

| Before (vanilla) | After (extension) |
|---|---|
| `grep -r retry` → 47 noisy hits | Top 3 semantic hits with scores + preview |
| Pi guesses, asks clarification | Pi grounds answer in exact file:line |
| No privacy cost, but slow | Same privacy, 40ms query, fewer tokens |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Pi Core                                                         │
│  pi.on(session_start) ──┐                                       │
│  pi.on(tool_result) ────┤                                       │
│  pi.on(before_agent_start)                                      │
│  pi.registerTool("semantic_search")                             │
│  pi.registerCommand("/semantic-search|/reindex|/status")        │
│  pi.registerEntryRenderer("semantic-card") + setWidget          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ Semantic Grep Extension (Node, TypeScript via jiti)             │
│  Factory closure: { model, vectors[], fileIndex }               │
│  ├── Model Loader (onnxruntime-node + @xenova/transformers)     │
│  ├── Chunker (200 tokens/block, overlap 20)                     │
│  ├── Embedder (384-dim, cosine)                                 │
│  ├── Index (in-memory array + hash map file→blocks)            │
│  └── Renderer (Box/Text via @earendil-works/pi-tui)            │
└─────────────────────────────────────────────────────────────────┘
```

### Pi Hook Map

| Hook | Kind | Purpose |
|---|---|---|
| `session_start` | notify | Load model (async), rebuild `vectors` from branch `details` (branch-safe Fig 6) |
| `before_agent_start` | transform | Inject hint: *prefer semantic_search for where/how questions* (chained) |
| `registerTool semantic_search` | tool | LLM-callable: `{query:string, topK?:number, filterPath?:string}` |
| `tool_result` | transform | On `write`/`edit`/`bash` that changes files → patch vectors + save `details` snapshot |
| `registerCommand` | command | `/semantic-search`, `/semantic-reindex`, `/semantic-status` with `ctx.ui` dialogs |
| `session_shutdown` | notify | Close model session, clear timers (never start resources in factory) |
| `appendEntry` + `registerEntryRenderer` | persistence+UI | Durable stats + pretty card rendering (not in LLM context) |
| `ctx.ui.setWidget` | UI | Ambient status, `ctx.hasUI` guard for rpc/print |

---

## 4. Data Flow

### 4.1 Indexing (at `session_start`)

1. Discover files: respect `.gitignore`, skip `node_modules`, `.pi`, binaries. Use `getLanguageFromPath` to skip non-text.
2. Chunk: 200 tokens/block (~150 lines), overlap 20 tokens, keep `file`, `startLine`, `endLine`, `preview` (first 200 chars).
3. Embed: batch 32 blocks → `model.embed()` → `Float32Array[384]`. ~1.2s for 2k blocks on CPU.
4. Store: `vectors: {id, embedding, meta}[]` in closure. Also `fileIndex: Map<file, blockIds>`.
5. Snapshot `details`: return `details: { version:1, fileHashes, blockCount }` on next tool result so branch can rebuild without re-embedding after fork.

**Incremental:** On `tool_result` where `toolName in [write, edit, bash]` and file changed → hash check → re-chunk + re-embed only that file → patch array → update `details`.

### 4.2 Query

1. LLM calls `semantic_search({query:"payment retry logic", topK:3})`.
2. Embed query → same model, 30ms.
3. Cosine similarity vs all vectors → topK (brute force <5k, <10ms). For >10k future: HNSW.
4. Return under 50KB/2000 lines ceiling (pi-tools §6): `content: [{type:"text", text: formattedTopK}]`, `details: {query, hits}`.
5. `renderResult` not needed; `registerEntryRenderer` handles card display if user runs `/semantic-search` explicitly.

**Truncation guard:** Preview limited to 500 chars/hit, scores rounded to 2 decimals, file paths relative to `ctx.cwd` with `CONFIG_DIR_NAME` awareness.

---

## 5. State & Branching (Critical)

Three lifetimes (Fig 6):

- **Closure `vectors`** — fast, lost on reload/fork. Working copy only.
- **`details` on toolResult** — canonical, branch-safe. Every `semantic_search` and `write` result persists snapshot. On `session_start`, scan `ctx.sessionManager.getBranch()` for last `toolName==="semantic_search"` → restore.
- **`appendEntry("semantic-stats")`** — durable UI stats, not in LLM context. Rendered via `registerEntryRenderer("semantic-stats")`.

```text
session_start(reason:"fork")
  → getBranch() → find last details.snapshot → vectors = snapshot.vectors (no re-embed)
  → if none → full re-index
```

Labels: `pi.setLabel(leafId, "semantic:indexed")` optional checkpoint for `/tree` navigation.

---

## 6. Model Choice

**`Xenova/all-MiniLM-L6-v2` via `onnxruntime-node`**
- Size: 22MB quantized, 80MB full — fits pi extension `package.json` dependency without pain.
- Dim: 384 — memory 2k blocks × 384 × 4B ≈ 3MB.
- Why: Best size/quality for code synonyms. Code-specific `code-bert` larger, slower, marginal gain for MVP. `jina-embeddings-v2` is alternative if multilingual needed.
- Fallback: If model load fails (offline first run), degrade to `hasUI ? notify("Model missing, run /semantic-reindex") : text fallback` — never block session.

**Alternatives considered:**
- `Transformers.js` browser build → not needed, pi is Node.
- Cloud embeddings → violates privacy goal, adds cost.

---

## 7. UI Design

### Tool Result Card (via `registerEntryRenderer`)

```
┌─ 🔍 Semantic Grep: "payment retry logic" ──────────────────┐
│ 1. src/payments/stripeRetry.ts:42  0.89  █████████░        │
│    "exponential backoff for charge failures"                │
│ 2. src/workers/paymentQueue.ts:108 0.84  ████████░░        │
│    "re-queues failed charges with jitter"                   │
│ 3. src/utils/retryWithJitter.ts 0.81                        │
└─────────────────────────────────────────────────────────────┘
[ expanded ] shows full preview + cosine + file hash
```

Theme tokens only: `theme.fg("accent"/"muted"/"dim")`, `theme.bold`, `highlightCode`.

### Widget

- `ctx.ui.setWidget("semantic-grep", ["🧠 2140 blocks • synced 2m ago"], {placement:"belowEditor"})`
- Guard: `if (ctx.hasUI) setWidget else no-op`
- `setStatus` variant for indexing: `ctx.ui.setStatus("semantic-grep", "Indexing…")`

### Commands

- `/semantic-search <query>` — handler calls same search function, then `ctx.ui.notify` + `appendEntry`
- `/semantic-reindex` — `await ctx.ui.confirm("Re-index 342 files?")` then rebuild
- `/semantic-status` — `ctx.ui.notify(JSON.stringify(stats), "info")`

---

## 8. File Structure (Design, Not Code)

```
semantic-grep/
├── index.ts                 # default export (pi: ExtensionAPI) — registers all hooks
├── package.json             # { name:"semantic-grep", dependencies:{ "@xenova/transformers":"^2.17", "onnxruntime-node":"^1.18" } }
├── model-cache/             # .gitignored, populated at runtime
├── README.md                # user docs (install, commands, privacy)
└── DESIGN.md                # this file (symlink or copy to repo root for review)
```

**No build step.** Pi loads via `jiti`. `node_modules` resolved from extension dir. Async factory `await`ed before `session_start`.

---

## 9. Lifecycle & Modes

- **Factory:** `export default async function(pi)` — awaits model discovery but *does not* start watchers/timers. Those start in `session_start`.
- **Modes:** `ctx.mode==="tui"` → show widget/card; `ctx.hasUI===false` (json/print) → tool still works, UI no-ops.
- **Shutdown:** `pi.on("session_shutdown", () => model?.dispose())` idempotent.
- **Trust:** No `project_trust` hook needed; global extension. Project-local variant loads only after trust.

---

## 10. Performance & Limits

- Index 340 files (2k blocks): ~1.2s cold, ~50ms incremental.
- Query: ~30ms embed + ~10ms search.
- Memory: ~3MB vectors + 22MB model.
- Pi ceiling: tool output 50KB/2000 lines — enforce preview truncation.
- Cache: `ctx.signal` passed to `embed()` for abort on compaction.

---

## 11. Privacy & Safety

- Full permissions — extension can read all files pi can. Never exfiltrate embeddings.
- `tool_call` for `read` not blocked; only `semantic_search` reads via embeddings.
- Error handling: throwing in `execute` → `isError:true` to LLM, session continues (pi-tools §8).

---

## 12. Future Extensions (Out of Scope for MVP)

- HNSW for >10k blocks, or SQLite `vec` extension.
- Personal fine-tune adapter via `YouCoder` learning loop (harvest `details` corrections).
- `context` event rewrite: auto-inject top hit into LLM context before `before_provider_request`.
- Provider reranker: combine with `Provider Oracle` idea.

---

## 13. Decision Log

- **ONNX vs Transformers.js Node:** ONNX faster on CPU, smaller bundle — chosen.
- **In-memory vs DB:** In-memory simpler, branch-safe via `details` — no external DB for MVP.
- **Chunk size 200:** Balances hit precision vs recall; 500 loses line accuracy.
- **No fine-tune MVP:** Frozen model proves value before complexity.

---

## 14. Acceptance Criteria (Design Review)

- [ ] End-user can install via `~/.pi/agent/extensions/` and see indexed count on `pi` start.
- [ ] Query "auth retry" returns `stripeRetry.ts` in top 3 with score >0.8 in design walkthrough.
- [ ] `/fork` preserves index without re-embedding (via `details` rebuild).
- [ ] Widget hidden in `print` mode, tool still callable.
- [ ] No data leaves device (verified via `before_provider_request` no-op).

---

*Next: Implementation plan will map this design to tasks: model loader → chunker → registerTool → state rebuild → UI. No code in this doc.*
