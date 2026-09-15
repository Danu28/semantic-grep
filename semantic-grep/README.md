# Semantic Grep — Pi Extension

**Neural semantic search for `pi` coding agent.** Finds code by *meaning*, not keywords.

> `semantic_search("auth retry logic")` → `src/payments/stripeRetry.ts:42 (0.89)` even with zero keyword overlap.

- 100% local • offline • private — no API key
- TypeScript via `jiti` (no build), 384-dim embeddings (Xenova/all-MiniLM-L6-v2 or hash-neural fallback)
- Branch-safe via `details` snapshot, incremental patch on `write`/`edit`

## Install

```bash
# global (all projects)
cp -r semantic-grep ~/.pi/agent/extensions/
cd ~/.pi/agent/extensions/semantic-grep && npm install

# project-local (requires trust)
cp -r semantic-grep .pi/extensions/
cd .pi/extensions/semantic-grep && npm install

pi
```

First run:
```
[semantic-grep] Indexed 342 files → 2140 blocks (1.2s) • model: hash-neural-384
```

## Use

**Automatic (LLM):** Just ask pi *“where is payment retry?”* — pi calls `semantic_search` tool.

**Slash commands:**
- `/semantic-search <query>` — e.g. `/semantic-search where do we validate email?`
- `/semantic-reindex` — rebuild index (up to 500 files, ~1s)
- `/semantic-status` — blocks, files, model, last sync

**Widget:** `🧠 Semantic: 2140 blocks • synced just now` (hidden in `print` mode).

## How It Works

- `session_start` → walk files (gitignore-aware, 300KB cap) → chunk 800 chars (~200 tokens) → embed 384-dim → cosine search
- `before_agent_start` → injects hint to prefer `semantic_search` over `bash grep`
- `tool_result` on `write`/`edit` → patches vectors for changed file only
- `details.snapshot` → survives `/fork`/`/resume`/`/reload` without re-embedding

## Neural Engine

Primary: `@xenova/transformers` (`all-MiniLM-L6-v2`, 22MB quantized).  
Fallback: self-contained **hash-neural 384** — hash buckets + tiny MLP (`tanh(W·x+b)`) + residual + L2 norm — deterministic, 0-download, <1ms embed. Same 384-dim cosine interface.

## Structure

```
semantic-grep/
├── index.ts       # ExtensionAPI factory — registers tool, commands, renderers
├── package.json   # deps: @xenova/transformers
└── model-cache/   # .gitignored
```

## Limits

- Capped to 500 files / ~5k blocks for MVP brute-force search (<10ms). For >10k, HNSW planned.
- Tool output <50KB / 2000 lines (pi ceiling).
- Skips binaries, node_modules, .git, model-cache.

## Design

See `../DESIGN.md` for full design doc.
