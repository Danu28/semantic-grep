# Semantic Grep — Pi Extension

**Neural semantic search for `pi` coding agent.** Finds code by *meaning*, not keywords.

> `semantic_search("auth retry logic")` → `src/payments/stripeRetry.ts:42 (0.89)` even with zero keyword overlap.

- 100% local • offline • private — no API key, **0 vulnerabilities** (`npm audit` clean)
- **Pure JS** `hash-neural-384` — no download, `<1ms` embed, `384-dim` (`tanh(W·x+b)`), deterministic
- TypeScript via `jiti` (no build), branch-safe via `details` snapshot, incremental patch on `write`/`edit`

## Install

```bash
# via pi (recommended) — installs from GitHub
pi install git:github.com/Danu28/semantic-grep
# then
npm --prefix $(pi extensions path)/semantic-grep install  # if needed
pi

# manual / local dev
npm install
pi -e ./index.ts --mode print -p "hello"  # smoke test

# alternative manual copy
# cp -r . ~/.pi/agent/extensions/semantic-grep && npm --prefix ~/.pi/agent/extensions/semantic-grep install
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

## Neural Engine — Pure `hash-neural-384` (0 vulns)

**Shipped:** self-contained **hash-neural 384** — hash buckets + tiny MLP (`tanh(W·x+b)` + residual + `L2` norm) — deterministic, 0-download, <1ms embed, `384-dim` cosine. No native deps, `npm audit` **0**.

**Opt-in (advanced):** `npm install @xenova/transformers` (`all-MiniLM-L6-v2`, 22MB) + `onnxruntime-node` — `index.ts` auto-detects via `tryXenova()` and uses transformer if present; otherwise `hash-neural`. Opt-in brings `5 vulns` transitive (`protobufjs`/`sharp`) — see Security.

## Structure

```
semantic-grep/  (repo root)
├── index.ts       # ExtensionAPI factory — registers tool, commands, renderers
├── package.json   # pi.extensions: ["./index.ts"] deps: @xenova/transformers
├── DESIGN.md      # full design doc
├── tests/         # sample repo + test harness
└── model-cache/   # .gitignored, populated at runtime
```

## Limits

- Capped to 500 files / ~5k blocks for MVP brute-force search (<10ms). For >10k, HNSW planned.
- Tool output <50KB / 2000 lines (pi ceiling).
- Skips binaries, node_modules, .git, model-cache.

## Security — `hash-neural` = 0 vulns, `pi install` clean

- **`semantic-grep` pure:** `npm audit` → **`0 vulnerabilities`** — no `protobufjs`/`sharp`/`onnxruntime` because no native deps. Verified `found 0` with `--omit=optional` and full `audit`.
- **Xenova opt-in:** If you `npm install @xenova/transformers`, you get `5 vulns (1 critical protobufjs, 4 high sharp)` via `onnx-proto → onnxruntime-web` — local-only, needs crafted `.proto`/image, not code chunks. `npm audit fix --force` downgrades to `1.4.2` (breaking) — not recommended.
- **`pi-brain` dev:** `vitest` moderate + `esbuild` low — dev-only, `esbuild` fix via `npm audit fix` (no breaking, `0.28.x`).

> **Shipped pure `hash-neural`** — safe by default. Xenova is manual opt-in only.

## Design

See `./DESIGN.md` for full design doc.
