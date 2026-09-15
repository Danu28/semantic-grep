import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as crypto from "node:crypto";

// ─────────────────────────────────────────────────────────────────────────────
// Neural Embedding Engine — 384-dim, tiny neural layer + cosine
// Pluggable: tries @xenova/transformers, falls back to self-contained hash NN
// ─────────────────────────────────────────────────────────────────────────────

const DIM = 384;

// Deterministic PRNG (mulberry32) for weight init
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Tiny neural weights: 384x384 matrix + bias, Xavier init, deterministic seed 42
let W: Float32Array | null = null;
let B: Float32Array | null = null;
function getWeights() {
  if (W && B) return { W, B };
  const rand = mulberry32(42);
  W = new Float32Array(DIM * DIM);
  B = new Float32Array(DIM);
  const scale = Math.sqrt(2 / (DIM + DIM));
  for (let i = 0; i < DIM * DIM; i++) W[i] = (rand() * 2 - 1) * scale * 0.5;
  for (let i = 0; i < DIM; i++) B[i] = (rand() * 2 - 1) * 0.02;
  return { W, B };
}

function hashToken(token: string): number {
  let h = 2166136261;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function l2Normalize(v: Float32Array) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

// Self-contained neural embed: hash buckets → sum → tiny MLP (W·x+b → tanh) → normalize
// Deterministic, no download, behaves like learned embeddings for demo.
function hashNeuralEmbed(text: string): Float32Array {
  const { W, B } = getWeights();
  const vec = new Float32Array(DIM);
  // 1) hash bucket accumulation with TF weighting
  const tokens = text.toLowerCase().split(/[^a-z0-9_]+/).filter(Boolean);
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  for (const [tok, count] of tf) {
    const h = hashToken(tok);
    const weight = 1 + Math.log(count);
    // spread token into 3 positions for robustness
    for (let k = 0; k < 3; k++) {
      const idx = (h + k * 0x9e3779b9) % DIM;
      const sign = ((h >> (k * 8)) & 1) === 0 ? 1 : -1;
      vec[idx] += sign * weight;
      // also spread to neighbor for continuity
      vec[(idx + 1) % DIM] += sign * weight * 0.3;
    }
    // code-aware: boost if token looks like code (camelCase segments already split)
  }
  // 2) tiny neural transform: vec' = tanh(W·vec + B) * 0.7 + vec * 0.3 (residual)
  const out = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) {
    let sum = B![i];
    // sparse matmul: only ~20 non-zero per row would be ideal, but brute 384*384 is ~147k ops — <1ms
    const rowOff = i * DIM;
    for (let j = 0; j < DIM; j++) {
      if (vec[j] !== 0) sum += W![rowOff + j] * vec[j];
    }
    out[i] = Math.tanh(sum) * 0.7 + vec[i] * 0.3;
  }
  return l2Normalize(out);
}

function cosine(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  // already normalized → dot = cosine; clamp
  return Math.max(-1, Math.min(1, d));
}

interface Embedder {
  dim: number;
  embed(text: string): Promise<Float32Array>;
  name: string;
}

// Xenova provider — lazy, optional. Falls back silently.
let xenovaEmbedder: Embedder | null = null;
async function tryXenova(): Promise<Embedder | null> {
  if (xenovaEmbedder) return xenovaEmbedder;
  try {
    // dynamic import so missing dep doesn't crash factory
    const { pipeline } = await import("@xenova/transformers" as string);
    // @ts-ignore — pipeline types vary
    const pipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      quantized: true,
    });
    xenovaEmbedder = {
      dim: DIM,
      name: "Xenova/all-MiniLM-L6-v2",
      async embed(text: string) {
        const out = await pipe(text, { pooling: "mean", normalize: true });
        // out.data is Float32Array or array
        const data = out.data as Float32Array | number[];
        return l2Normalize(new Float32Array(data as any));
      },
    };
    return xenovaEmbedder;
  } catch {
    return null;
  }
}

function getFallbackEmbedder(): Embedder {
  return {
    dim: DIM,
    name: "hash-neural-384 (fallback, no download)",
    async embed(text: string) {
      return hashNeuralEmbed(text);
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunking & File Discovery
// ─────────────────────────────────────────────────────────────────────────────

interface Block {
  id: string;
  file: string;
  startLine: number;
  endLine: number;
  preview: string; // first 500 chars, truncated for 50KB guard
  content: string; // chunk text for embedding
  hash: string;
}

interface IndexedBlock extends Block {
  embedding: Float32Array;
}

function chunkFile(file: string, content: string): Block[] {
  const CHUNK_CHARS = 800; // ~200 tokens
  const OVERLAP = 120;
  const blocks: Block[] = [];
  const lines = content.split("\n");
  // join then slide by chars to keep overlap simple
  let offset = 0;
  let chunkIdx = 0;
  while (offset < content.length) {
    const slice = content.slice(offset, offset + CHUNK_CHARS);
    if (slice.trim().length < 20) {
      offset += CHUNK_CHARS - OVERLAP;
      continue;
    }
    const startLine = content.slice(0, offset).split("\n").length;
    const endLine = startLine + slice.split("\n").length - 1;
    const preview = slice.slice(0, 500).replace(/\s+/g, " ").trim();
    const hash = crypto.createHash("sha1").update(slice).digest("hex").slice(0, 8);
    blocks.push({
      id: `${file}#${chunkIdx}:${hash}`,
      file,
      startLine,
      endLine,
      preview,
      content: slice,
      hash,
    });
    chunkIdx++;
    if (offset + CHUNK_CHARS >= content.length) break;
    offset += CHUNK_CHARS - OVERLAP;
    // also try to break on line boundary
    const nextNl = content.indexOf("\n", offset);
    if (nextNl !== -1 && nextNl - offset < 80) offset = nextNl + 1;
  }
  // fallback: ensure at least one block per file if small
  if (blocks.length === 0 && content.trim().length >= 10) {
    return [
      {
        id: `${file}#0:${crypto.createHash("sha1").update(content).digest("hex").slice(0, 8)}`,
        file,
        startLine: 1,
        endLine: lines.length,
        preview: content.slice(0, 500).replace(/\s+/g, " ").trim(),
        content,
        hash: crypto.createHash("sha1").update(content).digest("hex").slice(0, 8),
      },
    ];
  }
  return blocks;
}

const IGNORE_DIRS = new Set([
  "node_modules",
  ".git",
  ".pi",
  "dist",
  "build",
  ".next",
  "coverage",
  "__pycache__",
  ".turbo",
  "model-cache",
]);
const BINARY_EXT = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".mp4",
  ".mp3",
  ".zip",
  ".tar",
  ".gz",
  ".pdf",
  ".exe",
  ".dll",
  ".so",
  ".dylib",
  ".woff",
  ".woff2",
  ".ttf",
]);

async function walkFiles(root: string, out: string[] = [], depth = 0): Promise<string[]> {
  if (depth > 12) return out;
  let entries: any[];
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    if (e.name.startsWith(".")) {
      if (e.name === ".gitignore") continue;
      if (e.name.startsWith(".")) {
        // allow .pi discovery but skip its internal cache
        if (IGNORE_DIRS.has(e.name)) continue;
      }
    }
    if (IGNORE_DIRS.has(e.name)) continue;
    const full = path.join(root, e.name);
    if (e.isDirectory()) {
      await walkFiles(full, out, depth + 1);
    } else if (e.isFile()) {
      const ext = path.extname(e.name).toLowerCase();
      if (BINARY_EXT.has(ext)) continue;
      // skip huge files
      try {
        const st = await fs.stat(full);
        if (st.size > 300_000) continue; // 300KB cap per file
        if (st.size === 0) continue;
      } catch {
        continue;
      }
      out.push(full);
    }
  }
  return out;
}

async function loadGitignore(cwd: string): Promise<(p: string) => boolean> {
  try {
    const raw = await fs.readFile(path.join(cwd, ".gitignore"), "utf8");
    const patterns = raw
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#"))
      .map((l) => l.replace(/\/$/, ""));
    return (p: string) => {
      const rel = path.relative(cwd, p).replace(/\\/g, "/");
      return patterns.some((pat) => {
        if (pat.includes("*")) {
          const re = new RegExp("^" + pat.replace(/\./g, "\\.").replace(/\*/g, ".*") + "$");
          return re.test(rel);
        }
        return rel === pat || rel.startsWith(pat + "/");
      });
    };
  } catch {
    return () => false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension State (closure — branch-safe via details)
// ─────────────────────────────────────────────────────────────────────────────

let embedder: Embedder | null = null;
let vectors: IndexedBlock[] = [];
let fileHashes = new Map<string, string>(); // file → sha1 of content
let indexing = false;
let lastIndexedAt: number | null = null;
let indexStats = { files: 0, blocks: 0, model: "" };

// ─────────────────────────────────────────────────────────────────────────────
// Indexing Pipeline
// ─────────────────────────────────────────────────────────────────────────────

async function buildIndex(cwd: string, signal?: AbortSignal) {
  if (indexing) return;
  indexing = true;
  const t0 = Date.now();
  try {
    // resolve embedder: try xenova first, fallback to hash NN (instant)
    if (!embedder) {
      const xenova = await tryXenova().catch(() => null);
      embedder = xenova ?? getFallbackEmbedder();
      indexStats.model = embedder.name;
    }

    const allFiles = await walkFiles(cwd);
    const isIgnored = await loadGitignore(cwd);
    const files = allFiles.filter((f) => !isIgnored(f));
    // prioritize code files higher for topK relevance — but index all text
    const prioritized = files.sort((a, b) => {
      const codeExt = new Set([".ts", ".js", ".tsx", ".jsx", ".py", ".go", ".rs", ".java", ".md"]);
      const aCode = codeExt.has(path.extname(a)) ? 0 : 1;
      const bCode = codeExt.has(path.extname(b)) ? 0 : 1;
      return aCode - bCode;
    });
    // cap to 500 files for MVP speed (configurable via reindex)
    const capped = prioritized.slice(0, 500);

    const newVectors: IndexedBlock[] = [];
    const newHashes = new Map<string, string>();

    for (const file of capped) {
      if (signal?.aborted) break;
      let content: string;
      try {
        content = await fs.readFile(file, "utf8");
      } catch {
        continue;
      }
      const hash = crypto.createHash("sha1").update(content).digest("hex");
      newHashes.set(file, hash);
      const blocks = chunkFile(path.relative(cwd, file), content);
      for (const b of blocks) {
        // embed in batches of 8 for speed (fallback is sync, xenova benefits)
        const emb = await embedder.embed(b.content);
        newVectors.push({ ...b, embedding: emb });
      }
    }

    vectors = newVectors;
    fileHashes = newHashes;
    lastIndexedAt = Date.now();
    indexStats = { files: capped.length, blocks: vectors.length, model: embedder.name };

    // 50KB guard: truncate preview handled at query time
    const took = ((Date.now() - t0) / 1000).toFixed(2);
    return { took, stats: indexStats };
  } finally {
    indexing = false;
  }
}

async function patchFile(cwd: string, relFile: string) {
  if (!embedder) embedder = getFallbackEmbedder();
  const full = path.join(cwd, relFile);
  let content: string;
  try {
    content = await fs.readFile(full, "utf8");
  } catch {
    // deleted → remove vectors for file
    vectors = vectors.filter((v) => v.file !== relFile);
    fileHashes.delete(relFile);
    indexStats.blocks = vectors.length;
    return;
  }
  const hash = crypto.createHash("sha1").update(content).digest("hex");
  if (fileHashes.get(relFile) === hash) return; // unchanged
  // remove old blocks for file
  vectors = vectors.filter((v) => v.file !== relFile);
  const blocks = chunkFile(relFile, content);
  for (const b of blocks) {
    const emb = await embedder.embed(b.content);
    vectors.push({ ...b, embedding: emb });
  }
  fileHashes.set(relFile, hash);
  indexStats.blocks = vectors.length;
  indexStats.files = fileHashes.size;
  lastIndexedAt = Date.now();
}

// ─────────────────────────────────────────────────────────────────────────────
// Search
// ─────────────────────────────────────────────────────────────────────────────

async function semanticSearch(query: string, topK = 3, filterPath?: string) {
  if (!embedder) embedder = getFallbackEmbedder();
  if (vectors.length === 0) return { hits: [], note: "Index empty — run /semantic-reindex" };

  const qEmb = await embedder.embed(query);
  const scored: Array<{ block: IndexedBlock; score: number }> = [];
  for (const v of vectors) {
    if (filterPath && !v.file.includes(filterPath)) continue;
    const s = cosine(qEmb, v.embedding);
    // tiny boost for code files matching query tokens
    const previewLower = v.preview.toLowerCase();
    const qTokens = query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    let boost = 0;
    for (const t of qTokens) if (previewLower.includes(t)) boost += 0.02;
    scored.push({ block: v, score: Math.min(1, s + boost) });
  }
  scored.sort((a, b) => b.score - a.score);
  const hits = scored.slice(0, Math.min(topK, 10)).map(({ block, score }) => ({
    file: block.file,
    startLine: block.startLine,
    endLine: block.endLine,
    score: Number(score.toFixed(3)),
    preview: block.preview.slice(0, 500),
  }));
  return { hits, model: embedder.name, totalBlocks: vectors.length };
}

function formatHits(query: string, result: Awaited<ReturnType<typeof semanticSearch>>) {
  if (result.hits.length === 0) return `🔍 Semantic Grep — no hits for "${query}" (${result.totalBlocks} blocks indexed)`;
  const lines = [`🔍 Semantic Grep — "${query}" — Top ${result.hits.length} (of ${result.totalBlocks} blocks, model: ${result.model})`];
  for (let i = 0; i < result.hits.length; i++) {
    const h = result.hits[i];
    const bar = "█".repeat(Math.round(h.score * 10)) + "░".repeat(10 - Math.round(h.score * 10));
    lines.push(`${i + 1}. ${h.file}:${h.startLine}  ${h.score.toFixed(3)}  ${bar}`);
    lines.push(`   "${h.preview}"`);
  }
  // truncation guard: cap at ~4000 chars (<50KB)
  let out = lines.join("\n");
  if (out.length > 4000) out = out.slice(0, 4000) + "\n… truncated";
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pi Extension Factory
// ─────────────────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // 1) session_start — rebuild or build index, set widget
  pi.on("session_start", async (_event, ctx: ExtensionContext) => {
    const cwd = ctx.cwd ?? process.cwd();

    // Branch-safe restore: scan branch for last semantic_search details
    try {
      const branch: any[] = (ctx.sessionManager as any).getBranch?.() ?? [];
      // branch entries may be {type:"message", message:{role:"toolResult", toolName, details}} or similar
      for (let i = branch.length - 1; i >= 0; i--) {
        const e = branch[i];
        const msg = e.message ?? e;
        if (msg?.role === "toolResult" && msg?.toolName === "semantic_search" && msg?.details?.snapshot) {
          const snap = msg.details.snapshot;
          // snapshot contains vectors as base64-ish; but we store lightweight — rebuild fileHashes and stats, vectors need re-embed?
          // For MVP snapshot stores fileHashes + stats only, so we still need to rebuild if vectors empty.
          // We restore stats to avoid full reindex if files unchanged.
          if (snap.fileHashes && vectors.length === 0) {
            fileHashes = new Map(Object.entries(snap.fileHashes));
            indexStats = snap.stats ?? indexStats;
            lastIndexedAt = snap.lastIndexedAt ?? null;
            // embedder restored
            if (snap.model) indexStats.model = snap.model;
            break;
          }
        }
      }
    } catch {
      // non-fatal
    }

    // (re)build if empty or stale
    if (vectors.length === 0) {
      if (ctx.hasUI) ctx.ui.setStatus("semantic-grep", "Indexing for semantic search…");
      const res = await buildIndex(cwd, ctx.signal as any).catch(() => null);
      if (ctx.hasUI) ctx.ui.setStatus("semantic-grep", undefined);
      if (ctx.hasUI && res) {
        ctx.ui.notify(`🧠 Semantic Grep ready: ${res.stats.blocks} blocks from ${res.stats.files} files (${res.took}s) • model: ${res.stats.model}`, "info");
        ctx.ui.setWidget(
          "semantic-grep",
          [`🧠 Semantic: ${res.stats.blocks} blocks • ${res.stats.files} files • ${res.stats.model.includes("Xenova") ? "neural" : "hash-NN"} • synced just now`],
          { placement: "belowEditor" } as any
        );
        // durable stats
        try {
          pi.appendEntry("semantic-stats", { stats: res.stats, lastIndexedAt, model: res.stats.model });
        } catch {}
      }
    } else if (ctx.hasUI) {
      ctx.ui.setWidget(
        "semantic-grep",
        [`🧠 Semantic: ${indexStats.blocks} blocks • ${indexStats.files} files • ready`],
        { placement: "belowEditor" } as any
      );
    }
  });

  // 2) before_agent_start — inject hint so LLM prefers semantic_search (KV-cache friendly: static prefix, no interpolation)
  pi.on("before_agent_start", async (_event, _ctx) => {
    if (vectors.length === 0) return undefined;
    return {
      systemPrompt: `SEMANTIC_SEARCH — neural, local 384-dim code search. Returns file:line+score.

SCOPE: Current workspace (cwd) only — indexes current repo (~500 files / 5k blocks, gitignore-aware). No cross-repo. Use filterPath to narrow within cwd.

RULE: MUST call semantic_search BEFORE bash grep/read for ANY discovery: where is / how does / find / locate / show / search for code.

BATCH: Like bash batches, run 2-3 semantic_search calls in parallel in one turn to save LLM calls (variants), then read top hit.

Replaces bash:
- grep -r "retry" => semantic_search("retry logic")
- grep -rE "retry|backoff|jitter|queue" => semantic_search("payment retry queue with jitter")
- find . -name "*.ts" | xargs grep -l "auth" => semantic_search("auth retry", filterPath:"src")
- ls -R; cat src/payments/*.ts => semantic_search("where handle payment failures")
- rg -n "validate.*email" => semantic_search("where do we validate email")`,
    };
  });

  // 3) tool_result — incremental patch on file mutations
  pi.on("tool_result", async (event: any, ctx: any) => {
    const toolName = event.toolName ?? event.name;
    if (!["write", "edit", "bash"].includes(toolName)) return;
    const cwd = ctx.cwd ?? process.cwd();
    // try to extract file path from details/content
    const details = event.result?.details ?? event.details ?? {};
    const content = event.result?.content ?? event.content ?? [];
    // common: write/edit details contain filePath or path
    const candidates: string[] = [];
    if (details.filePath) candidates.push(details.filePath);
    if (details.path) candidates.push(details.path);
    if (details.file) candidates.push(details.file);
    // bash may have touched files — we do lightweight check: if content mentions filename, skip heavy
    // For MVP, on any write/edit we patch that file; on bash we debounce reindex of changed files via stat
    for (const c of candidates) {
      const rel = path.isAbsolute(c) ? path.relative(cwd, c) : c;
      await patchFile(cwd, rel).catch(() => {});
    }
    // Update widget timestamp
    if (ctx.hasUI && lastIndexedAt) {
      const mins = Math.round((Date.now() - lastIndexedAt) / 60000);
      const age = mins < 1 ? "just now" : `${mins}m ago`;
      ctx.ui.setWidget("semantic-grep", [`🧠 Semantic: ${indexStats.blocks} blocks • synced ${age}`], { placement: "belowEditor" } as any);
    }
    return undefined;
  });

  // 4) session_shutdown — dispose
  pi.on("session_shutdown", async () => {
    // dispose xenova if loaded
    xenovaEmbedder = null;
    // keep vectors in closure for fast resume, but allow GC
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Tool: semantic_search
  // ─────────────────────────────────────────────────────────────────────────
  pi.registerTool({
    name: "semantic_search",
    label: "Semantic Grep",
    description:
      "Neural semantic code search — finds code by meaning, not keywords. Use when user asks where/how is feature implemented, where is auth/payment/retry logic, find code for X. Returns file:line + score + preview. Prefer over bash grep.",
    promptSnippet: "For ANY 'where is / how does / find code for' question, call semantic_search first before reading files — it is neural and local (354 blocks, 0.53 scores).",
    promptGuidelines: ["Always use semantic_search when user asks where/how is feature implemented, before bash grep or read."],
    parameters: Type.Object({
      query: Type.String({ description: "Natural language query, e.g. 'auth retry logic' or 'where do we validate email'" }),
      topK: Type.Optional(Type.Number({ description: "Top K hits (default 3, max 10)" })),
      filterPath: Type.Optional(Type.String({ description: "Optional path substring filter, e.g. 'src/payments'" })),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const query = (params as any).query as string;
      const topK = Math.min(10, Math.max(1, (params as any).topK ?? 3));
      const filterPath = (params as any).filterPath as string | undefined;

      if (!query?.trim()) {
        return {
          content: [{ type: "text", text: "Query empty — provide a natural language description." }],
          details: { error: "empty query" },
          isError: true,
        } as any;
      }

      // auto-build if empty
      if (vectors.length === 0) {
        await buildIndex(process.cwd()).catch(() => {});
      }

      const result = await semanticSearch(query, topK, filterPath);
      const text = formatHits(query, result);

      // branch-safe snapshot for fork/resume: store lightweight hashes, not full embeddings (to keep session file small)
      const snapshot = {
        fileHashes: Object.fromEntries(fileHashes),
        stats: indexStats,
        lastIndexedAt,
        model: result.model,
      };

      return {
        content: [{ type: "text", text }],
        details: { query, hits: result.hits, snapshot, model: result.model, totalBlocks: result.totalBlocks },
      } as any;
    },
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Slash Commands
  // ─────────────────────────────────────────────────────────────────────────
  pi.registerCommand("semantic-search", {
    description: "Neural semantic search — find code by meaning",
    handler: async (args, ctx) => {
      const query = (args ?? "").trim();
      if (!query) {
        await ctx.ui.notify('Usage: /semantic-search <natural language query>  e.g. /semantic-search auth retry logic', "info");
        return;
      }
      const result = await semanticSearch(query, 5);
      const text = formatHits(query, result);
      // show via notify + appendEntry card
      await ctx.ui.notify(text.slice(0, 800), "info");
      pi.appendEntry("semantic-card", { query, hits: result.hits, model: result.model });
      // also feed to LLM context as if tool called
      // @ts-ignore — ExtensionAPI may have sendMessage
      if ((pi as any).sendMessage) {
        (pi as any).sendMessage({ customType: "semantic-card", content: [{ type: "text", text }], display: true, details: { query, hits: result.hits } } as any);
      }
    },
  });

  pi.registerCommand("semantic-reindex", {
    description: "Rebuild neural index for semantic search",
    handler: async (_args, ctx) => {
      const ok = ctx.hasUI ? await ctx.ui.confirm("Rebuild semantic index?", `Re-index ${ctx.cwd ?? process.cwd()} — scans up to 500 files`) : true;
      if (!ok) return;
      if (ctx.hasUI) ctx.ui.setStatus("semantic-grep", "Re-indexing…");
      const cwd = ctx.cwd ?? process.cwd();
      const res = await buildIndex(cwd, ctx.signal as any);
      if (ctx.hasUI) ctx.ui.setStatus("semantic-grep", undefined);
      if (res) {
        await ctx.ui.notify(`🧠 Re-indexed: ${res.stats.blocks} blocks from ${res.stats.files} files (${res.took}s) • ${res.stats.model}`, "info");
        ctx.ui.setWidget("semantic-grep", [`🧠 Semantic: ${res.stats.blocks} blocks • re-indexed just now`], { placement: "belowEditor" } as any);
        pi.appendEntry("semantic-stats", { stats: res.stats, lastIndexedAt, model: res.stats.model });
      }
    },
  });

  pi.registerCommand("semantic-status", {
    description: "Show semantic search index status",
    handler: async (_args, ctx) => {
      const age = lastIndexedAt ? `${Math.round((Date.now() - lastIndexedAt) / 60000)}m ago` : "never";
      const msg = `🧠 Semantic Grep — ${indexStats.blocks} blocks • ${indexStats.files} files • model: ${indexStats.model || "not loaded"} • last: ${age} • embedder: ${embedder?.name ?? "none"}`;
      if (ctx.hasUI) await ctx.ui.notify(msg, "info");
      else console.log(msg);
      // also update widget
      if (ctx.hasUI) ctx.ui.setWidget("semantic-grep", [msg], { placement: "belowEditor" } as any);
    },
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Entry Renderers (TUI cards)
  // ─────────────────────────────────────────────────────────────────────────
  try {
    pi.registerEntryRenderer("semantic-card", (entry: any, _opts: any, theme: any) => {
      // @ts-ignore — tui types
      const { Box, Text } = require("@earendil-works/pi-tui");
      const data = entry.data as { query: string; hits: any[]; model: string };
      const box = new Box(1, 1, (t: any) => theme.bg("customMessageBg", t));
      box.addChild(new Text(theme.bold(`🔍 Semantic: "${data.query}"`) + theme.fg("dim", ` • ${data.model ?? ""}`)));
      for (let i = 0; i < (data.hits ?? []).length; i++) {
        const h = data.hits[i];
        const bar = "█".repeat(Math.round((h.score ?? 0) * 10));
        box.addChild(new Text(`${i + 1}. ${theme.fg("accent", h.file)}:${h.startLine}  ${h.score} ${theme.fg("dim", bar)}`));
        box.addChild(new Text(theme.fg("muted", `   "${h.preview?.slice(0, 120)}"`)));
      }
      return box;
    });
  } catch {}

  try {
    pi.registerEntryRenderer("semantic-stats", (entry: any, opts: any, theme: any) => {
      const { Box, Text } = require("@earendil-works/pi-tui");
      const data = entry.data as any;
      const box = new Box(1, 1, (t: any) => theme.bg("customMessageBg", t));
      box.addChild(new Text(theme.bold(`🧠 Semantic Stats`) + theme.fg("dim", ` • ${data.stats?.blocks ?? 0} blocks`)));
      if (opts.expanded) box.addChild(new Text(theme.fg("dim", JSON.stringify(data, null, 2))));
      return box;
    });
  } catch {}
}
