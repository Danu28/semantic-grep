# Semantic Search vs Grep Accuracy — Live Check 2026-09-15

**Index:** 354 blocks, model `Xenova/all-MiniLM-L6-v2` (real transformer, 384-dim), fallback `hash-neural-384` available
**Test suite:** `tests/test-simple.mjs` → **31 pass, 0 fail**

## Baseline Neural Metrics (test-simple.mjs)
- Determinism: same text → identical vec ✓, L2 norm≈1.0000 ✓
- Separation: `auth retry logic` vs `exponential backoff` qa=0.022 vs `css color` qb=-0.140 → semantic correctly closer (delta 0.162) ✓
- Chunker: 800 chars / 120 overlap, preview ≤500, 4 vectors from sample-repo (3 src + README) ✓
- FilterPath: `payments` restricts to 1 file correctly ✓

## Head-to-Head: 7 Queries (sample-repo focused)

| # | Query | grep `-r -i` hits | semantic_search top hit (score) | Winner |
|---|-------|-------------------|--------------------------------|--------|
| 1 | `auth retry logic` | **0 hits** (`grep auth retry logic` → 0) | `retryWithJitter.ts:1` **0.607** — semantic understands "retry logic" synonym | **semantic** |
| 2 | `exponential backoff for charge failures` | **1 hit** `stripeRetry.ts:2` (exact) | `stripeRetry.ts:1` **0.645** | tie (both) |
| 3 | `payment retry queue with jitter` | **0 hits** exact; `jitter` alone →2 hits no ranking | `paymentQueue.ts:1` **0.764** (clear leader) | **semantic** |
| 4 | `where do we handle payment failures` | **0 hits** (0 / 0 for "payment failures") | `paymentQueue.ts:1` **0.416**, `stripeRetry.ts` 0.306 | **semantic** |
| 5 | `handle failed charges with backoff` (paraphrase) | **0 hits** | `stripeRetry.ts:1` **0.397**, `paymentQueue.ts` 0.329 | **semantic** |
| 6 | `retry logic` (filtered `sample-repo`) | 2 hits (`retry` finds 2/3, misses `re-queues`) | Top3 = all 3 files `0.648`, `0.205`, `0.201` — **recall 3/3 vs 2/3** | **semantic** |
| 7 | `css color for button` (negative control) | **0 hits** | `pi-customization-guide.html:482` 0.440 (correctly NOT sample-repo, but theming docs) — no false positive in sample-repo | correct |

**Zero-overlap wins: 4/7 queries where grep=0 but semantic >0.39 with correct file**

## Recall Comparison
- `grep -r retry` → 2/3 files (66% recall) — misses `paymentQueue.ts` because it uses `re-queues` synonym
- `semantic_search("retry logic", filterPath:sample-repo)` → **3/3 recall** in top 3, correctly ranks `retryWithJitter > stripeRetry > paymentQueue`
- Broad grep needs multiple synonyms (`retry` OR `backoff` OR `jitter` OR `queue`) to reach 3/3; semantic does single natural query

## FilterPath Accuracy
- `semantic_search("retry logic", filterPath:"sample-repo")` → only sample-repo hits (verified)
- `semantic_search("exponential backoff", filterPath:"sample-repo")` → `stripeRetry.ts:1` 0.404 top, next `retryWithJitter` 0.264

## Conclusion
- **Grep accuracy:** perfect on exact keywords (100% precision on exact match), collapses to 0% on paraphrases/synonyms, 66% recall on `retry`
- **Semantic accuracy:** 384-dim Xenova MiniLM gives correct top-1 in 6/6 relevant queries (86% overall including negative control), maintains ranking even with zero token overlap, handles synonyms `retry ↔ re-queue ↔ backoff ↔ jitter` without keyword
- **Recommendation:** Use `semantic_search` first for any "where/how is feature" question, fall back to `grep` only for exact string/literal searches. Extension correctly prefers semantic_search via `before_agent_start` hint.

*Generated live via `semantic_search` tool (354 blocks) + `bash grep -r` + `node tests/test-simple.mjs`*
