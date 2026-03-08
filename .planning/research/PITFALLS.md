# Pitfalls Research

**Domain:** Vector database — Rust, HNSW + Sparse + PQ + Streaming, open-core release
**Researched:** 2026-03-05
**Confidence:** HIGH (grounded in the actual VelesDB v1.4.x codebase)

---

## Critical Pitfalls

### Pitfall 1: PQ Training With Dimension Not Divisible by `num_subspaces`

**What goes wrong:**
`ProductQuantizer::train()` panics at runtime with `assert!(dimension % num_subspaces == 0)`.
This is a hard crash with no graceful error, exposed to every caller that chooses `num_subspaces`
without knowing the embedding dimension in advance.

**Why it happens:**
The current k-means implementation (line 57 in `pq.rs`) enforces divisibility as a pre-condition.
Popular embedding dimensions are not always power-of-two multiples:
- `text-embedding-3-small` → 1536 dimensions (works with m=8, m=16, m=32, m=64)
- `nomic-embed-text` → 768 dimensions (works with m=8, m=16, m=32)
- `all-MiniLM-L6-v2` → 384 dimensions (fails with m=7, m=11, m=13)

When SDK users pass `num_subspaces` that seems "reasonable" for their model but does not divide
the actual dimension, the server panics mid-request instead of returning a clear error.

**How to avoid:**
- Return `Result<_, VelesError>` from `train()` instead of panicking. Replace `assert!` with
  `ensure!(dimension % num_subspaces == 0, VelesError::PQDimensionMismatch { dimension, num_subspaces })`.
- In the REST API handler for collection creation, validate `num_subspaces` against the declared
  dimension before calling `train()`, and return HTTP 422 with a descriptive message.
- Document the constraint prominently in rustdoc and in the OpenAPI `POST /collections` schema
  as a `multipleOf` constraint.

**Warning signs:**
- Test suite only covers small synthetic vectors (`dim=4` in `pq.rs` inline tests).
- No test exercises the panic path for non-divisible dimensions.
- `panic = "abort"` in `[profile.release]` means a panic in the server process kills the entire
  process rather than just the request thread.

**Phase to address:** Phase EPIC-063 (Product Quantization) — first user story, before any
integration with `VectorCollection`.

---

### Pitfall 2: PQ k-Means With Deterministic Init Produces Degenerate Codebooks

**What goes wrong:**
The k-means implementation in `pq.rs` (lines 179–181) uses a deterministic initializer:
`centroids[i] = samples[i % samples.len()]`. For homogeneous or near-duplicate training sets
(common during integration tests), several centroid slots will start as identical vectors and
never diverge — producing a degenerate codebook with fewer than `k` effective centroids. Recall
collapses silently: the quantizer "works" (no error), but all encoded vectors map to the same
few codes, destroying retrieval quality.

**Why it happens:**
k-means++ initialization avoids this by seeding with max-distance samples. The current naive
sequential init is fast but fragile for skewed data distributions. The problem is masked in unit
tests that use hand-crafted, well-separated clusters.

**How to avoid:**
- Replace the sequential init with k-means++ (or at minimum, random-without-replacement sampling).
- Add a codebook validation step after training: if any two centroids have L2 distance below a
  threshold (e.g., `1e-6`), log a warning and optionally re-seed the degenerate centroid.
- Add a property-based test (proptest) that trains PQ on random uniform vectors and verifies that
  all `k` centroids are distinct within a tolerance.
- Expose a `recall@10` smoke benchmark for PQ that compares results against brute-force on a
  held-out set, similar to the existing `recall_benchmark` bench.

**Warning signs:**
- Recall drops unexpectedly after PQ training (recall benchmark fails or degrades by > 5%).
- Codebook `centroids[s][i]` equals `centroids[s][j]` for `i != j` in any subspace.
- Production queries return the same small set of candidates regardless of query vector.

**Phase to address:** EPIC-063, training pipeline user story.

---

### Pitfall 3: Sparse Vector Inverted Index Lock Contention Under Write-Heavy Workloads

**What goes wrong:**
The existing `PostingList` (in `index/posting_list.rs`) is a per-term in-memory structure with no
shared synchronization layer visible at the module level. When sparse vector upserts arrive
concurrently (streaming inserts from EPIC-064), a naive approach of wrapping the entire term
dictionary in a single `RwLock<HashMap<TermId, PostingList>>` causes severe write contention:
every insert that touches any term acquires an exclusive write lock on the full map, serializing
all insertions.

**Why it happens:**
The pattern that works for dense HNSW (one write lock per collection) does not scale to sparse
vectors where each document touches potentially thousands of terms. With BERT-based sparse
encoders (SPLADE) producing 50–200 non-zero dimensions per document, a batch of 100 concurrent
streaming inserts results in 5,000–20,000 contested write operations against a single lock.

**How to avoid:**
- Use a sharded term dictionary: partition term IDs by `term_id % NUM_SHARDS` and assign a
  `parking_lot::RwLock<HashMap<TermId, PostingList>>` per shard. 64 or 128 shards is sufficient
  for most workloads. VelesDB already uses this pattern for `ConcurrentEdgeStore` (256 shards).
- For the `PostingList::promote_to_large()` path, ensure the promotion is done under a write lock
  that is released immediately after — do not hold a shard lock while running k-means or any
  other expensive initialization.
- Benchmark with `concurrency_benchmark` as a baseline and add a `sparse_concurrent_insert` bench
  to confirm throughput does not degrade under the new feature.

**Warning signs:**
- `cargo bench -- sparse` shows throughput degradation proportional to thread count (not constant).
- `parking_lot` contention visible in profiling (e.g., `perf lock` on Linux).
- Streaming insert latency p99 degrades but p50 remains acceptable — a classic write-lock symptom.

**Phase to address:** EPIC-062 (Sparse Vectors) design phase, before implementation. Revisited in
EPIC-064 (Streaming Inserts) when concurrent paths are wired together.

---

### Pitfall 4: Streaming Inserts Race Between WAL Flush and HNSW Index Rebuild

**What goes wrong:**
The auto-reindex state machine (`AutoReindexManager`) allows queries to continue during reindex
(`test_queries_continue_during_reindex` passes), but it does not explicitly define what happens
to streaming inserts that arrive while the HNSW index is in `ReindexState::Building`. Inserts
committed to the WAL during a rebuild may be missed by the new index if the rebuild takes a
snapshot of the vector store at the start of the rebuild and does not incorporate inserts that
arrive mid-build.

**Why it happens:**
HNSW index rebuilds are inherently a bulk operation over a snapshot. Streaming inserts are
inherently a continuous operation. Without an explicit "delta insert" mechanism — either a
shadow buffer that collects inserts during rebuild and is merged after, or a coordinated pause
at the flip point — the rebuilt index silently loses vectors that arrived during the build window.

**How to avoid:**
- Define a formal contract in `AutoReindexManager`: during `ReindexState::Building`, streaming
  inserts go into both (a) the WAL (for persistence) and (b) a `pending_during_rebuild: Vec<Point>`
  buffer. When the rebuild completes and the new index is swapped in, drain the buffer into the
  new index before switching reads to it.
- Add an integration test that:
  1. Triggers a manual reindex (`trigger_manual_reindex()`).
  2. Inserts 100 points while the reindex is `Building`.
  3. Completes the reindex (`complete_reindex()`).
  4. Searches for the 100 inserted points and asserts recall = 1.0.
- Model this with a loom test to detect ordering violations.

**Warning signs:**
- Integration test that inserts during reindex then searches finds fewer results than expected.
- WAL replay on restart finds vectors not in the HNSW index.
- The `auto_reindex/tests.rs` suite has no test that inserts during a `Building` state.

**Phase to address:** EPIC-064 (Streaming Inserts), specifically the HNSW integration user story.
Must be addressed before EPIC-062/063 integration since HNSW is shared across all vector types.

---

### Pitfall 5: Query Plan Cache Survives Collection Drop (Memory Leak + Stale Plans)

**What goes wrong:**
The `QueryCache` (tested in `velesql/cache_tests.rs`) stores parsed ASTs keyed by query string.
It has `clear()` but no targeted eviction by collection name. If a collection is dropped
(`Database::delete_collection`) while plans referencing it remain in the cache, two problems
arise:
1. **Memory leak:** Cached plans hold `String` collection names and AST allocations. For a cache
   of 1,000 entries at ~2 KB per AST, this is ~2 MB — small but unbounded with collection churn.
2. **Stale plan execution:** If a new collection with the same name is created with a different
   schema (e.g., different dimension), a cached plan that encodes an old dimension assumption
   (pre-computed subspace counts, column types) is re-used without re-parsing, producing wrong
   results or a panic.

**Why it happens:**
The cache tests (`cache_tests.rs`) only test the happy path (hit/miss/evict by LRU), thread
safety under concurrent reads, and invalid query rejection. There is no test for cache invalidation
triggered by collection lifecycle events (`create`, `delete`, `recreate`).

**How to avoid:**
- Add a `Database::notify_cache_invalidation(collection_name: &str)` that flushes any cached plan
  where the AST references that collection name. Call it inside `delete_collection()` and
  `create_vector_collection()` (for the recreate-with-different-schema case).
- Alternatively, include a "schema version" in the cache key: `"{query_text}#{collection_version}"`.
  Each collection carries a monotonic version counter incremented on schema change.
- Add a conformance test: create collection A, cache a plan for it, drop A, recreate A with
  different dimension, execute the same query string — assert the new plan is used (not the stale one).

**Warning signs:**
- `cache.len()` grows monotonically even after collection drops.
- Queries against a recreated collection return incorrect results or panic on dimension assertions.
- No `on_collection_drop` hook called from `Database` into `QueryCache`.

**Phase to address:** EPIC-065 (Advanced Caching) — must be addressed in the invalidation design,
not retrofitted after the cache is in production.

---

### Pitfall 6: `VectorSliceGuard` Epoch Panic Across Streaming Compaction

**What goes wrong:**
`VectorSliceGuard` (in `storage/guard.rs`) panics when the underlying mmap is remapped (epoch
mismatch) and any code holds the guard across a compaction boundary via `AsRef<[f32]>` or `Deref`.
Streaming inserts increase compaction frequency. Under EPIC-064, a long-running search query
that processes many results concurrently with high-frequency inserts is now more likely to hold
a `VectorSliceGuard` across a compaction event.

**Why it happens:**
The safe `guard.as_slice()?` API exists but is opt-in. Any code path that takes `&[f32]`
implicitly calls the panicking `Deref` implementation. The concern is documented in CONCERNS.md
but not enforced by the type system.

**How to avoid:**
- Audit every call site in `collection/vector_collection.rs`, `index/hnsw/`, and new streaming
  insert paths to confirm they use `guard.as_slice()?` not `*guard` or `guard.as_ref()`.
- Add a clippy lint (custom or via `#[deny(clippy::deref_by_slicing)]` at module level) to
  prevent new call sites from using the panicking path.
- In the streaming insert path, ensure compaction is coordinated: hold a read lock on the epoch
  counter while consuming a guard, preventing compaction from advancing the epoch mid-read.

**Warning signs:**
- Panics with "epoch mismatch" in production logs during high-throughput streaming insert tests.
- The stress test (`stress_concurrency_tests.rs`) does not inject compaction events mid-search.

**Phase to address:** EPIC-064 (Streaming Inserts) — must be analyzed in the compaction
interaction design before implementation.

---

### Pitfall 7: `bincode` 1.3 Deserialization UB Triggered by Streaming Edge Data

**What goes wrong:**
`bincode` 1.3 is used to serialize/deserialize the edge store (`EdgeStore::to_bytes`/`from_bytes`).
RUSTSEC-2025-0141 is actively suppressed. Streaming inserts for knowledge graph edges (if EPIC-064
supports graph streaming) increase the volume of `bincode`-deserialized bytes, increasing the
attack surface for the advisory.

**Why it happens:**
The advisory is acknowledged in CONCERNS.md and `deny.toml` but labeled "migration planned."
v1.5 adds streaming, which means more deserialization throughput, making this more urgent.

**How to avoid:**
- Migrate `EdgeStore` serialization from `bincode` 1.3 to `postcard` or `bincode` 2.x in v1.5,
  treating it as a required quality item before the streaming feature ships.
- If migration is deferred, add a length + checksum validation wrapper around every
  `bincode::deserialize` call to reject malformed bytes before they reach the `bincode` parser.
- Remove `|| true` from `cargo audit` in CI to make new advisories visible.

**Warning signs:**
- `cargo audit` exits non-zero but CI ignores it (current state: `|| true` soft-suppression).
- New `RUSTSEC` entries appear for `bincode` 1.x without anyone noticing.

**Phase to address:** Quality/Stabilization phase, before v1.5.0 release publication.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `assert!` instead of `Result` in `ProductQuantizer::train()` | Fast implementation | Panics kill the entire server process (`panic = "abort"`) on bad user input | Never — public API must return `Result` |
| Deterministic k-means init in PQ | Reproducible, no randomness dependency | Degenerate codebooks on skewed data; silent recall collapse | Only in unit tests with synthetic data, never in production |
| Single `RwLock` over the full term dictionary for sparse index | Simple to implement | Write serialization destroys streaming throughput | Acceptable for read-heavy workloads with <10 concurrent writers; not for streaming |
| `QueryCache::clear()` on collection drop (nuke everything) | Simple invalidation | Cache thrashing — unrelated queries lose their cached plans | Acceptable as a temporary fix during EPIC-065; not as a permanent solution |
| `|| true` on `cargo audit` in CI | Avoids CI failures for known issues | New advisories are completely invisible | Never — use `deny.toml` allowlist instead |
| `#[allow(dead_code)]` on entire `ShardedVectors` impl block | Suppresses warnings during migration | New dead code accumulates silently without any warnings | Never on a public impl block — remove after EPIC-A.2 integration confirmed |
| Statistics sampling capped at 1,000 rows | Fast `analyze()` | CBO planner uses stale cardinality estimates for collections >1K rows; bad query plans | Acceptable for v1.5 MVP if collection sizes are expected to stay small |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PyPI wheel cross-platform (maturin) | Publishing only `linux-x86_64` wheels then marking "universal2" | Build separate wheels per target (`maturin build --release -i python3`) for `linux-x86_64`, `linux-aarch64`, `macos-x86_64`, `macos-arm64`, `windows-x86_64` using GitHub Actions matrix before `twine upload` |
| npm WASM packaging (`@wiscale/velesdb-wasm`) | Bundling the WASM binary with the `persistence` feature enabled | Always build WASM with `--no-default-features` to exclude memmap2/rayon/tokio; verify `wasm32-unknown-unknown` target compiles clean before publishing |
| crates.io version bumping | Bumping `[workspace.package] version` without bumping each crate's `[[package]] version` (they use `version.workspace = true`, so this is handled, but verify) | Run `cargo publish --dry-run -p velesdb-core` for each crate and confirm the published version matches the workspace version; publish dependency crates first (core before server, cli, python, wasm, mobile) |
| LangChain / LlamaIndex SDK parity | Shipping a new feature in core (sparse, PQ) without updating the integration adapters before releasing | Block core 1.5.0 publication on integration adapter tests passing; use a feature-flag in the TypeScript SDK to guard new features |
| VelesQL conformance JSON | Adding new syntax to the `.pest` grammar without adding cases to `conformance/velesql_parser_cases.json` | Every grammar change in EPIC-062/063/064/065 must add corresponding conformance cases; CI runs `velesql_parser_conformance` across crates and the TypeScript SDK |
| OpenAPI spec drift | Implementing new REST endpoints (`/collections/{name}/sparse`, `/collections/{name}/pq-train`) without updating `openapi.yaml` | Generate spec from code with `utoipa` or enforce `openapi.yaml` round-trip in CI; do not manually maintain the spec |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| ADC (Asymmetric Distance Computation) lookup table rebuilt per query | PQ search latency linear in candidate count instead of constant | Pre-compute per-query lookup table once, reuse across all candidates in the scan | Immediately visible at >1,000 candidates; current `distance_pq_l2` rebuilds on every call if not cached by caller |
| Posting list promotion (Small→Large) under write lock | Stalls all concurrent readers during `RoaringBitmap` init from `FxHashSet` | Promote outside the shard lock: build the new `RoaringBitmap` unlocked, then swap under a brief write lock | Noticeable at >50K documents per term |
| `list_collections` acquires 4 sequential RwLocks | Response latency spikes proportional to lock contention from streaming inserts | Add a single name registry `Arc<RwLock<HashSet<String>>>` updated on every create/delete | Evident at >100 requests/second on `GET /collections` with concurrent inserts |
| PQ codebook serialization as nested `Vec<Vec<Vec<f32>>>` | Large collections take seconds to serialize the codebook on flush | Flatten codebook to `Vec<f32>` with fixed-stride access, serialize as a single contiguous block | At `num_subspaces=64, num_centroids=256, subspace_dim=24`: 64×256×24×4 bytes = 15.7 MB of nested allocations per collection |
| `QueryContext` not threaded into HNSW traversal | Memory guardrails not enforced during large ANN scans; OOM possible | Thread `QueryContext` into `execute_match` before enabling streaming inserts that can trigger concurrent large scans | Visible at >100K candidates per ANN search with no memory limit |
| Criterion perf-smoke baseline recorded on GitHub-hosted ubuntu-latest | Local Windows machine (development) results never match CI baseline; benchmark comparison always drifts | Record `baseline.json` on the same hardware as CI; use relative thresholds (15%) and never use absolute ns values for cross-machine comparison | Already a problem: baseline recorded `2026-02-20` on `github-hosted ubuntu-latest` but new benches for sparse/PQ/streaming will be run on whatever machine the developer uses |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Path traversal via sparse vector collection name | Collection named `../../../etc/passwd` creates or overwrites files outside the data directory (existing issue, documented in CONCERNS.md) | Validate collection names against allowlist `[a-zA-Z0-9_-]+` at every `Database::create_*` entry point; never use raw name to construct `Path::join` |
| `cargo audit || true` in CI masks new advisories | New `RUSTSEC` entries are completely invisible to the CI pipeline | Remove `|| true`; maintain `deny.toml` with explicit `[advisories] ignore = ["RUSTSEC-2024-0320"]` entries; let CI fail on unlisted advisories |
| REST server bound to `0.0.0.0:8080` with `CorsLayer::permissive()` | Any process on the network can read/write/delete all collections, including sparse index and PQ codebooks | Change default bind to `127.0.0.1` in production; add optional `VELESDB_API_KEY` bearer token check middleware |
| PQ codebook exposed via GET API without authentication | Trained codebooks reveal statistical properties of the vector corpus (privacy leakage) | Treat codebooks as internal state; do not expose them via the REST API without at least documenting the privacy implication |
| `bincode` 1.3 deserialization on edge store bytes | Potential UB on malformed input (RUSTSEC-2025-0141) | Migrate to `postcard` or `bincode` 2.x before v1.5.0 ships; add input length bounds check as interim mitigation |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| PQ `num_subspaces` must divide dimension — no clear error | Rust panic message with assertion backtrace; Python SDK propagates as `RuntimeError: thread panicked` | Return structured error `{"error": "PQ_DIMENSION_MISMATCH", "message": "dimension 768 must be divisible by num_subspaces 7; suggested values: 8, 16, 32, 48, 96, 192"}` with suggested valid values |
| Sparse vector search returns different result ordering than dense search on identical data | Users migrate from dense to sparse and see unexplained recall changes — no warning that sparse uses `InnerProduct` while dense defaults to `Cosine` | Document distance metric per index type prominently; warn in the OpenAPI description and CLI help when metric differs from collection default |
| VelesQL `FROM multi-alias` silently drops aliases beyond the first (BUG-8) | Multi-collection queries return wrong results without any error | Fix BUG-8 in v1.5 (change `from_alias: Option<String>` to `Vec<String>`); alternatively, return a parse error for multi-alias FROM until it is implemented |
| Criterion benchmark names not matching `baseline.json` keys | `compare_perf.py` silently skips unmatched benchmarks — new benchmarks for sparse/PQ/streaming are never compared against baseline | Enforce that every `[[bench]]` entry in `Cargo.toml` has a corresponding key in `baseline.json`; add a CI step that checks for orphaned or missing benchmark keys |
| `cargo publish` order dependency not documented | Publishing `velesdb-server` before `velesdb-core` fails with "no matching package" on crates.io | Document publish order in RELEASE.md: `core → server → cli → wasm → python → mobile → tauri-plugin`; automate with a `scripts/publish.sh` that enforces order |

---

## "Looks Done But Isn't" Checklist

- [ ] **Product Quantization:** Training implemented, but no recall validation harness. Verify a `recall@10 > 0.85` test exists on a 10K vector dataset with `num_subspaces=8, num_centroids=256`.
- [ ] **Sparse Vectors:** Inverted index built, but no `sparse_concurrent_insert` benchmark. Verify throughput does not degrade below 10K inserts/sec under 8 concurrent writers.
- [ ] **Streaming Inserts:** WAL append implemented, but no test that inserts arrive during HNSW reindex and are not lost. Verify with the `pending_during_rebuild` integration test described above.
- [ ] **Query Plan Cache:** Cache hit/miss implemented, but no collection-drop invalidation test. Verify that dropping and recreating a collection with a different schema invalidates stale plans.
- [ ] **WASM Package:** Module compiled, but verify `--no-default-features` produces a binary that loads in a browser without `instantiateStreaming` errors; test in Node.js and a real browser.
- [ ] **PyPI Wheels:** Wheel built for linux-x86_64, but verify `linux-aarch64` (AWS Graviton) and `macos-arm64` (Apple Silicon) wheels are also published and importable.
- [ ] **crates.io Publication:** `cargo publish --dry-run` passes, but verify that `README.md` renders correctly on crates.io (no broken relative links to `docs/` or `CHANGELOG`).
- [ ] **OpenAPI Spec:** New endpoints for sparse/PQ/streaming added, but verify `openapi.yaml` is regenerated and round-trips cleanly (no properties missing, no `additionalProperties` drift).
- [ ] **VelesQL Conformance:** Grammar updated for sparse syntax (e.g., `WHERE sparse_vector NEAR $v`), but verify `conformance/velesql_parser_cases.json` has cases for the new syntax that the TypeScript SDK also passes.
- [ ] **Criterion Baseline:** New benchmarks added (`sparse_insert`, `pq_encode`, `streaming_insert_p99`), but verify they are registered in `baseline.json` so `compare_perf.py` does not silently skip them.
- [ ] **bincode migration:** RUSTSEC-2025-0141 migrated to `postcard`, but verify that existing `hnsw.bin` and `edge_store` files are still loadable (add a migration test that reads a v1.4 fixture and writes it with the new serializer).

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| PQ degenerate codebook in production | HIGH | Drop the collection's PQ codebook, re-train with k-means++ init on the full dataset (which may take minutes for >1M vectors), then re-encode all vectors; add the recall validation test to prevent recurrence |
| Streaming insert lost during HNSW rebuild | HIGH | Replay the WAL from the last known-good checkpoint to identify missing vectors; re-insert them manually; add the `pending_during_rebuild` buffer to prevent recurrence |
| Query plan cache serving stale plans | MEDIUM | Call `QueryCache::clear()` to flush the entire cache (causes a brief latency spike as plans are re-parsed); then implement targeted invalidation |
| crates.io publish order failure | LOW | `cargo publish` the missing dependency crate first, then re-publish; crates.io does not require version rollback |
| PyPI wheel missing a platform | MEDIUM | Publish a new patch version (`1.5.1`) with the missing platform wheel; document which platforms were missing in the release notes |
| `cargo audit` reveals new RUSTSEC post-release | HIGH if severity CRITICAL | Yank the affected crate version from crates.io (`cargo yank --version 1.5.0 -p velesdb-core`), patch, publish `1.5.1`; notify PyPI and npm packages |
| OpenAPI spec drift causing SDK breakage | MEDIUM | Regenerate spec from code, publish a new SDK minor version (`1.5.1`) with the corrected spec; add round-trip CI check |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| PQ `assert!` panic on non-divisible dimension | EPIC-063 training design | Test: send `num_subspaces=7` to a 768-dim collection, expect HTTP 422 not 500 |
| PQ degenerate codebook from deterministic init | EPIC-063 training implementation | Property test: random uniform vectors, all `k` centroids distinct post-training |
| Sparse index lock contention | EPIC-062 architecture design | Benchmark: `sparse_concurrent_insert` with 8 threads; throughput must not drop vs. 1 thread |
| Streaming insert lost during HNSW rebuild | EPIC-064 HNSW integration | Integration test: insert during `Building` state, assert all vectors searchable post-reindex |
| Query cache stale plan after collection drop | EPIC-065 invalidation design | Conformance test: drop + recreate collection, same query string, new plan used |
| `VectorSliceGuard` epoch panic | EPIC-064 compaction audit | Loom test: compaction races with streaming read; no panic |
| `bincode` 1.3 advisory | Quality/Stabilization phase | `cargo audit` exits 0 (no suppressed critical advisories) |
| Criterion baseline drift | Phase adding new benchmarks | CI: `compare_perf.py` has no "benchmark not found in baseline" warnings |
| crates.io publish order | Release phase | `scripts/publish.sh` dry-run succeeds for all 8 crates in dependency order |
| PyPI missing platform wheels | Release phase | Matrix build produces wheels for linux-x86_64, linux-aarch64, macos-arm64, windows-x86_64 |
| OpenAPI spec drift | SDK parity phase | CI round-trip: generated spec equals committed `openapi.yaml` |
| VelesQL conformance gap for new syntax | EPIC-062/063 parser work | `velesql_parser_conformance` test passes for all new cases in all crates + TypeScript SDK |
| BUG-8 multi-alias FROM silent wrong results | Quality/Stabilization phase | Conformance case: `SELECT * FROM a, b` returns parse error or correct multi-collection results |

---

## Sources

- VelesDB codebase analysis: `crates/velesdb-core/src/quantization/pq.rs` (k-means impl, training assertions)
- VelesDB codebase analysis: `crates/velesdb-core/src/index/posting_list.rs` (inverted index, promotion threshold)
- VelesDB codebase analysis: `crates/velesdb-core/src/collection/auto_reindex/tests.rs` (reindex state machine, missing streaming-during-rebuild test)
- VelesDB codebase analysis: `crates/velesdb-core/src/velesql/cache_tests.rs` (no collection-drop invalidation test)
- VelesDB codebase analysis: `.planning/codebase/CONCERNS.md` (path traversal, bincode advisory, epoch panic, 4-lock list_collections)
- VelesDB codebase analysis: `benchmarks/baseline.json` (recorded on github-hosted ubuntu, 15% threshold)
- VelesDB codebase analysis: `Cargo.toml` (workspace version, publication metadata, `|| true` audit soft-suppress in CI)
- Domain knowledge: Faiss PQ implementation documentation (subspace divisibility constraint, ADC lookup table pattern)
- Domain knowledge: HNSW streaming insert coordination (delta buffer pattern used by Weaviate, Qdrant)
- Domain knowledge: k-means++ vs sequential init recall implications (standard ML curriculum, Qdrant blog on PQ tuning)
- Confidence: HIGH — all pitfalls are grounded in actual code found in the repository, not hypothetical

---
*Pitfalls research for: VelesDB v1.5 — Sparse Vectors + PQ + Streaming + Caching + Release*
*Researched: 2026-03-05*
