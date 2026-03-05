# Project Research Summary

**Project:** VelesDB v1.5 Open-Core Release
**Domain:** Local-first Rust vector database — EPIC-062 through EPIC-065
**Researched:** 2026-03-05
**Confidence:** HIGH (primary sources: direct codebase analysis; external claims MEDIUM)

## Executive Summary

VelesDB v1.5 adds four interdependent subsystems to a production-quality v1.4 engine: sparse vector storage and search (EPIC-062), product quantization (EPIC-063), streaming inserts (EPIC-064), and advanced query plan caching (EPIC-065). All four EPICs fill genuine gaps relative to Qdrant v1.9+ and Milvus 2.4+ — sparse vectors are the most impactful because 2025 production RAG workloads depend on hybrid dense+sparse fusion, which VelesDB cannot do today. The recommended approach is to build on existing primitives throughout: `PostingList`/`RoaringBitmap` for sparse storage, the existing PQ scaffold and SIMD dispatch for compression, `tokio::sync::mpsc` and the existing `upsert()` path for streaming, and the existing two-tier `LruCache`/`DashMap` infrastructure for plan caching. No new mandatory dependencies are needed beyond two optional ones (`ndarray`, `ndarray-rand`) for OPQ rotation math.

The recommended implementation sequence follows architectural dependency order: EPIC-063 first (zero API surface change, pure performance), EPIC-062 second (new index module + new Collection field, enables hybrid search), EPIC-065 third (plan cache + `write_generation` field needed before streaming), EPIC-064 last (background task depends on stable collection-level fields). Hybrid dense+sparse search is a v1.5 stretch goal enabled automatically once EPIC-062 ships, since RRF fusion already exists in v1.4. The build-order rationale also applies to quality work: the `bincode` 1.3 RUSTSEC-2025-0141 advisory must be resolved before v1.5.0 ships, since EPIC-064 streaming inserts increase deserialization volume and worsen the advisory's attack surface.

The highest-risk pitfalls are concrete and codebase-grounded: the `ProductQuantizer::train()` `assert!` panic on non-divisible dimension configs, the k-means deterministic init that produces degenerate codebooks on skewed data, and the WAL/HNSW rebuild race that can silently lose streaming inserts during an auto-reindex. All three are preventable through targeted design decisions in the earliest implementation user stories of each EPIC. The remaining pitfalls (sparse index lock contention, stale plan cache after collection drop, `VectorSliceGuard` epoch panic under compaction, benchmark baseline drift) are each well-understood with clear prevention strategies documented in PITFALLS.md.

## Key Findings

### Recommended Stack

VelesDB v1.5 requires no new mandatory crate dependencies. Every feature leverages existing workspace primitives. The sparse vector index extends `PostingList`/`RoaringBitmap` with a sharded term dictionary (following the `ConcurrentEdgeStore` 256-shard pattern). PQ acceleration uses the existing SIMD dispatch infrastructure to add an ADC lookup-table kernel alongside the existing `cosine_dispatched`/`dot_product_dispatched` functions. Streaming inserts use the already-present `tokio::sync::mpsc` bounded channel from the `persistence` feature. Plan caching extends the existing `LruCache`/`DashMap` two-tier infrastructure in `cache/`.

**Core technologies:**
- `parking_lot::RwLock` with sharded term dictionary — sparse index concurrency, following `ConcurrentEdgeStore` pattern already validated in the graph layer
- `tokio::sync::mpsc` bounded channel — streaming insert back-pressure, reuses existing async runtime
- `FxHashMap<u32, PostingList>` sharded 64-way — sparse term dictionary, avoids single-lock bottleneck
- `ndarray` 0.16 (optional, `opq` feature flag) — OPQ rotation matrix math; pure Rust, WASM-safe
- Existing `LruCache` + `DashMap` — compiled plan cache with `AtomicU64` write-generation invalidation
- `postcard` or `bincode` 2.x — migration target for RUSTSEC-2025-0141 (replaces `bincode` 1.3 on `EdgeStore`)

All external HNSW libraries (`faiss-rs`, `usearch`, `hnsw_rs`, `hora`) are confirmed inappropriate: C++ FFI breaks WASM cross-compilation, `hnsw_rs` is 0.8x VelesDB speed, `hora` is abandoned. VelesDB's native HNSW is the correct choice with HIGH confidence.

### Expected Features

**Must have (table stakes):**
- Sparse vector storage and search (EPIC-062) — production RAG in 2025 requires hybrid search; VelesDB is the only major local vector DB without it
- Product quantization with rescore (EPIC-063) — fills the 8-32x compression range between SQ8 (4x) and Binary (32x); mandatory for >1M vector workloads
- Streaming inserts with searchable-immediately guarantee (EPIC-064) — AI agent memory patterns require continuous ingestion without forced client batching
- Query plan cache with LRU + TTL (EPIC-065) — any SQL-adjacent system is expected to cache plans; parse overhead at 1000 QPS is 50-200ms/s wasted
- Hybrid dense+sparse fusion — flows automatically from EPIC-062 + existing RRF; no extra phase needed
- Per-collection PQ config (`m`, `nbits`, `rescore`) — follows existing per-collection SQ8/Binary pattern
- `bincode` 1.3 migration (RUSTSEC-2025-0141) — security posture for open-core release

**Should have (competitive differentiators):**
- VelesQL `NEAR_SPARSE` syntax — no competitor offers SQL-like sparse search; strong DX advantage
- EXPLAIN cache hit/miss output — unique among local vector DBs; pedagogically valuable for debug
- `FLUSH PLAN CACHE` VelesQL command — operational necessity for schema-change workflows
- `SHOW PLAN CACHE STATS` — hit rate observability surfaces cache value to users
- Streaming insert SSE progress feed — low cost given existing SSE infrastructure; high visibility for Tauri/desktop users

**Defer to v2+:**
- Sparse vector GPU acceleration — cuSPARSE-equivalent on wgpu has no viable path; CPU is acceptable given sparse sparsity
- Online PQ centroid retraining — unpredictable latency spikes; expose explicit `RETRAIN` command instead
- Cross-collection sparse+dense hybrid in one VelesQL query — requires multi-source query planning; defer to v2.0 query engine
- Distributed sparse index sharding — Premium tier only via `DatabaseObserver`
- Auto-tokenization / built-in BM25 tokenizer — NLP problem, not DB problem; accept `{term_id: weight}` externally
- DiskANN / disk-backed ANN, IVF partitioning, FP16/BF16 storage — deferred per ANN_SOTA_AUDIT

### Architecture Approach

All four EPICs attach to the existing layered architecture at specific, low-risk extension points: new fields on `Collection` (`sparse_index: Arc<RwLock<SparseIndex>>`, `insert_tx`, `write_generation: AtomicU64`), a new module under `index/sparse/`, SIMD kernels alongside existing ones in `simd_native/`, a background `StreamIngester` task isolated in `collection/core/stream_ingest.rs`, and a separate `CompiledPlanCache` struct in `velesql/` with different invalidation semantics from the existing AST cache. The `Database` facade and all adapter crates (server, CLI, Python, WASM, Mobile) remain structurally unchanged; v1.5 features are additive extensions, not refactors.

**Major components (new or significantly modified):**
1. `index/sparse/` (new) — `SparseIndex` with sharded `PostingList`, inner-product scorer, append-only `sparse_postings.bin` persistence
2. `quantization/pq_simd.rs` (new) — ADC lookup-table AVX2/NEON kernels; scalar fallback in existing `pq.rs`
3. `collection/core/stream_ingest.rs` (new) — `StreamIngester` background tokio task; bounded mpsc channel; micro-batch HNSW integration
4. `velesql/compiled_plan_cache.rs` (new) — `CompiledPlanCache` with per-collection `write_generation` invalidation; separate from immutable AST cache
5. `collection/types.rs` (modified) — three new fields + lock-ordering extension to position 9-10
6. SDK propagation — `velesdb-server` new endpoints, Python `sparse_search()`/`stream_insert()`, TypeScript `sparseSearch()`/`streamInsert()`

Build order from ARCHITECTURE.md: EPIC-063 → EPIC-062 → EPIC-065 → EPIC-064.

### Critical Pitfalls

1. **PQ `assert!` panic on non-divisible dimension** — Replace `assert!(dimension % num_subspaces == 0)` with `Result`-returning validation in `ProductQuantizer::train()` before any integration work; return HTTP 422 with suggested valid values. `panic = "abort"` in release profile means this crashes the entire server process.

2. **Degenerate PQ codebook from deterministic k-means init** — Replace sequential centroid init (`centroids[i] = samples[i % samples.len()]`) with k-means++ random-without-replacement sampling; add post-training codebook validation (no two centroids closer than 1e-6); add recall@10 > 0.85 property test on 10K vectors.

3. **Streaming inserts lost during HNSW auto-reindex** — The `AutoReindexManager` does not define behavior for inserts arriving during `ReindexState::Building`. Implement `pending_during_rebuild: Vec<Point>` buffer; drain into new index before flip. Add integration test: insert 100 points during `Building`, assert 100 found post-reindex.

4. **Sparse index write contention under concurrent streaming** — A single `RwLock<HashMap<TermId, PostingList>>` serializes all inserts (50-200 term writes per SPLADE document × N concurrent writers). Use 64-shard partitioning by `term_id % 64`, following the existing `ConcurrentEdgeStore` pattern.

5. **Stale query plan cache after collection drop/recreate** — `QueryCache::clear()` exists but no targeted invalidation by collection name. Add `notify_cache_invalidation(collection_name)` called from `delete_collection()` and `create_vector_collection()`; keyed by collection name + `write_generation` to handle recreate-with-different-schema.

6. **`bincode` 1.3 RUSTSEC-2025-0141 on `EdgeStore`** — `cargo audit || true` in CI currently makes this invisible. Migrate `EdgeStore` serialization to `postcard` before v1.5.0 ships; add v1.4 fixture load test to ensure backward compatibility.

## Implications for Roadmap

Based on combined research, the following phase structure is recommended. All four EPICs are v1.5 scope; phases map to EPICs with quality work bookending them.

### Phase 1: PQ SIMD Acceleration (EPIC-063)
**Rationale:** Zero API surface change — purely algorithmic improvement to the existing PQ scaffold. Lowest risk entry point. Must fix the `assert!` panic and degenerate codebook before any user-facing work; doing so first makes every subsequent phase safer.
**Delivers:** Production-quality PQ training (k-means++ init, codebook validation), ADC lookup-table SIMD kernels (AVX2/NEON), recall@10 property test harness, `QuantizationConfig::ProductQuantization` per-collection config complete.
**Addresses:** Compression gap between SQ8 (4x) and Binary (32x); 8-32x configurable compression for >1M vector workloads.
**Avoids:** PQ `assert!` panic (Pitfall 1), degenerate codebook (Pitfall 2), ADC lookup table rebuilt per query (Performance Trap).

### Phase 2: Sparse Vectors + Hybrid Search (EPIC-062)
**Rationale:** The highest user-impact gap. Depends only on existing infrastructure — no EPIC-063/065/064 dependency. Hybrid search is a stretch goal within this phase because RRF is already implemented; once sparse search works, `fusion::rrf_merge` handles the rest with no extra phase.
**Delivers:** `index/sparse/` module (sharded `PostingList`, inner-product scorer, mmap-backed `sparse_postings.bin`), `Collection::sparse_search()`, hybrid dense+sparse fusion via existing RRF, VelesQL `NEAR_SPARSE` syntax (grammar + conformance cases), Python `sparse_search()`/`search_hybrid()`, TypeScript `sparseSearch()`.
**Addresses:** Table-stakes gap vs Qdrant/Milvus; enables SPLADE/BM42 hybrid RAG pipelines.
**Avoids:** Single-lock sparse index contention (Pitfall 3 — use 64-shard partitioning from day one); CSR matrix anti-pattern (use posting lists, not CSR).

### Phase 3: Advanced Query Caching (EPIC-065)
**Rationale:** Must land before streaming inserts because `write_generation: AtomicU64` is introduced here, and EPIC-064's streaming path must increment it on every upsert. Building the plan cache before EPIC-064 ensures the invalidation contract is established when the high-write streaming path arrives.
**Delivers:** `CompiledPlanCache` with LRU + `write_generation` invalidation, EXPLAIN `cache_hit`/`plan_reuse_count` output, `FLUSH PLAN CACHE` and `SHOW PLAN CACHE STATS` VelesQL commands, collection-drop invalidation hook, 50%+ latency reduction on repeated identical queries (benchmark target).
**Addresses:** Repeated-query latency for LangChain/LlamaIndex integration patterns; operational cache management.
**Avoids:** Stale plan after collection drop/recreate (Pitfall 5 — build invalidation correctly from the start, not retrofitted); AST cache invalidation anti-pattern (keep two separate caches, never wipe AST cache on writes).

### Phase 4: Streaming Inserts (EPIC-064)
**Rationale:** Most complex phase — depends on stable `Collection` fields (EPIC-062's `sparse_index`, EPIC-065's `write_generation`). The HNSW rebuild race pitfall is critical and must be designed before implementation. `VectorSliceGuard` epoch audit also belongs here.
**Delivers:** `StreamIngester` background tokio task, bounded mpsc channel with HTTP 429 back-pressure, `pending_during_rebuild` buffer for HNSW reindex coordination, `VectorSliceGuard` epoch audit across all streaming paths, Python `stream_insert()` async context manager, TypeScript `streamInsert()`, optional SSE progress feed.
**Addresses:** Continuous AI agent ingestion without forced client batching; causal searchability guarantee (insert returns → vector findable in next query).
**Avoids:** WAL/HNSW rebuild race losing inserts (Pitfall 4 — `pending_during_rebuild` buffer required), `VectorSliceGuard` epoch panic (Pitfall 6 — audit all call sites before streaming goes live), holding multiple write locks during batch (micro-batch upsert loop pattern).

### Phase 5: Quality, Security, and Release Readiness
**Rationale:** Specific quality items that cut across all four EPICs and block the open-core release. RUSTSEC-2025-0141 must be resolved before publication. SDK parity, OpenAPI spec correctness, VelesQL conformance completeness, and platform wheel coverage are all release blockers.
**Delivers:** `bincode` 1.3 → `postcard` migration with v1.4 fixture backward-compat test, `cargo audit` CI fix (remove `|| true`, use `deny.toml` allowlist), BUG-8 multi-alias FROM fix, Criterion baseline registration for new sparse/PQ/streaming benchmarks, PyPI matrix wheels (linux-x86_64, linux-aarch64, macos-arm64, windows-x86_64), WASM `--no-default-features` browser smoke test, OpenAPI round-trip CI check, `scripts/publish.sh` with enforced dependency order.
**Avoids:** RUSTSEC-2025-0141 UB under streaming load (Pitfall 7), benchmark baseline drift making perf-smoke comparisons meaningless, crates.io publish-order failures, PyPI missing-platform patches.

### Phase Ordering Rationale

- **EPIC-063 before EPIC-062:** PQ has zero API surface, so bugs in training are fixable in isolation before sparse adds a new index type and new Collection fields.
- **EPIC-062 before EPIC-065:** Sparse index introduces `sparse_index` as a new Collection field. Plan cache introduces `write_generation`. Both must be present and stable before EPIC-064 wires the streaming path through all of them.
- **EPIC-065 before EPIC-064:** The plan cache's `write_generation` counter is incremented by every upsert — including streaming upserts from EPIC-064. If EPIC-064 ships first, the counter is missing and the cache cannot function correctly.
- **Quality phase last:** Addresses cross-cutting concerns that only make sense after all feature code is written (benchmark keys, conformance cases, OpenAPI endpoints, wheel targets all depend on final feature scope).

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Sparse Vectors):** WAND traversal algorithm for approximate sparse search is a niche topic; if throughput benchmarks show the naive posting-list scan is insufficient at >500K documents, research WAND/MaxScore algorithm implementation. The inverted index compaction strategy for >1M documents needs design.
- **Phase 4 (Streaming Inserts):** The `VectorSliceGuard` epoch/compaction interaction needs a focused design review session before implementation. The loom test for epoch races should be written before the streaming path goes live, not after.
- **Phase 5 (Release):** `bincode` → `postcard` migration needs a wire-format compatibility analysis for existing `hnsw.bin` and `edge_store` on-disk files. This is a data migration problem, not just a code change.

Phases with standard patterns (skip research):
- **Phase 1 (PQ SIMD):** ADC lookup-table vectorization is a well-documented algorithm (FAISS precedent, existing SIMD dispatch pattern in codebase). The OPQ rotation matrix math is standard PCA + iterative refinement. No novel research needed.
- **Phase 3 (Plan Cache):** Two-level LRU cache with write-generation invalidation follows PostgreSQL prepared-statement cache patterns exactly. The existing `LruCache`/`DashMap` infrastructure in `cache/` is already the right building block.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All core recommendations are based on direct codebase inspection. Two new optional deps (`ndarray`, `ndarray-rand`) are pure Rust with confirmed WASM compatibility per training knowledge — verify against official ndarray docs before adding. External library comparisons (usearch, faiss-rs, hora) are HIGH confidence based on codebase history and CLAUDE.md. |
| Features | MEDIUM | Table-stakes features are HIGH confidence (codebase gaps are factual). Competitor API patterns (Qdrant v1.9+, Milvus 2.4+) are MEDIUM — WebSearch unavailable, sourced from training data through August 2025; verify current API shapes before implementing SDK compatibility. |
| Architecture | HIGH | Primary source is direct codebase analysis. All extension points identified (lock ordering, `DistanceEngine`, `types.rs` field positions) are verified against actual source files. The four data-flow diagrams are derived from existing code paths, not invented. |
| Pitfalls | HIGH | Every critical pitfall is grounded in actual code found in the repository: the `assert!` in `pq.rs:57`, the deterministic init on lines 179-181, the missing streaming-during-reindex test in `auto_reindex/tests.rs`, the `|| true` in CI, the missing collection-drop invalidation in `cache_tests.rs`. These are facts, not hypotheses. |

**Overall confidence:** HIGH

### Gaps to Address

- **RaBitQ (arXiv:2405.12497):** Promising binary quantization algorithm for v1.6 PQ enhancement. arXiv ID should be verified before citing in documentation. Mark as a post-v1.5 research item in the roadmap.
- **SPLADE/BM42 API patterns:** Competitor sparse vector insert/search API shapes (Qdrant, Milvus) sourced from training data. Verify current API shapes against official docs when designing VelesDB's Python/TS SDK surface to ensure migration path for existing users.
- **ndarray 0.16 WASM compatibility:** Stated as MEDIUM confidence — verify that `ndarray` 0.16 compiles cleanly under `wasm32-unknown-unknown` with `--no-default-features` before merging the `opq` feature flag.
- **PQ recall@10 targets:** The 95-98% recall@10 with rescore targets for standard embedding dimensions (768D, 1536D) are from training data. Validate against the actual VelesDB HNSW + rescore pipeline using the existing `recall_benchmark` harness on real embedding data during Phase 1.
- **Criterion baseline cross-machine drift:** The current `baseline.json` was recorded on `github-hosted ubuntu-latest`. New benchmarks for sparse/PQ/streaming will be recorded on a different machine. Use relative thresholds (15%) and never absolute ns values; ensure CI records the baseline on the same hardware as the perf-smoke job.
- **BUG-8 (multi-alias FROM silent wrong results):** Documented as a known bug. Phase 5 should fix it before open-core release since it produces wrong results without error — a correctness issue that would damage trust for new users discovering VelesDB through the v1.5 release.

## Sources

### Primary (HIGH confidence)
- VelesDB codebase (`crates/velesdb-core/src/`) — direct analysis, 2026-03-05
- `.planning/codebase/ARCHITECTURE.md`, `STRUCTURE.md`, `STACK.md` — project documentation
- `.planning/codebase/CONCERNS.md` — known gaps, security issues, tech debt
- `quantization/pq.rs` — PQ training implementation, k-means init, assert! locations
- `index/posting_list.rs` — inverted index, promotion threshold, adaptive structure
- `collection/auto_reindex/tests.rs` — reindex state machine, missing streaming test confirmed
- `velesql/cache_tests.rs` — plan cache tests, missing collection-drop invalidation confirmed
- `collection/types.rs` — canonical lock ordering (lines 147-160)
- `simd_native/dispatch/mod.rs` — `DistanceEngine` extension pattern
- `benchmarks/baseline.json` — hardware context, 15% regression threshold

### Secondary (MEDIUM confidence)
- Qdrant v1.9+ documentation (training data) — sparse vector API, PQ config, oversampling/rescore pattern
- Milvus 2.4+ documentation (training data) — IVF_PQ params, sparse inverted index
- SPLADE v2 (Formal et al., 2021, ECIR) — output format, average non-zeros, vocabulary size
- BM42 sparse vectors (Qdrant, 2024) — model-agnostic sparse weight approach
- OPQ (Ge et al., 2013, CVPR) — rotation matrix pre-processing, recall improvement over standard PQ
- Qdrant engineering blog 2024 — "pending inserts" buffered async insertion pattern

### Tertiary (LOW confidence)
- RaBitQ (Gao & Long, arXiv:2405.12497, 2024) — binary quantization with theoretical recall bounds; arXiv ID unverified
- DiskANN streaming inserts (Microsoft Research, 2024) — concurrent insert without rebuild; Pattern B for v1.6
- ndarray 0.16 WASM compatibility — training knowledge; requires verification

---
*Research completed: 2026-03-05*
*Ready for roadmap: yes*
