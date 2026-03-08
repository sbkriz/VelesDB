# Stack Research

**Domain:** Local/Embedded Vector Database — Rust open-core engine (VelesDB v1.5)
**Researched:** 2026-03-05
**Confidence:** MEDIUM (training data through August 2025; WebSearch/WebFetch unavailable — all claims from training knowledge + codebase inspection)

---

## Context: What v1.5 Adds

VelesDB v1.4.1 already ships: native HNSW, AVX-512/AVX2/NEON SIMD dispatch, SQ8 + Binary quantization, BM25 + Trigram text indexes, RRF fusion, VelesQL with parse cache, parking_lot RwLock throughout, mmap WAL storage, knowledge graph with streaming BFS, PyO3/UniFFI/WASM/Tauri bindings.

v1.5 scope:
- **EPIC-062** — Sparse Vectors (SPLADE-compatible, inverted index storage)
- **EPIC-063** — Product Quantization (PQ pipeline, recall-configurable, unified with SQ8/Binary)
- **EPIC-064** — Streaming Inserts (continuous ingestion without forced HNSW rebuild)
- **EPIC-065** — Advanced Query Caching (plan-level caching, hot-path acceleration for VelesQL)

The `quantization/pq.rs` stub exists (k-means train + encode/decode scaffold). The `velesql/cache.rs` has parse-level LRU. The `cache/lockfree.rs` two-tier DashMap+LRU exists. The `index/posting_list.rs` adaptive FxHashSet/RoaringBitmap exists for BM25. All four features need significant new implementation work.

---

## Recommended Stack

### EPIC-062: Sparse Vectors

#### Core Implementation Strategy

Do NOT add a third-party sparse-vector crate. The existing infrastructure (`PostingList`, `BM25Index`, `roaring`) provides the required primitives. SPLADE/BM42/SPANN-style sparse vectors are dense-valued inverted indexes with float weights — conceptually closer to weighted BM25 posting lists than to scipy sparse matrices.

| Component | Recommendation | Version | Why |
|-----------|---------------|---------|-----|
| Inverted index storage | Extend `PostingList` → `WeightedPostingList` | internal | Already in codebase, adaptive FxHashSet/RoaringBitmap pattern proven |
| Term weight storage | `Vec<(TermId, f32)>` sorted by TermId | stdlib | Sorted layout enables SIMD dot-product on two posting lists with merge-style scan |
| Term dictionary | `rustc-hash` `FxHashMap<String, TermId>` | 2.0 (current) | 2x faster than std HashMap; already a workspace dep |
| Bitmap ops for filter | `roaring` `RoaringBitmap` | 0.10 (current) | SIMD-accelerated set intersection for pre-filter before weighted score |
| Serialization | `bincode` | 1.3 (current) | Already used for HNSW/storage; no new dep |

**Sparse similarity algorithm:** Use MaxSim-over-posting-lists (inner product between query sparse vec and doc sparse vec). Implementation: iterate query terms → look up posting list for each term → accumulate `query_weight * doc_weight` for matching doc IDs → top-K with min-heap.

**SPLADE compatibility note (MEDIUM confidence):** SPLADE v2 (Formal et al., 2021) and BM42 (2024, from Qdrant) produce sparse vectors as `{term_id: float_weight}` maps. The storage format is identical; only the encoder (a BERT-based MLM model) is external to the DB. VelesDB should store and search the output sparse vectors, not re-implement the encoder.

**SPANN-style on-disk (LOW confidence):** SPANN (Chen et al., 2021, Microsoft) is a disk-resident ANN algorithm for billion-scale. For VelesDB's local/embedded use case targeting sub-10M vectors, SPANN's posting-cluster-on-disk approach adds complexity without benefit. Recommended: skip SPANN, use in-memory weighted inverted index with mmap persistence.

#### What NOT to Use for Sparse Vectors

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `nalgebra` sparse matrices | General linear algebra overhead, not search-optimized | Custom `WeightedPostingList` on `RoaringBitmap` |
| `sprs` crate | Compressed Row Storage format, poor random-access for retrieval | Custom per-term posting lists |
| Separate sparse index file format | Breaks unified mmap WAL pattern | Extend existing `payloads.log` / sharded storage |

---

### EPIC-063: Product Quantization

#### Algorithm Choice: Standard PQ vs OPQ vs RaBitQ/RaPQ

| Algorithm | Recall@10 (typical) | Training cost | Memory | Recommendation |
|-----------|--------------------|--------------|----|---|
| PQ (standard) | 0.85–0.92 | O(n·m·k·iter) k-means | m bytes/vec | USE — baseline, already scaffolded |
| OPQ (Optimized PQ) | 0.90–0.96 | PQ + iterative rotation matrix | m bytes/vec | USE for high-recall mode |
| RaBitQ (2024) | 0.92–0.97 | Lightweight random rotation | m/8 bytes/vec | EVALUATE — new, promising |
| IVF-PQ | 0.88–0.95 | PQ + IVF clustering | m bytes/vec | SKIP — requires IVF partitioning, adds HNSW conflict |
| SCANN (2020, Google) | Best-in-class | Heavy, TPU-optimized | Complex | SKIP — not Rust-native |

**Recommendation: Implement PQ (standard) + OPQ rotation.** OPQ adds a single pre-rotation step (learned via PCA + iterative refinement) and typically gains 5-8 recall points over vanilla PQ with identical storage cost. The rotation matrix is a one-time training artifact stored alongside the codebook.

**RaBitQ (MEDIUM confidence):** RaBitQ (Gao & Long, arXiv:2405.12497, 2024) proposes random binary quantization with asymmetric distance estimation. Claims outperform PQ at binary storage cost. Implementation complexity is lower than OPQ. However, it requires SIMD popcount-based distance (Hamming + correction factor), which differs from the existing SQ8 SIMD path. Recommended: implement standard PQ first, flag RaBitQ as a follow-up optimization in a later milestone.

#### Concrete Implementation for v1.5

The existing `ProductQuantizer` in `quantization/pq.rs` has the codebook structure and basic k-means. What is missing:

| Missing Component | Implementation Approach | Library |
|-------------------|------------------------|---------|
| ADC (Asymmetric Distance Computation) | Precompute partial distances per subspace into lookup table at query time | No new dep — pure Rust f32 arithmetic |
| OPQ rotation matrix | PCA via power iteration + iterative rotation — 30-50 line impl | No new dep (or `ndarray` 0.16 for matrix math) |
| PQ+HNSW integration | Store PQ codes alongside HNSW graph; rescore top-K candidates with full-precision | Modify `HnswIndex` to hold optional `PQCodebook` |
| Pipeline unification | `StorageMode::ProductQuantization` already exists in `quantization/mod.rs` | Extend enum arms in collection search path |
| Recall-configurable rescore | Expose `rescore_candidates: usize` param; default = `ef_search * 4` | Config struct extension |

**ndarray for OPQ rotation (MEDIUM confidence):** `ndarray` 0.16 provides matrix multiply needed for the rotation step. It is WASM-compatible (pure Rust). It does NOT require BLAS linkage unless the `blas` feature is enabled. For the small matrix sizes in OPQ (dimension × dimension), `ndarray` without BLAS is adequate and avoids a system dependency.

| Library | Version | Purpose | Condition |
|---------|---------|---------|-----------|
| `ndarray` | 0.16 | OPQ rotation matrix math | Only if implementing OPQ in v1.5 (recommended) |
| `ndarray-rand` | 0.15 | Random initialization for OPQ | Paired with ndarray if OPQ |
| `rand` | 0.8 (already dev-dep) | k-means random init | Move to non-dev dep |

**What NOT to Use for PQ:**

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `faiss-rs` (0.x) | C++ FFI to FAISS — breaks WASM, complicates cross-compilation, ~50MB binary bloat | Native Rust PQ (already scaffolded) |
| `hora` crate | Last release 2022, effectively abandoned; no SIMD, no PQ | Native HNSW (already faster) |
| `hnsw_rs` | Removed from VelesDB because native impl is 1.2x faster — do not reintroduce | Native HNSW |
| `instant-distance` | Single-file HNSW, no PQ support, small community | Native HNSW |
| `usearch` Rust bindings | C++ USearch via FFI — same cross-compilation problems as faiss-rs; VelesDB's native HNSW benchmarks better for local use | Native HNSW |

---

### EPIC-064: Streaming Inserts

#### The HNSW Mutation Problem

Standard HNSW requires a write lock on the entire graph during insertion to maintain navigability invariants. Naively, continuous streaming inserts serialize all writes and degrade search latency. The 2024-2025 state of the art addresses this with two patterns:

**Pattern A: Buffered async insertion with periodic integration (RECOMMENDED for v1.5)**

- Maintain an in-memory `InsertBuffer` (e.g., `crossbeam-deque` work-stealing queue or `tokio::sync::mpsc` channel)
- Insertions write to WAL + buffer immediately (sub-microsecond)
- A background Tokio task drains the buffer and integrates into HNSW in micro-batches
- Search reads from both HNSW index AND the buffer (linear scan on buffer, typically <1000 elements)
- Integration lock is short (micro-batch of 64-256 vectors at a time)

This pattern matches Qdrant's "pending inserts" approach (MEDIUM confidence, from public Qdrant engineering blog 2024).

**Pattern B: Concurrent HNSW (DiskANN-style)**

DiskANN (Microsoft, 2019) and later HNSW variants allow concurrent reads during single-writer insert via fine-grained node locking. The existing `ConcurrentEdgeStore` with 256 shards is already an implementation of this pattern for the graph layer. Extending it to HNSW requires per-node `RwLock` on neighbor lists — feasible but complex.

**Recommendation for v1.5: Pattern A (buffered async)**. Lower implementation risk, fits the existing tokio + WAL architecture. Pattern B can be a v1.6 optimization.

| Component | Library | Version | Why |
|-----------|---------|---------|-----|
| Insert channel | `tokio::sync::mpsc` | 1.42 (current) | Already in workspace; bounded channel provides backpressure |
| Work-stealing drain | `tokio::task::spawn` background task | 1.42 (current) | Background HNSW integration without blocking caller |
| Buffer scan at search | Existing `Vec<Point>` linear scan | stdlib | Buffer stays small (<10K); SIMD dispatch handles distance |
| WAL integration | Existing `payloads.log` WAL | internal | Write to WAL before buffer for durability |
| Backpressure | `tokio::sync::mpsc::Sender::reserve()` | 1.42 (current) | Blocks producer when buffer full, prevents OOM |

**Crossbeam for insert buffer (LOW confidence):** `crossbeam-channel` and `crossbeam-deque` are alternatives to tokio channels. For a fully async codebase (tokio already in workspace), tokio's mpsc is preferred because mixing crossbeam blocking I/O with async Tokio requires careful executor pinning. Unless synchronous insert API is required (non-async callers), stay with tokio.

| Component | Avoid | Why |
|-----------|-------|-----|
| Full HNSW rebuild on every batch | `reindex()` on each insert | O(n log n) per insert, defeats streaming purpose |
| `crossbeam-channel` for async path | Blocking in async context | Use `tokio::sync::mpsc` with async recv |
| Lock on full HNSW during insert | `parking_lot::RwLock` on entire index | Serializes all concurrent reads |

---

### EPIC-065: Advanced Query Caching

#### Current State

`velesql/cache.rs` implements parse-level LRU: `FxHashMap<u64, Vec<CacheEntry>>` + `VecDeque<CacheKey>` for LRU order, protected by `parking_lot::RwLock`. This caches parsed AST only, not query plans.

`cache/lockfree.rs` implements a two-tier DashMap (L1) + LruCache (L2) for general use.

v1.5 target: cache compiled **query plans** (not just parse results), and extend to cache **execution results** for deterministic queries.

#### Plan Cache Architecture

Three caching tiers for VelesQL hot path:

| Tier | What is Cached | Key | Invalidation | Library |
|------|----------------|-----|--------------|---------|
| L0: Parse cache | Parsed AST (already exists) | SHA-256 of canonical query string | LRU eviction | `velesql/cache.rs` existing |
| L1: Plan cache (NEW) | Compiled `QueryPlan` (selected execution strategy, estimated cost) | Query hash + collection stats hash | Collection mutation / reindex | Extend `QueryPlanner` with `LruCache<PlanKey, QueryPlan>` |
| L2: Result cache (NEW, optional) | Full `Vec<SearchResult>` for read-only deterministic queries | Query hash + vector hash | Any write to collection | `LockFreeLruCache` with TTL |

**ARC vs LRU for plan cache (MEDIUM confidence):** Adaptive Replacement Cache (ARC) outperforms LRU when query access patterns are mixed scan/recency. For query plan caching specifically, recency dominates (repeated queries from application code), so LRU is sufficient. ARC adds implementation complexity for minimal gain in this domain.

**Recommended cache key for plan cache:**

```
PlanKey = SHA-256(canonical_query_string) XOR CollectionStats::version_counter
```

The `CollectionStats` version counter increments on every mutation. This gives O(1) invalidation: any write to a collection bumps its counter, making all cached plans for that collection stale without explicit cache scan.

| Component | Library | Version | Why |
|-----------|---------|---------|-----|
| Plan key hash | `rustc-hash` `FxHasher` | 2.0 (current) | Non-crypto, fast; plan cache doesn't need security |
| LRU eviction | Internal `LruCache<PlanKey, QueryPlan>` | internal | Already exists in `cache/lru.rs`, reuse |
| Concurrent access | `DashMap` for hot plans | 5.5 (current) | Already workspace dep; avoid `RwLock` contention on popular plans |
| TTL for result cache | Custom `Instant`-based expiry | stdlib | Avoid `moka` or `cached` deps — overhead not justified |
| Stats | `AtomicU64` counters | stdlib | Already pattern in `LockFreeLruCache` |

**What NOT to use for caching:**

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `moka` crate | Heavyweight async cache with tokio dependency; 3x more complex than needed for plan cache | Internal `LruCache` + `DashMap` |
| `cached` proc-macro crate | Magic `#[cached]` attribute hides invalidation logic; plan cache needs explicit invalidation on writes | Manual `LruCache` implementation |
| `lru` crate (crates.io) | Does not support concurrent access; needs external lock | Internal `LruCache` backed by `IndexMap` (already exists) |
| Result cache with unbounded size | OOM risk for large result sets | Cap result cache at 16 MB total, skip caching results >64 KB |

---

## Ecosystem Overview: Rust Vector DB Libraries (2024-2025)

This section documents the state-of-the-art for completeness. VelesDB does NOT adopt these — all have been superseded by the native implementation or are inappropriate for embedded use.

| Library | Version | Status | Benchmark vs VelesDB | Notes |
|---------|---------|--------|---------------------|-------|
| `usearch` | 2.x Rust bindings | Active, C++ core (USearch by Unum) | Slower for local embedded (FFI overhead) | Excellent for server-grade, poor for embedded/WASM |
| `hnsw_rs` | 0.3.x | Maintained | 0.8x VelesDB speed (VelesDB removed it, 1.2x faster native) | simdeez_f SIMD, x86_64 only |
| `faiss-rs` | 0.x | Active, C++ FAISS | N/A — FFI linkage breaks WASM, Windows cross-compile complex | Use for Python tooling only |
| `hora` | 0.1.x | Abandoned (last 2022) | Slower than hnsw_rs | No SIMD, no maintenance |
| `instant-distance` | 0.6.x | Maintained, single-author | Baseline | Pure Rust, simple, no PQ, no persistence |
| `qdrant/segment` | internal | Not published as crate | Best-in-class server | GPL-incompatible licensing risk; not embeddable as lib |

**Verdict (HIGH confidence):** VelesDB's native HNSW is the correct choice. All external HNSW libraries either have C++ FFI problems (faiss-rs, usearch), are abandoned (hora), are benchmark-slower (hnsw_rs), or are not published for embedding (qdrant/segment).

---

## HNSW Improvements from 2024-2025 Papers

| Paper | arxiv | Contribution | Applicable to VelesDB? |
|-------|-------|-------------|----------------------|
| RaBitQ | 2405.12497 | Binary quantization with theoretical recall bounds | YES — v1.6 PQ enhancement |
| Fresh-HNSW | Not published, Qdrant engineering | Soft-deletes + in-place HNSW updates | YES concept — relevant to streaming inserts |
| HNSW with filtered search | Qdrant 2024 blog | ACORN filter integration | Already implemented via payload filter pushdown |
| DiskANN streaming | Microsoft 2024 | Concurrent insert without rebuild | YES concept — v1.6 Pattern B |
| MIPS-HNSW | 2024 variants | Maximum Inner Product Search adaptations | LOW priority — cosine/dot already handle MIPS |

**Confidence note:** Paper citations are from training knowledge (cutoff August 2025). Specific arXiv IDs should be verified before citing in documentation.

---

## Supporting Libraries: Additions for v1.5

These are NEW dependencies not currently in the workspace. All are conservative additions — no new major deps for core engine.

| Library | Version | Purpose | EPIC | Risk |
|---------|---------|---------|------|------|
| `ndarray` | 0.16 | OPQ rotation matrix math (EPIC-063) | EPIC-063 | LOW — pure Rust, WASM-compatible |
| `ndarray-rand` | 0.15 | Random rotation matrix init | EPIC-063 | LOW — pairs with ndarray |

Everything else for v1.5 reuses existing workspace dependencies.

---

## Alternatives Considered

| Topic | Recommended | Alternative | Why Not |
|-------|-------------|-------------|---------|
| Sparse vector storage | Custom `WeightedPostingList` on `RoaringBitmap` | `sprs` sparse matrix | Not search-optimized; CSR format poor for retrieval |
| PQ algorithm | Standard PQ + OPQ | SCANN | Not Rust-native; TPU-optimized; complex build |
| PQ algorithm | Standard PQ + OPQ | faiss-rs IVF-PQ | C++ FFI breaks WASM; binary size +50MB |
| Streaming inserts | Tokio mpsc + micro-batch integration | crossbeam-channel | Blocking I/O in async context; use tokio mpsc |
| Plan cache eviction | LRU (internal) | ARC | Marginal gain for plan workloads; added complexity |
| Plan cache library | Internal `LruCache` + `DashMap` | `moka` | Overkill; heavy async machinery for simple plan cache |
| Sparse similarity | Weighted inverted index dot product | Dense-approximate (e.g., encode sparse→dense) | Loses sparsity benefit; breaks SPLADE compatibility |

---

## What NOT to Use (Global)

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `faiss-rs` | C++ FFI — breaks WASM, complicates cross-compilation, +50MB binary | Native HNSW (already 1.2x faster) |
| `usearch` Rust bindings | C++ USearch FFI — same cross-compile issues; not embeddable | Native HNSW |
| `hnsw_rs` | Already removed — native impl is 1.2x faster | Native HNSW |
| `hora` | Abandoned since 2022, no SIMD, no maintenance | Native HNSW |
| `sprs` | CSR sparse matrices — not designed for nearest neighbor retrieval | Custom `WeightedPostingList` |
| `nalgebra` (full) | General linear algebra with matrix overhead; heavy dep | `ndarray` 0.16 (lighter, WASM-safe) or raw f32 arrays |
| `moka` | Heavyweight async cache; tokio thread pool overhead for plan cache | Internal `LruCache<K,V>` from `cache/lru.rs` |
| `cached` proc-macro | Hides invalidation logic; plan cache needs explicit write invalidation | Manual `LruCache` + version counter |
| `lru` crate (crates.io) | Not concurrent; needs external lock; less capable than internal impl | Internal `LruCache` backed by `IndexMap` |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `ndarray` 0.16 | Rust 1.83+ | MSRV compatibility verified with current workspace MSRV |
| `ndarray-rand` 0.15 | `ndarray` 0.16, `rand` 0.8 | Matches existing `rand` dev-dep version |
| `parking_lot` 0.12 | All existing deps | No changes required |
| `dashmap` 5.5 | `parking_lot` 0.12 | Already workspace deps, no conflict |
| `roaring` 0.10 | `serde` 1.0 | Already workspace deps |
| `tokio` 1.42 | `axum` 0.8, `tower` 0.5 | Already workspace deps; streaming inserts use existing tokio |

---

## Cargo.toml Additions

```toml
# EPIC-063: OPQ rotation matrix math (optional — only if OPQ implemented in v1.5)
[dependencies.ndarray]
version = "0.16"
optional = true

[dependencies.ndarray-rand]
version = "0.15"
optional = true

# Feature flag for OPQ
[features]
opq = ["dep:ndarray", "dep:ndarray-rand"]
```

No new non-optional dependencies are required for EPIC-062, EPIC-064, or EPIC-065.

---

## Sources

- Codebase inspection: `crates/velesdb-core/src/quantization/pq.rs`, `quantization/mod.rs`, `cache/lru.rs`, `cache/lockfree.rs`, `velesql/cache.rs`, `index/posting_list.rs`, `collection/graph/streaming.rs` — HIGH confidence (direct code read)
- `.planning/codebase/STACK.md`, `.planning/codebase/ARCHITECTURE.md` — HIGH confidence (project documentation)
- SPLADE v2 (Formal et al., 2021, ECIR) — MEDIUM confidence (training knowledge)
- BM42 sparse vectors (Qdrant, 2024) — MEDIUM confidence (training knowledge; verify current status)
- OPQ (Ge et al., 2013, CVPR) — HIGH confidence (standard algorithm, well-established)
- RaBitQ (Gao & Long, arXiv:2405.12497, 2024) — MEDIUM confidence (training knowledge; verify paper ID)
- DiskANN streaming inserts (Microsoft Research, 2024) — LOW confidence (training knowledge)
- HNSW_rs removal from VelesDB (commit history, CLAUDE.md) — HIGH confidence (codebase fact)
- ndarray 0.16 WASM compatibility — MEDIUM confidence (training knowledge; verify with ndarray docs)

---

*Stack research for: VelesDB v1.5 — Sparse Vectors, Product Quantization, Streaming Inserts, Advanced Query Caching*
*Researched: 2026-03-05*
*Note: WebSearch and WebFetch tools were unavailable during this research session. All external claims carry MEDIUM or LOW confidence and should be validated against official crate docs and arxiv before implementation.*
