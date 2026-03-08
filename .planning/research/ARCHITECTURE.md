# Architecture Research

**Domain:** Rust local-first vector database — VelesDB v1.5 new features
**Researched:** 2026-03-05
**Confidence:** HIGH (primary source: direct codebase analysis; no web tools available)

---

## Standard Architecture

### System Overview

The v1.5 features insert into the existing layered architecture at specific, well-defined extension points. The diagram below shows the full stack with the four new feature areas highlighted.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ADAPTER LAYER  (unchanged for v1.5)                                         │
│  velesdb-server (Axum) | velesdb-cli | velesdb-python | velesdb-wasm         │
│  velesdb-mobile | tauri-plugin-velesdb                                       │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │ Database::execute_query / upsert / search
┌────────────────────────────▼─────────────────────────────────────────────────┐
│  DATABASE FACADE  (database.rs)                                               │
│  Four typed registries (RwLock<HashMap>) — no domain logic                   │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │ delegates to Collection
┌──────▼──────────────────────────────────────────────────────────────────────┐
│  COLLECTION LAYER  (collection/)                                              │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  Collection (god-object, types.rs)  — all Arc<RwLock<_>> fields       │   │
│  │                                                                        │   │
│  │  EXISTING                  │  NEW v1.5                                 │   │
│  │  ─────────────────         │  ──────────────────────────               │   │
│  │  pq_cache                  │  sparse_index (SparseIndex)               │   │
│  │  pq_quantizer              │  insert_tx (Sender<StreamBatch>)          │   │
│  │  query_cache (LRU AST)     │  compiled_plan_cache (new layer)          │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  search/     core/crud.rs   quantization/   velesql/cache.rs                 │
│  ─────────── ────────────   ─────────────   ─────────────────                │
│  SparseSearch StreamIngester PQSIMDKernels   CompiledPlanCache               │
│  HybridFusion (EPIC-064)    (EPIC-063)       (EPIC-065)                      │
│  (EPIC-062)                                                                   │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────────────┐
│  INDEX LAYER  (index/)                                                        │
│                                                                               │
│  HnswIndex    BM25Index     NEW: SparseIndex                                  │
│  (unchanged)  (unchanged)   (inverted postings lists)                         │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────────────┐
│  SIMD / COMPUTE LAYER  (simd_native/ + simd_dispatch.rs)                     │
│                                                                               │
│  EXISTING                 NEW v1.5                                            │
│  ────────────────         ────────────────────────────────                   │
│  AVX-512 cosine           pq_distance_avx2() — 8-way SIMD over               │
│  AVX2 dot/L2              lookup-table rows (sub-dim 4-8 floats)              │
│  NEON cosine              sparse_dot_avx2() — gather + SIMD dot              │
│  Scalar fallback          DistanceEngine::pq() extension point                │
└──────┬──────────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────────────┐
│  STORAGE LAYER  (storage/)                                                    │
│                                                                               │
│  MmapStorage  LogPayloadStorage  NEW: sparse_postings.bin                    │
│  WAL          compaction          (append-only, per-term sorted lists)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | v1.5 Change |
|-----------|----------------|-------------|
| `Database` facade | Collection lifecycle, query routing | None — stable boundary |
| `Collection` god-object | Per-collection state, all CRUD/search | Add `sparse_index`, `insert_tx` Sender, `compiled_plan_cache` fields |
| `HnswIndex` | Dense ANN search | None — incremental insert already works |
| `SparseIndex` (new) | Inverted posting lists for sparse vectors | New module under `index/sparse/` |
| `simd_native/` | Distance kernels | Add PQ ADC SIMD kernel + sparse dot kernel |
| `velesql/cache.rs` | LRU AST parse cache (exists) | Extend with compiled plan cache (separate struct) |
| `collection/core/crud.rs` | Synchronous upsert | Add async `stream_insert()` entry point |
| `quantization/pq.rs` | PQ train/encode/ADC | Add SIMD ADC + lookup table precomputation |
| `storage/` | mmap + WAL | Add `sparse_postings.bin` file per collection |

---

## Recommended Project Structure

Extensions for v1.5 integrate cleanly into existing module layout:

```
crates/velesdb-core/src/
│
├── index/
│   ├── hnsw/                    # Unchanged
│   ├── bm25.rs                  # Unchanged
│   ├── sparse/                  # NEW (EPIC-062)
│   │   ├── mod.rs               # SparseIndex public API
│   │   ├── postings.rs          # InvertedList: Vec<(doc_id, value)>
│   │   ├── scorer.rs            # Sparse dot product (IP metric only)
│   │   ├── storage.rs           # append-only postings.bin persistence
│   │   └── sparse_tests.rs
│   └── mod.rs                   # Re-export SparseIndex
│
├── quantization/
│   ├── pq.rs                    # EXISTS: train/encode/ADC scalar
│   ├── pq_simd.rs               # NEW (EPIC-063): AVX2/NEON ADC kernel
│   │                            # lookup_table_avx2(), distance_pq_avx2()
│   └── mod.rs                   # Re-export pq_simd
│
├── collection/
│   ├── types.rs                 # ADD fields: sparse_index, insert_tx
│   ├── core/
│   │   ├── crud.rs              # ADD stream_insert() + batch flush
│   │   └── stream_ingest.rs     # NEW (EPIC-064): StreamIngester actor
│   │       # tokio::sync::mpsc channel, background task, back-pressure
│   └── search/
│       ├── sparse.rs            # NEW (EPIC-062): sparse_search()
│       └── hybrid_sparse.rs     # NEW: RRF fusion dense+sparse
│
├── velesql/
│   ├── cache.rs                 # EXISTS: LRU AST parse cache
│   └── compiled_plan_cache.rs   # NEW (EPIC-065): compiled plan cache
│       # CompiledPlan = resolved collection refs + chosen strategy
│       # Invalidation: per-collection write counter (AtomicU64)
│
└── simd_native/
    ├── dispatch/
    │   └── pq.rs                # NEW: resolve_pq_adc(simd_level, subspace_dim)
    └── x86_avx2/
        └── pq.rs                # NEW: pq_distance_avx2() unsafe kernel
```

### Structure Rationale

- **`index/sparse/`** follows the existing `index/hnsw/` pattern — self-contained module with storage, algorithm, and tests co-located.
- **`quantization/pq_simd.rs`** alongside `pq.rs` keeps the SIMD acceleration co-located with the data model it accelerates, mirrors the existing scalar.rs / scalar_simd pattern.
- **`collection/core/stream_ingest.rs`** isolates the async ingestion actor from synchronous CRUD — the `crud.rs` file stays synchronous, streaming is an additive surface.
- **`velesql/compiled_plan_cache.rs`** is a separate struct from `QueryCache` — different invalidation semantics (AST cache is immutable; plan cache is write-invalidated).
- **`simd_native/dispatch/pq.rs`** mirrors the existing `cosine.rs / dot.rs / euclidean.rs` dispatch file pattern exactly.

---

## Architectural Patterns

### Pattern 1: Inverted Index for Sparse Vectors (EPIC-062)

**What:** A `SparseIndex` based on postings lists (term → sorted list of (doc_id, value) pairs). Inner-product distance only (the natural metric for sparse vectors from BM25, SPLADE, etc.). Search = iterate non-zero query dimensions, look up postings, accumulate dot products, heap-topk.

**When to use:** Whenever the input vector has >90% zero dimensions (typical for SPLADE/sparse encoders). For hybrid dense+sparse, run both indexes and fuse via the existing RRF `fusion/` module.

**Trade-offs:**
- Storage format: sorted postings lists (ascending doc_id) enable merge-join for multi-term queries and O(1) append. CSR (Compressed Sparse Row) is more compact for read-heavy workloads but harder to update incrementally — use postings lists for v1.5.
- Memory: postings can be mmap-backed exactly like `sharded_vectors.rs` using the existing `MmapStorage` pattern.
- Not suitable for dense vectors — the two indexes live independently in `Collection`.

**Example (Rust sketch):**
```rust
// index/sparse/postings.rs
pub struct PostingList {
    /// Sorted (doc_id, value) pairs. Append-only; compaction offline.
    entries: Vec<(u64, f32)>,
}

pub struct SparseIndex {
    /// term_id (u32) → PostingList
    postings: parking_lot::RwLock<HashMap<u32, PostingList>>,
    // Persisted to sparse_postings.bin
}
```

### Pattern 2: PQ SIMD Acceleration via Lookup-Table Vectorization (EPIC-063)

**What:** Asymmetric Distance Computation (ADC) builds a lookup table LT[subspace][centroid] = distance(query_sub, centroid) once per query. Then for each candidate PQ code, the distance = sum over subspaces of LT[s][code[s]]. The table-lookup inner loop over candidates is embarrassingly SIMD-parallel.

**When to use:** Any collection with `StorageMode::ProductQuantization`. The existing scalar ADC in `pq.rs::distance_pq_l2()` is the fallback — SIMD version lives alongside it in `pq_simd.rs`.

**Trade-offs:**
- The lookup table itself (m subspaces × k centroids × f32) fits in L1 cache for m=8, k=256: 8×256×4 = 8KB — perfect.
- AVX2 processes 8 subspace distances in parallel with `_mm256_loadu_ps` gather+sum.
- For NEON: 4-wide `vld1q_f32` / `vaddq_f32` achieves 4x width — same pattern, different intrinsics.
- Binary size impact: two additional files (~200 lines each) — negligible for 15MB budget.

**Extension into `DistanceEngine`:**
Add a `pq_adc()` method to `DistanceEngine` in `simd_native/dispatch/mod.rs` that accepts `(query_lookup_table: &[f32], pq_codes: &[u16]) -> f32`. The dispatch resolution at `DistanceEngine::new()` picks AVX2 or scalar path once.

### Pattern 3: Channel-Based Streaming Insert Pipeline (EPIC-064)

**What:** A background `StreamIngester` task runs on a tokio runtime (gated under `persistence` feature). The public API receives a `tokio::sync::mpsc::Sender<StreamBatch>`. Callers send batches; the ingester applies them to `Collection::upsert()` and drives HNSW incremental insert. Back-pressure is natural: bounded channel blocks the sender when the ingester is behind.

**When to use:** Real-time ingestion scenarios where the caller cannot wait for each upsert to complete — RAG document streaming, event ingestion, LLM tool-call buffering.

**Trade-offs:**
- `crossbeam::channel` is an alternative but `tokio::sync::mpsc` is already in the dependency tree (via `persistence` feature) and integrates better with the existing async server.
- Channel capacity = configurable (default 1024 batches). Too small → producer blocks; too large → memory pressure.
- WAL durability: each batch flush to HNSW also triggers a WAL sync — same guarantee as synchronous upsert.
- WASM incompatible (background task requires OS threads) — correctly excluded by `persistence` feature gate.

**Component boundary:**
```
Caller (server handler or SDK)
  → Sender<StreamBatch>
  ↓ (bounded channel, async)
StreamIngester (background tokio::task)
  → Collection::upsert(batch)
  → HNSW incremental insert
  → WAL flush
  → ack via oneshot if requested
```

### Pattern 4: Two-Level Query Plan Cache (EPIC-065)

**What:** The existing `velesql/cache.rs` caches the *parsed AST* (immutable, collection-name-independent). The new `CompiledPlanCache` caches the *resolved execution plan* (collection pointers + chosen strategy + index selections). Two separate LRU caches with different invalidation rules.

**When to use:** The compiled plan cache is only valid while the collection's schema is unchanged. Each collection tracks a monotonic `write_generation: AtomicU64` that increments on every upsert. The cache stores the generation at plan creation time; a cache hit is invalidated if generation has advanced.

**Trade-offs:**
- AST cache (existing): never invalidated (SQL text → AST is pure parsing, collection-independent). Safe to keep forever.
- Compiled plan cache: invalidated per-collection-write. For write-heavy workloads, effective hit rate drops — the cache provides value mainly for read-heavy / mixed workloads.
- Two-lock hierarchy: read AST cache first, then compiled plan cache. Never hold both write locks simultaneously (deadlock risk). Existing lock-ordering doc in `types.rs` must be extended.
- Cache size: small (default 256 compiled plans). Compiled plans are larger than ASTs; keep the cache tight.

**Example invalidation check:**
```rust
// velesql/compiled_plan_cache.rs
struct CachedPlan {
    plan: ExecutionPlan,
    collection_generation: u64,
}

impl CompiledPlanCache {
    pub fn get(&self, key: &str, current_gen: u64) -> Option<ExecutionPlan> {
        let entry = self.inner.read().get(key)?;
        if entry.collection_generation == current_gen {
            Some(entry.plan.clone())
        } else {
            None  // Generation advanced — stale plan, re-plan
        }
    }
}
```

---

## Data Flow

### Sparse Search Flow (EPIC-062)

```
Client: POST /collections/{name}/search  { "sparse_vector": {...} }
    |
    v
Handler (velesdb-server)
    |
    v
Collection::sparse_search(query_sparse: SparseVector, top_k: usize)
    |  -- read lock on sparse_index --
    v
SparseIndex::search()
    |  iterate non-zero query dims
    |  look up PostingList per term
    |  accumulate dot products into BinaryHeap<(score, doc_id)>
    v
Vec<(doc_id, score)>  [topk candidates]
    |
    v  [optional: hybrid path]
fusion::rrf_merge(sparse_results, dense_results)  [existing RRF module]
    |
    v
Vec<SearchResult>  →  JSON response
```

### Streaming Insert Flow (EPIC-064)

```
Client (continuous producer)
    |
    v
POST /collections/{name}/stream_insert  OR  Collection::stream_sender()
    |
    v
Sender<StreamBatch>  [bounded mpsc, default cap=1024]
    |  [async, back-pressure automatic]
    v
StreamIngester::run()  [tokio background task, Arc<Collection>]
    |  recv batch
    v
Collection::upsert(batch.points)  [synchronous, existing path]
    |  -- HNSW insert, WAL flush, quantization cache update --
    v
optional: oneshot reply to caller (flush confirmation)
```

### Compiled Plan Cache Flow (EPIC-065)

```
Database::execute_query(sql, params)
    |
    v
QueryCache::parse(sql)               [L1: AST cache — never invalidated]
    |  cache miss → Parser::parse() → store AST
    |  cache hit → return cached AST
    v
CompiledPlanCache::get(key, gen)      [L2: compiled plan cache]
    |  cache miss / stale gen:
    |    QueryPlanner::plan(ast, stats) → ExecutionPlan
    |    store plan with current_gen
    |  cache hit:
    |    return ExecutionPlan directly
    v
Collection::execute_plan(plan)        [unchanged execution path]
```

### PQ SIMD Distance Flow (EPIC-063)

```
Collection::search() [ProductQuantization mode]
    |
    v
Precompute LT[m][k]: lookup_table_from_query(query, codebook)
    |  one pass over query vector, m*k multiplications
    v
For each candidate in HNSW beam:
    |
    v
simd_native::pq::distance_pq_avx2(codes: &[u16], lookup_table: &[f32]) -> f32
    |  AVX2: load 8 consecutive LT entries, accumulate with _mm256_add_ps
    |  NEON: load 4 consecutive LT entries, accumulate with vaddq_f32
    |  scalar fallback: existing pq.rs::distance_pq_l2
    v
BinaryHeap update → final topk reranking with exact SIMD distance
```

### State Management

All new state follows the existing `Arc<RwLock<_>>` pattern established in `types.rs`:

| New Field | Type | Lock position in canonical ordering |
|-----------|------|--------------------------------------|
| `sparse_index` | `Arc<RwLock<SparseIndex>>` | Position 9 (after `edge_store` at 8) |
| `insert_tx` | `Arc<Mutex<Option<Sender<StreamBatch>>>>` | Position 10 |
| `write_generation` | `Arc<AtomicU64>` | Lock-free — no position needed |

The `CompiledPlanCache` lives on `Collection` (same as `query_cache`) and is accessed through the same LRU read/write pattern. The lock-ordering document in `docs/CONCURRENCY_MODEL.md` must be updated.

---

## Scaling Considerations

VelesDB is local-first — horizontal scaling is explicitly out of scope (Premium only). The relevant "scaling" is single-node performance as collection size grows.

| Scale (vectors) | Sparse Index Concern | PQ Concern | Streaming Concern |
|-----------------|---------------------|------------|-------------------|
| < 100K | In-memory postings fine | PQ training needs ~10K samples min | Channel cap 1024 batches is sufficient |
| 100K–1M | Postings exceed L3 — mmap-back postings.bin | PQ codebook fits in L3 (8K) | Monitor ingester lag via metrics |
| > 1M | Postings compaction needed (merge sort offline) | PQ is the right choice over Full precision | Consider per-collection ingester thread affinity |

### Scaling Priorities

1. **First bottleneck:** `SparseIndex` postings list growth → implement `compaction()` (offline merge, same pattern as `storage/compaction.rs`).
2. **Second bottleneck:** `StreamIngester` channel saturation under burst inserts → expose `stream_stats()` with lag metric, let caller tune channel capacity.

---

## Anti-Patterns

### Anti-Pattern 1: CSR Matrix for Sparse Vectors

**What people do:** Implement sparse storage as a compressed sparse row (CSR) matrix — efficient for static read-only workloads (like SciPy sparse matrices).

**Why it's wrong:** CSR requires rewriting the entire row on every insert. VelesDB needs incremental inserts (EPIC-064). CSR also does not support per-term posting list merging during search without full materialization.

**Do this instead:** Use sorted posting lists per term (inverted index). Append-only for writes. Compact offline (like WAL compaction) when read amplification grows.

### Anti-Pattern 2: Invalidating the AST Cache on Schema Changes

**What people do:** Treat the AST cache and the compiled plan cache as one thing and wipe both on every write.

**Why it's wrong:** The AST (parse result) is pure text → AST transformation — no collection-specific information. Wiping it on writes loses 90%+ hit rate on hot queries.

**Do this instead:** Keep AST cache immutable (never invalidated). Only invalidate the compiled plan cache, and only for the affected collection's plans (keyed by collection name + query hash).

### Anti-Pattern 3: Training PQ on Every Insert

**What people do:** Call `ProductQuantizer::train()` after each upsert to keep the codebook current.

**Why it's wrong:** k-means training is O(n × m × k × iters) — prohibitively slow on hot paths. The existing code already buffers the first N vectors (`pq_training_buffer`) and trains once. Retraining destroys existing PQ codes.

**Do this instead:** Train once on the buffer (current pattern). Add a manual `RETRAIN` command (VelesQL extension) for offline retraining — never retrain automatically.

### Anti-Pattern 4: Holding Multiple Write Locks During Streaming

**What people do:** Acquire `vector_storage.write()` + `payload_storage.write()` + `pq_cache.write()` for the entire duration of a stream batch, blocking all concurrent reads.

**Why it's wrong:** Destroys read concurrency — especially bad for a search-heavy workload ingesting concurrently.

**Do this instead:** Process the stream batch in micro-batches (e.g., 64 points). Acquire write locks per micro-batch, release immediately. Each micro-batch is its own lock acquisition. This matches the existing `upsert()` pattern — the `StreamIngester` simply calls `upsert(micro_batch)` in a loop.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes for v1.5 |
|----------|---------------|----------------|
| `SparseIndex` ↔ `Collection` | Direct method calls under `RwLock<SparseIndex>` | Identical pattern to `HnswIndex` (already `Arc<HnswIndex>`) |
| `StreamIngester` ↔ `Collection` | `Arc<Collection>` held by background task | `Collection::upsert()` is the only entry point — no new coupling |
| `CompiledPlanCache` ↔ `QueryPlanner` | `CompiledPlanCache::get/put` wraps `QueryPlanner::plan()` | Cache is a decorator, not a replacement |
| `pq_simd` ↔ `simd_dispatch.rs` | New `distance_pq_dispatched()` function added to `simd_dispatch.rs` | Follows existing `cosine_dispatched / dot_product_dispatched` pattern |
| `SparseIndex` ↔ `fusion/` | Returns `Vec<(doc_id, f32)>` → existing `rrf_merge()` consumes it | No changes needed in `fusion/` |
| `write_generation` ↔ `CompiledPlanCache` | `AtomicU64::fetch_add` on every `upsert()` | No lock contention — atomic |

### External Surfaces (SDK Propagation)

v1.5 features that need propagation to adapter crates:

| Feature | velesdb-server | Python SDK | TypeScript SDK | WASM | Mobile |
|---------|---------------|------------|----------------|------|--------|
| Sparse Vectors | New endpoint + DTO | `sparse_search()` method | `sparseSearch()` | Yes (in-memory) | Yes |
| PQ (SIMD) | No new endpoint — transparent | No change | No change | Scalar fallback | Scalar fallback |
| Streaming Insert | New SSE/WS endpoint or chunked POST | `stream_insert()` | `streamInsert()` | No (`persistence` gated) | No |
| Plan Cache | No endpoint change | No change | No change | Yes (in-memory plans) | Yes |

---

## Build Order (Construction Dependencies)

The four features have the following dependency relationships, which determine safe build order:

```
EPIC-063 (PQ SIMD)
  └── Required by: nothing new — pure performance improvement to existing PQ path
  └── Depends on: existing simd_native/ + pq.rs
  └── Build first: no inter-feature dependencies

EPIC-062 (Sparse Vectors)
  └── Required by: hybrid search in EPIC-062 itself (uses existing fusion/)
  └── Depends on: existing index/, storage/, fusion/ — no other EPIC
  └── Build second: independent of EPIC-063/064/065

EPIC-065 (Advanced Caching)
  └── Depends on: existing QueryCache + QueryPlanner
  └── Requires write_generation field on Collection (simple AtomicU64 addition)
  └── Build third: must land before EPIC-064 since streaming inserts must increment generation

EPIC-064 (Streaming Inserts)
  └── Depends on: write_generation (EPIC-065 prerequisite), existing upsert()
  └── Requires: tokio (already in `persistence` feature)
  └── Build last: all other collection-level fields must be stable
```

**Recommended implementation sequence:**
1. EPIC-063: PQ SIMD kernels (pure algorithmic — zero API surface change)
2. EPIC-062: Sparse Vectors (new index module + new Collection field + new search path)
3. EPIC-065: Plan cache (new Collection field + write_generation + cache struct)
4. EPIC-064: Streaming inserts (new Collection field + background task)

---

## Compatibility with Existing Architecture

### 15MB Binary Budget

Each new feature's binary size contribution (estimated):

| Feature | New Code | Estimated Binary Addition |
|---------|----------|--------------------------|
| PQ SIMD kernels | ~200 lines unsafe AVX2 + ~100 NEON | ~5KB compiled |
| SparseIndex | ~600 lines | ~10KB compiled |
| StreamIngester | ~300 lines | ~5KB compiled |
| CompiledPlanCache | ~200 lines | ~3KB compiled |
| **Total** | ~1300 lines | **~23KB** |

All four features together add ~23KB to the binary — well within budget (current binary well under 15MB; 23KB is < 0.2% of budget). No new mandatory dependencies are required.

### WASM Compatibility

- **PQ SIMD:** AVX2/NEON kernels are `#[cfg(target_arch = "x86_64")]` / `#[cfg(target_arch = "aarch64")]` — automatically excluded from WASM. Scalar fallback already exists.
- **Sparse Vectors:** Postings list in-memory works in WASM. File persistence gated by `persistence` feature — already the pattern.
- **Streaming Inserts:** `tokio::task::spawn` is gated by `persistence` feature — WASM gets synchronous-only upsert, no streaming surface.
- **Plan Cache:** Fully in-memory; no `persistence` dependency — works in WASM.

### Lock Ordering Extension

The existing canonical ordering in `types.rs` (lines 147-160) must be extended:

```
Canonical order (acquire lower numbers first):
  1. config
  2. vector_storage
  3. payload_storage
  4. sq8_cache / binary_cache / pq_cache  (any order among themselves)
  5. pq_quantizer → pq_training_buffer
  6. secondary_indexes
  7. property_index / range_index         (any order among themselves)
  8. edge_store
  9. sparse_index                          [NEW — EPIC-062]
  // write_generation: AtomicU64 — lock-free, no ordering needed
  // insert_tx: Mutex<Option<Sender>> — only locked to clone/drop Sender
```

The `CompiledPlanCache` is a `RwLock<LRUMap>` accessed only under the query execution path, never while holding collection data locks — no ordering conflict.

---

## Sources

- Codebase analysis: `crates/velesdb-core/src/` (direct read, 2026-03-05)
- `quantization/pq.rs`: existing ADC implementation analyzed — confirms lookup-table approach
- `simd_native/dispatch/mod.rs`: `DistanceEngine` pattern analyzed — confirms extension point
- `collection/types.rs` lines 147-160: canonical lock ordering — confirmed extension point
- `velesql/cache.rs`: existing LRU AST cache analyzed — confirmed two-level cache rationale
- `collection/graph/streaming.rs`: existing streaming BFS iterator — confirms Rust streaming pattern in codebase
- `.planning/PROJECT.md`: v1.5 EPICs and constraints (15MB binary, WASM, ELv2)
- `.planning/codebase/ARCHITECTURE.md`: layer boundaries confirmed
- `.planning/codebase/STRUCTURE.md`: file locations confirmed
- FAISS ADC pattern (training data, HIGH confidence for the algorithm; not verified against live docs)
- Inverted index for sparse vectors (training data, HIGH confidence — industry standard since SPLADE 2021)

---

*Architecture research for: VelesDB v1.5 — Sparse Vectors, PQ SIMD, Streaming Inserts, Plan Cache*
*Researched: 2026-03-05*
