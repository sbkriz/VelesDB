# Feature Research

**Domain:** Local-first vector database — v1.5 feature additions (Sparse Vectors, Product Quantization, Streaming Inserts, Advanced Caching)
**Researched:** 2026-03-05
**Confidence:** MEDIUM — WebSearch and WebFetch were unavailable. Findings derive from training data (cutoff August 2025) + official docs cited in local codebase + internal VelesDB documentation. Competitor API patterns for Qdrant v1.9+, Milvus 2.4+, LanceDB, and Chroma are from training data and flagged accordingly.

---

## Context: VelesDB v1.4.1 Baseline

Before cataloguing v1.5 features, it is essential to understand what already exists so we do not re-build:

- HNSW native + SIMD (AVX-512/AVX2/NEON) with runtime dispatch — **done**
- SQ8 (4x) + Binary (32x) quantization — **done** (PQ is the *new* addition)
- RRF score fusion — **done**
- VelesQL 2.0 with GROUP BY / JOIN / UNION — **done**
- Graph traversal (BFS/DFS, multi-hop MATCH) — **done**
- REST API 22+ endpoints + SSE — **done**
- LangChain + LlamaIndex integrations — **done**
- Agent Memory SDK (semantic, episodic, procedural) — **done**

The v1.5 scope (EPIC-062 through EPIC-065) adds four specific subsystems and propagates them to all SDKs.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that users of vector DBs in 2025-2026 consider baseline. Missing them makes VelesDB feel incomplete relative to peers.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Sparse vector storage and search** (EPIC-062) | SPLADE/BM42 hybrid search is now the dominant RAG pattern in 2025. Qdrant has had it since v1.7 (2024-Q1). Milvus 2.4 added it. Users who do production RAG expect dense+sparse in one DB. | HIGH | Requires new index type, WASM-incompatible due to persistence requirement. CSR/COO storage format choice matters for memory efficiency. |
| **Product Quantization** (EPIC-063) | PQ is the industry-standard compression for >1M vector workloads. Users migrating from Faiss or Qdrant expect it. SQ8+Binary alone leave a gap for 8-32x compression with recall control. | HIGH | PQ requires k-means training phase before index build. Adds a blocking "train" step users must understand. OPAQUE: users set M (segments) and nbits (bits per code), not centroids directly. |
| **Streaming inserts / real-time ingestion** (EPIC-064) | AI agent workloads (episodic memory, live document ingestion) produce continuous insert streams. Forcing users to batch manually is a DX regression vs Chroma and Qdrant which accept unbatched inserts. | MEDIUM | Key contract questions: ordering guarantees, searchability lag after insert, backpressure signaling. |
| **Query plan cache** (EPIC-065) | Any SQL-adjacent system (VelesQL qualifies) that parses and plans queries at runtime is expected to cache plans. Without it, repeated identical queries pay repeated parse+plan cost. This is not a premium feature — it is expected infrastructure. | MEDIUM | VelesQL has a complex AST (pest-based). Plan cache invalidation on schema changes (collection rename/delete) is critical. TTL-based expiry is table stakes; LRU eviction is sufficient. |
| **Hybrid search (dense + sparse fusion)** | Dense-only search misses keyword-exact recall. BEIR benchmarks show hybrid (dense + sparse) consistently outperforms dense-alone for out-of-domain queries. Users know this. LangChain and LlamaIndex have built-in hybrid retrievers that expect both scores. | HIGH | Depends on sparse vector index existing. RRF is already implemented (v1.4). The missing piece is the sparse score source. |
| **Per-collection quantization config** | Users need to set quantization *per-collection*, not globally. Qdrant and Milvus both allow collection-level quantization params. Applying PQ to all collections or none is too coarse. | LOW | VelesDB already scopes SQ8/Binary per-collection. Extending this pattern to PQ is architectural extension, not a new pattern. |
| **Recall-preserving reranking after compression** | When PQ or SQ8 is enabled, users expect a *rescore* phase using original f32 vectors for the top-N candidates. Without this, compressed recall loss is unacceptable for production. | MEDIUM | Qdrant calls this "oversampling + rescore". VelesDB already implements two-stage ANN (candidate generation + exact SIMD rerank). Extending to PQ path is incremental. |
| **Documented compression tradeoffs** | Users need a clear table: f32 vs SQ8 vs PQ vs Binary — memory, recall@10, latency. The current QUANTIZATION.md partially covers this for SQ8/Binary but PQ docs will be missing. | LOW | Documentation task, but if absent, users cannot make informed decisions and will avoid the feature. |

### Differentiators (Competitive Advantage)

Features that distinguish VelesDB from Qdrant, Milvus, LanceDB, and Chroma in the local-first AI agent niche.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **VelesQL sparse vector syntax** (native SQL-like sparse search) | No competitor offers SQL-like sparse vector queries. Qdrant uses JSON payloads. Milvus uses gRPC/Python. If VelesDB expresses `WHERE sparse_vector NEAR_SPARSE $q LIMIT 10` in VelesQL, it is unique. | HIGH | Requires VelesQL grammar extension (`NEAR_SPARSE` or overloaded `NEAR` with type detection). Pest grammar change + conformance cases needed. |
| **Unified quantization pipeline** (SQ8 + Binary + PQ in one API) | Most local DBs offer one or two quantization modes. VelesDB targeting three with a unified configuration surface and automatic fallback ordering is rare at this scale. | MEDIUM | Requires a `QuantizationConfig` enum that routes to the right pipeline. Already have SQ8/Binary; adding PQ completes the triad. |
| **Streaming inserts with searchable-immediately guarantee** | Chroma and LanceDB buffer inserts and do not guarantee immediate searchability. If VelesDB can guarantee that an inserted vector is searchable in the next query (with configurable consistency mode), this is a real differentiator for AI agent workloads. | HIGH | Requires careful HNSW insertion locking. The current `RwLock` architecture supports this but the invariant must be explicit and tested. |
| **Plan cache with EXPLAIN integration** | VelesDB already has EXPLAIN support (v1.4). A plan cache that shows cache hit/miss in EXPLAIN output is pedagogically valuable and debug-friendly. No local vector DB exposes this. | MEDIUM | EXPLAIN already shows plan structure. Adding `cache_hit: true/false` and `plan_reuse_count: N` is a minor output extension. |
| **BM42 / learned sparse model support via custom tokenizer hook** | SPLADE/BM42 users bring their own tokenizer. Providing a hook (e.g., a callable in Python bindings) to produce sparse weights rather than requiring a fixed tokenization scheme is a DX win over competitors that hardcode BM25. | HIGH | Python side: pass `{term_id: weight}` dict. Rust side: accept `HashMap<u32, f32>` as sparse vector. This is an API design decision, not a separate system. |
| **Cross-collection sparse+dense hybrid in one VelesQL query** | No competitor supports cross-collection hybrid queries in SQL. If VelesDB can do `SELECT ... FROM docs WHERE vector NEAR $dense AND sparse_vector NEAR_SPARSE $sparse JOIN metadata ...` in one query, this is a unique capability. | HIGH | Requires sparse index in place + VelesQL parser extension + executor hybrid path. Only possible after EPIC-062. |
| **Streaming insert SSE progress feed** | The server already has SSE support. Exposing a streaming insert progress stream (vectors indexed, HNSW build progress, estimated time) over SSE is unique and directly useful for desktop apps using the Tauri plugin. | LOW | Small addition given existing SSE infrastructure. High visibility for Tauri/desktop users. |

### Anti-Features (Commonly Requested, Often Problematic)

Features to explicitly NOT build in v1.5, with reasoning.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Sparse GPU acceleration** | GPU is already supported for dense vectors. Users naturally request it for sparse too. | Extremely high complexity (sparse matrix ops on GPU require cuSPARSE-equivalent, no good wgpu path for sparse). Explicitly out of scope per PROJECT.md. Shipping a broken or slow implementation damages trust more than not shipping. | Document clearly that sparse operates on CPU only. Sparse vectors are typically much smaller than dense (non-zero entries only), so CPU is acceptable. |
| **Distributed sparse vector sharding** | Advanced users ask for horizontal scaling of sparse indexes. | Distributed mode is reserved for Premium (EPIC-061). Implementing it in open-core breaks the Premium boundary enforced via `DatabaseObserver`. | Defer to Premium tier. Local-first use cases do not need it. |
| **Auto-tokenization / built-in BM25 tokenizer** | Users want the DB to produce sparse vectors from raw text internally. | This creates a tokenization dependency (stemming, stopwords, language detection). It is an NLP problem, not a DB problem. Every deployment environment will want different tokenization. | Require users to produce sparse weights externally (via SPLADE model, BM42, or their own BM25). Accept `{term_id: weight}` format. This is how Qdrant v1.9 works with SPLADE. |
| **Online PQ centroid retraining** | Users want PQ centroids to automatically retrain as data drifts. | Online k-means retraining during live inserts is extremely complex and causes unpredictable latency spikes. Qdrant does not offer this either. | Expose an explicit `RETRAIN` command or API endpoint. Let users trigger retraining offline. Document the drift risk. |
| **Streaming inserts with strict total-order guarantee** | Some users want inserts in stream to be visible in exactly insertion order. | Strict total ordering requires a global sequence number and prevents concurrent insertions. Kills throughput. | Provide "searchable-immediately" (new vector visible in next query) with causal consistency per collection, not strict global order. This satisfies 95% of use cases. |
| **Query plan cache with automatic schema-aware invalidation** | Users expect the cache to magically invalidate when any schema changes. | Full schema-aware invalidation requires tracking every collection creation/deletion/rename and tying it to cached plan keys. Complex and error-prone. | Invalidate on collection deletion/rename (simple event hooks). For field-level changes (which VelesDB does not currently have as DDL), this is moot. Also expose `FLUSH PLAN CACHE` VelesQL command for manual invalidation. |
| **PQ with variable M per segment** | Power users request non-uniform segment widths (some dimensions quantized finer than others). | Increases implementation complexity 10x. No mainstream competitor supports this. | Uniform M segments is sufficient for all practical cases. Document that M should divide the dimension evenly. |

---

## Feature Dependencies

```
Sparse Vectors (EPIC-062)
    └──requires──> Sparse Index (inverted list or WAND-compatible structure)
    └──requires──> Sparse vector storage format (CSR in vectors.bin or separate file)
    └──enables──>  Hybrid Search (dense + sparse fusion)
                       └──uses──> RRF fusion (already in v1.4)
    └──enables──>  BM42/SPLADE insert path (API design only)

Product Quantization (EPIC-063)
    └──requires──> PQ training phase (k-means on sample vectors)
    └──requires──> PQ codebook storage (new file in collection dir)
    └──enhances──> Two-stage ANN (already partially implemented per ANN_SOTA_AUDIT.md)
                       └──uses──> PQ codes for coarse pass + f32 for rerank
    └──conflicts-with──> Binary quantization (cannot combine; must choose one)
    └──parallel-with──> SQ8 (can coexist: PQ at index level, SQ8 at storage level)

Streaming Inserts (EPIC-064)
    └──requires──> HNSW insertion lock correctness (already present)
    └──requires──> Buffer + flush semantics (new)
    └──enables──>  SSE progress feed (small addition)
    └──conflicts-with──> PQ training phase (cannot train PQ on empty/tiny collection;
                         streaming must handle "pre-trained" and "training pending" states)

Advanced Caching (EPIC-065)
    └──requires──> VelesQL plan reuse key (query fingerprint / normalized AST hash)
    └──requires──> Cache invalidation hooks on collection events (delete, rename)
    └──enhances──> Query performance for repeated RAG patterns (same query, different params)
    └──parallel-with──> All other EPICs (cache is orthogonal to index type)

Hybrid Search (dense + sparse)
    └──requires──> Sparse Vectors (EPIC-062) — MUST ship first
    └──uses──> RRF fusion (v1.4, already done)
    └──requires──> VelesQL NEAR_SPARSE syntax or overloaded NEAR
```

### Dependency Notes

- **Hybrid Search requires Sparse Vectors:** You cannot implement hybrid search without a sparse index. EPIC-062 is a hard prerequisite for hybrid search.
- **PQ conflicts with Binary quantization at collection level:** A collection should choose one compression mode. The `QuantizationConfig` enum should be a sum type (SQ8 | PQ | Binary | None), not a flags set.
- **Streaming Inserts and PQ have a non-obvious interaction:** If a collection has an untrained PQ index and a streaming insert arrives, the system needs a defined behavior. Options: (a) reject inserts until PQ is trained, (b) store as f32 and compress lazily, (c) queue inserts and batch-train when threshold is hit. This must be designed explicitly.
- **Advanced Caching is independent:** EPIC-065 has no hard dependencies on EPIC-062/063/064. It can be developed in parallel and shipped in any order relative to the others.

---

## MVP Definition

### For v1.5 (the current milestone)

VelesDB v1.5 is defined to include all four EPICs. The question is ordering within the milestone:

- [x] **Sparse Vectors (EPIC-062)** — Must ship before Hybrid Search can be claimed. Core index + storage + VelesQL syntax + Python/TS SDK propagation. Minimum: insert sparse vector, search sparse vector, return results. Hybrid fusion with dense is a v1.5 stretch goal once sparse search works.
- [x] **Product Quantization (EPIC-063)** — Must include: training API, search with PQ codes, rescore (rerank) phase. Minimum: `QuantizationConfig::ProductQuantization { m: u8, nbits: u8 }` per-collection, with training triggered on `collection.train_pq()` call.
- [x] **Streaming Inserts (EPIC-064)** — Must include: insert-and-immediately-search contract, backpressure (capacity limit with error), no forced client batching. Minimum: `db.insert_stream()` or `POST /collections/{name}/upsert` accepts unbatched single-vector payloads without forced batch wrapper.
- [x] **Advanced Caching (EPIC-065)** — Must include: LRU plan cache with configurable max-size, TTL per entry, cache hit/miss exposed in EXPLAIN output, `FLUSH PLAN CACHE` command in VelesQL. Minimum: 50% latency reduction on repeated identical queries in benchmarks.

### Add After v1.5 Validation

- [ ] **BM42 tokenizer hook** — Triggered when Python SDK users report friction providing sparse weights from SPLADE models.
- [ ] **Streaming insert SSE progress feed** — Add once streaming inserts are stable and users request observability.
- [ ] **Auto-tune PQ parameters** — Expose a `suggest_pq_params(sample_size)` API that recommends M and nbits based on collection statistics.
- [ ] **VelesQL `NEAR_SPARSE` syntax** — If the overloaded `NEAR` approach proves confusing, add dedicated syntax. Parser extension is straightforward given the pest-based grammar.

### Future Consideration (v2.0+)

- [ ] **Sparse vector GPU acceleration** — Only after GPU is mature for dense (currently wgpu-based, not production-hardened). Complexity too high for local-first workloads.
- [ ] **Online PQ centroid retraining** — Only if user data shows significant embedding distribution drift. Complex enough to warrant a separate version increment.
- [ ] **Cross-collection sparse+dense hybrid queries** — Architecturally possible after EPIC-062, but requires multi-source query planning. Defer to v2.0 query engine work.
- [ ] **Distributed sparse index sharding** — Premium-only. Reserved for `DatabaseObserver` extension.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Sparse vector storage + search | HIGH | HIGH | P1 |
| Hybrid dense+sparse fusion (after sparse) | HIGH | MEDIUM | P1 |
| Product Quantization (M-segment, nbits) | HIGH | HIGH | P1 |
| PQ rescore / rerank phase | HIGH | MEDIUM | P1 |
| Streaming inserts (searchable-immediately) | HIGH | MEDIUM | P1 |
| Query plan cache (LRU + TTL) | MEDIUM | MEDIUM | P1 |
| VelesQL NEAR_SPARSE syntax | MEDIUM | MEDIUM | P2 |
| EXPLAIN cache hit/miss output | MEDIUM | LOW | P2 |
| FLUSH PLAN CACHE VelesQL command | LOW | LOW | P2 |
| SSE streaming insert progress | LOW | LOW | P3 |
| Auto-tune PQ parameter suggestion | MEDIUM | MEDIUM | P3 |
| BM42 tokenizer hook (Python) | MEDIUM | LOW | P3 |

**Priority key:**
- P1: Must have for v1.5 launch — user-facing contract
- P2: Should have — high DX value, low risk
- P3: Nice to have — defer if timeline is tight

---

## Competitor Feature Analysis

*Confidence: MEDIUM — based on training data (Qdrant v1.9, Milvus 2.4, LanceDB 0.10+, Chroma 0.5+). Web verification unavailable.*

### Sparse Vectors

| Aspect | Qdrant v1.9+ | Milvus 2.4+ | LanceDB 0.10+ | Chroma 0.5+ | VelesDB v1.5 target |
|--------|-------------|-------------|----------------|-------------|---------------------|
| Sparse storage format | Compressed (id, value pairs) | BITMAP / inverted index | Lance columnar + inverted | Not native (external) | CSR or (u32, f32) pair list per vector |
| Supported models | SPLADE, SPLADEv2, BM25 | SPLADE, BM25, any sparse | SPLADE, custom | None native | SPLADE, SPLADEv2, BM42, BM25 (model-agnostic: accept `{term_id: weight}`) |
| Insert API | `vectors: {"sparse": {"indices": [...], "values": [...]}}` | `sparse_vectors: {field: SparseFloatField}` | `.add(sparse_embeddings=[...])` | N/A | `SparseVector { indices: Vec<u32>, values: Vec<f32> }` |
| Hybrid fusion | RRF (default), custom | Weighted sum, RRF | RRF | N/A | RRF (already in v1.4) |
| Index type | SPARSE_IVF | SPARSE_INVERTED_INDEX | IVF + BM25 | N/A | Inverted list (WAND-compatible) |
| SQL/query syntax | JSON payload | Python/gRPC API | SQL-like (LanceDB SQL) | Python API | `WHERE sparse_vector NEAR_SPARSE $q` (VelesQL) |

**Gap:** VelesDB has no sparse vector support at all in v1.4. This is the most impactful gap relative to Qdrant and Milvus for production 2025 RAG.

### Product Quantization

| Aspect | Qdrant v1.9+ | Milvus 2.4+ | LanceDB | VelesDB v1.5 target |
|--------|-------------|-------------|---------|---------------------|
| PQ config params | `m` (segments), `compression` (x4/x8/x16/x32) | `m` (segment count), `nbits` (4 or 8) | Built into IVF_PQ index | `m: u8` (8 or 16 typical), `nbits: u8` (4 or 8) |
| Typical M values | 16 (768D), 32 (1536D) | 8–64 | Automatic | 8–32, recommend `dim / 48` |
| Training required | Yes (background, triggered on build) | Yes (explicit `build_index`) | Yes (automatic on add) | Yes (explicit `train_pq(sample_fraction)`) |
| Oversampling/rescore | Yes (oversampling factor config) | Yes (rerank with raw vectors) | Yes | Yes (rescore top-N with f32) |
| Memory reduction | ~16–32x vs f32 | ~16–32x | Automatic | ~16x (M=16, nbits=8) to ~32x (M=8, nbits=4) |
| Recall@10 impact | ~95–98% with rescore | ~94–97% with rescore | N/A | Target: ≥95% with rescore enabled |

**Gap:** VelesDB has SQ8 (4x) and Binary (32x). PQ fills the 8–32x range with configurable recall vs compression. This is the compression range users need for 1M+ vector workloads.

**Qdrant API pattern (MEDIUM confidence, training data):**
```json
{
  "product": {
    "compression": "x16",
    "always_ram": true
  }
}
```

**Milvus API pattern (MEDIUM confidence, training data):**
```python
index_params = {
  "index_type": "IVF_PQ",
  "metric_type": "COSINE",
  "params": {"nlist": 128, "m": 16, "nbits": 8}
}
```

**VelesDB target API:**
```rust
QuantizationConfig::ProductQuantization {
    m: 16,        // Number of sub-quantizers (segments). dim must be divisible by m.
    nbits: 8,     // Bits per code. 4 or 8. nbits=8 → 256 centroids per segment.
    rescore: true // Rerank top candidates with f32 after PQ search.
}
```

### Streaming Inserts

| Aspect | Qdrant | Milvus | Chroma | LanceDB | VelesDB v1.5 target |
|--------|--------|--------|--------|---------|---------------------|
| Single-vector upsert | Yes, no batching required | Requires batch wrapper | Yes, no batching required | Yes | Yes (goal: single POST per vector) |
| Searchable after insert | Immediate (HNSW in-memory update) | Eventual (background indexing for large batches) | Immediate (brute-force until threshold) | Eventual | Immediate (HNSW in-memory, same lock semantics as current insert) |
| Backpressure | HTTP 429 when queue full | gRPC backpressure | None explicit | None explicit | HTTP 429 + `Retry-After` header when collection capacity limit reached |
| Ordering guarantee | None (concurrent inserts) | None (concurrent inserts) | None | None | Causal per-collection (each insert is visible in the next query on the same collection) |
| Batch optimization | Server-side batching transparent | Explicit batch API | Server-side | Auto-batching | Server-side buffer with configurable flush interval |

**Gap:** The current VelesDB server accepts individual vectors via `POST /collections/{name}/upsert` but the insert path is synchronous (blocks until HNSW update completes). "Streaming Inserts" means making this path scalable — handling high-frequency individual inserts with a buffer, without blocking the caller for the full HNSW insertion latency.

### Query Plan Cache

| Aspect | PostgreSQL | Qdrant | Milvus | VelesDB v1.5 target |
|--------|-----------|--------|--------|---------------------|
| Cache type | Prepared statement cache | None (no query language) | None (gRPC API, no SQL) | LRU plan cache on VelesQL AST |
| Cache key | Parameterized query text | N/A | N/A | Normalized AST fingerprint (params excluded) |
| TTL | None (explicit DISCARD) | N/A | N/A | Configurable TTL (default: indefinite until invalidation) |
| Invalidation | DDL events | N/A | N/A | Collection deletion/rename events |
| Observability | `EXPLAIN (ANALYZE)` | N/A | N/A | `EXPLAIN` shows `cache_hit: true/false` |
| Max size | Shared buffer | N/A | N/A | Configurable max entries (default: 1000) |

**Note:** Because Qdrant and Milvus do not have SQL-like query languages, this is a VelesDB-specific competitive advantage. The plan cache is most valuable for:
1. LangChain/LlamaIndex integration patterns where the same template query runs thousands of times per session with only the vector parameter changing.
2. VelesDB-CLI interactive sessions where the user refines the same query repeatedly.

### Hybrid Search (Dense + Sparse)

| Aspect | Qdrant | Milvus | Weaviate | VelesDB v1.5 target |
|--------|--------|--------|---------|---------------------|
| Fusion strategy | RRF (default), DBSF (Distribution-based) | Weighted sum, RRF, range-based | BM25 + vector hybrid | RRF (already in v1.4 for multi-score fusion) |
| Named vectors | Yes (each vector field named) | Yes (multiple vector fields per record) | Yes | Needs design: `vector` (dense) + `sparse_vector` (sparse) per record |
| Query syntax | JSON with per-vector params | Python SDK multi-vector | GraphQL | VelesQL: `WHERE vector NEAR $dense AND sparse_vector NEAR_SPARSE $sparse` |
| Score normalization | DBSF available | Cosine normalization | Automatic | RRF (score-rank based, no normalization needed) |

**Gap vs Qdrant:** Qdrant's DBSF (Distribution-Based Score Fusion) is more sophisticated than RRF for some workloads. RRF is sufficient for v1.5. DBSF can be added in v1.6 if user feedback requests it.

---

## Detailed Feature Specifications

### Sparse Vectors (EPIC-062)

**What users need (2025 RAG patterns):**

1. **SPLADE / SPLADEv2 / BM42 output format:** These models produce `{term_id: weight}` pairs where term IDs are vocabulary indices (integers, typically < 30,000 for bert-base vocabulary) and weights are float32 importance scores. The sparse vector for a document might have 50-200 non-zero entries out of 30,000+ vocabulary terms.

2. **Storage format recommendation:** CSR (Compressed Sparse Row) — store only non-zero (index, value) pairs sorted by index. This is what Qdrant uses internally. Memory per vector: `nnz * (4 + 4)` bytes where `nnz` is average non-zeros (typical: 50-300 for SPLADE).

3. **Index type:** Inverted posting list (like BM25 engine). For each term ID, maintain a list of (doc_id, weight) pairs sorted by weight descending. WAND (Weak AND) traversal for approximate search. MaxScore algorithm for exact search.

4. **API expected by Python users:**
```python
# Insert sparse vector
collection.insert({
    "id": "doc_1",
    "vector": [0.1, 0.2, ...],           # dense, 768-dim
    "sparse_vector": {                    # sparse, variable
        "indices": [102, 5483, 12041],
        "values": [0.82, 0.64, 0.31]
    },
    "payload": {"text": "..."}
})

# Search sparse
results = collection.search_sparse(
    query_sparse={"indices": [102, 5483], "values": [0.90, 0.71]},
    limit=10
)

# Hybrid search
results = collection.search_hybrid(
    query_vector=[0.1, 0.2, ...],
    query_sparse={"indices": [102, 5483], "values": [0.90, 0.71]},
    fusion="rrf",
    limit=10
)
```

5. **VelesQL syntax target:**
```sql
-- Sparse-only search
SELECT * FROM docs
WHERE sparse_vector NEAR_SPARSE $sparse_query
LIMIT 10;

-- Hybrid dense + sparse (RRF fusion)
SELECT * FROM docs
WHERE vector NEAR $dense_query
  AND sparse_vector NEAR_SPARSE $sparse_query
FUSION RRF
LIMIT 10;
```

**Complexity drivers:** The inverted index for sparse vectors must handle inserts without full rebuild. This requires an append-friendly structure (LSM-like posting lists or in-memory inverted index that merges on flush). This is architecturally different from HNSW.

### Product Quantization (EPIC-063)

**What users need:**

1. **Training phase:** PQ requires fitting k-means (typically 256 centroids per segment) on a sample of vectors. This is a one-time blocking operation. Users accept this. Typical training data: 10,000-100,000 vectors (10-20% of collection, capped). Training time for M=16, 768D, 50K vectors: ~30 seconds on modern CPU.

2. **Standard configurations (industry observed, MEDIUM confidence):**
   - **768D embeddings (BERT, all-MiniLM):** `m=16, nbits=8` → 16-byte codes (16x compression vs f32, 32x vs binary overhead). Recall@10 with rescore: ~97%.
   - **1536D embeddings (text-embedding-3-large):** `m=32, nbits=8` → 32-byte codes. Recall@10 with rescore: ~96%.
   - **384D embeddings:** `m=8, nbits=8` → 8-byte codes (48x compression). Recall@10 with rescore: ~95%.

3. **Rescore is mandatory for production recall:** PQ-only search gives ~85-92% recall at k=10. With rescore (fetch top-4x candidates from PQ, rerank with f32 SIMD), recall@10 rises to ~95-98%. Rescore must be enabled by default.

4. **Codebook storage:** Each segment has `2^nbits` centroids of dimension `dim/m`. Storage: `m * 2^nbits * (dim/m) * 4` bytes. For M=16, nbits=8, dim=768: `16 * 256 * 48 * 4 = 7.86 MB`. This fits in RAM with room. Store in `pq_codebook.bin` in collection directory.

5. **User-facing API (Rust):**
```rust
let config = CollectionConfig {
    dimension: 768,
    quantization: QuantizationConfig::ProductQuantization {
        m: 16,
        nbits: 8,
        rescore: true,
        rescore_factor: 4,  // Fetch 4*k candidates for rescore
    },
    ..Default::default()
};
collection.train_pq(sample_fraction: 0.1)?;  // Blocking training
collection.build_pq_index()?;               // Encode all vectors
```

### Streaming Inserts (EPIC-064)

**What users need:**

The core user story: "I am running an AI agent that inserts memory entries continuously. I do not want to manage batching. Each insert should succeed and be immediately searchable."

1. **No forced batching:** Current API likely accepts one vector at a time (via REST). "Streaming" means this path is optimized for high-frequency single-item inserts without degrading HNSW or blocking.

2. **Backpressure contract:**
   - When the collection's in-memory write buffer is full, return HTTP 429 with `Retry-After: N` seconds.
   - Default buffer: 1000 pending vectors. Configurable per collection.
   - Buffer flush: background thread flushes to WAL + HNSW when buffer hits 50% or every 100ms.

3. **Searchability guarantee:** After `insert()` returns success (HTTP 200/201), the vector MUST be findable in the next `search()` call. This is the "immediately searchable" contract. Implementation: insert directly into HNSW in-memory graph under write lock, then write to WAL. Do not defer HNSW update.

4. **Ordering:**
   - No total-order guarantee. Concurrent inserts may interleave.
   - Causal guarantee: if insert A returns before insert B is submitted, A is visible when B's search runs.
   - This matches what Qdrant provides (no stronger guarantee needed for AI agent workloads).

5. **Python API target:**
```python
# Sync insert (blocking until searchable)
db.insert("memories", {"vector": embedding, "payload": {"text": "..."}})

# Async stream insert (fire-and-forget with backpressure)
async with db.stream_insert("memories") as stream:
    async for chunk in document_stream:
        embedding = model.encode(chunk)
        await stream.insert({"vector": embedding, "payload": {"text": chunk}})
        # Backpressure: stream.insert() blocks if buffer is full
```

### Advanced Caching (EPIC-065)

**What users need:**

1. **Plan cache for VelesQL:** Parse and plan cost is dominated by pest grammar parsing + AST construction + plan selection. For a 100-token VelesQL query, parsing takes ~50-200µs (estimate). For 1000 QPS of identical template queries, this is 50-200ms/s of pure parse overhead.

2. **Cache key design:** Normalize the query by stripping parameter values. `SELECT * FROM docs WHERE vector NEAR $q LIMIT 10` with `$q = [0.1, ...]` should cache-hit the same plan regardless of `$q` value. Key = hash(normalized_query_text).

3. **TTL and eviction:**
   - Default TTL: None (plan valid until explicit invalidation)
   - Eviction policy: LRU with max size (default: 1000 plans)
   - Invalidation triggers: collection deletion, collection rename

4. **Observability (critical for adoption):**
```sql
EXPLAIN SELECT * FROM docs WHERE vector NEAR $q LIMIT 10;
-- Output includes:
-- plan_source: "cache_hit" | "freshly_planned"
-- plan_reuse_count: 1423
-- estimated_cost: 0.004
-- chosen_strategy: "HnswScan"
```

5. **VelesQL management:**
```sql
FLUSH PLAN CACHE;                        -- Invalidate all cached plans
FLUSH PLAN CACHE FOR COLLECTION 'docs'; -- Invalidate plans for one collection
SHOW PLAN CACHE STATS;                  -- Hit rate, size, evictions
```

---

## What VelesDB Is Missing vs Competitors (Gap Analysis)

*Sorted by user impact.*

| Gap | Affected Competitors | User Impact | v1.5 Addresses? |
|-----|---------------------|-------------|-----------------|
| No sparse vector support | Qdrant, Milvus, Weaviate (BM25) | HIGH — 2025 production RAG requires hybrid search | YES (EPIC-062) |
| No Product Quantization | Qdrant, Milvus, Faiss (IVF_PQ) | HIGH — 1M+ vector workloads need PQ | YES (EPIC-063) |
| No query plan cache | PostgreSQL, any SQL DB | MEDIUM — repeated query latency | YES (EPIC-065) |
| Streaming inserts not optimized | Chroma, Qdrant | MEDIUM — AI agent memory patterns | YES (EPIC-064) |
| No server authentication | All | HIGH — production safety | NO (deferred to Premium) |
| No DiskANN / disk-backed ANN | Milvus (DiskANN), FAISS | MEDIUM — datasets > RAM | NO — HNSW in-memory is the v1.5 target |
| No IVF partition index | Milvus (IVF_PQ), FAISS | LOW — local-first rarely needs IVF | NO — deferred, ANN_SOTA_AUDIT identifies as future work |
| No HNSW on disk (mmap traversal) | Qdrant (segment-based) | LOW — local datasets usually < 10M vectors | NO — deferred |
| No FP16/BF16 vector storage | Milvus | LOW — bandwidth optimization | NO — deferred to v2.0 (EPIC-055) |
| No RBAC / multi-tenancy | Qdrant Cloud, Milvus | N/A for open-core | NO — Premium via DatabaseObserver |
| bincode 1.x advisory (RUSTSEC-2025-0141) | N/A | MEDIUM — security posture | v1.5 bug-fix scope (from CONCERNS.md) |

---

## Sources

- VelesDB internal documentation: `docs/guides/QUANTIZATION.md`, `docs/ANN_SOTA_AUDIT.md`, `docs/guides/SEARCH_MODES.md`, `docs/reference/velesdb-core-improvement-research.md`
- VelesDB PROJECT.md (v1.5 feature scope definition)
- VelesDB CONCERNS.md (known gaps and tech debt)
- Qdrant documentation (training data, MEDIUM confidence): sparse vector API, PQ config, oversampling/rescore pattern
- Milvus documentation (training data, MEDIUM confidence): IVF_PQ params (m, nbits, nlist), sparse inverted index
- LanceDB documentation (training data, MEDIUM confidence): Lance columnar format, hybrid search
- Chroma documentation (training data, MEDIUM confidence): single-vector insert model, brute-force threshold
- PostgreSQL documentation (training data, HIGH confidence): prepared statement cache, plan invalidation model
- SPLADE / BM42 research papers (training data, HIGH confidence): output format (`{term_id: weight}`), average non-zeros, typical vocabulary size

---

*Feature research for: VelesDB v1.5 — Sparse Vectors, Product Quantization, Streaming Inserts, Advanced Caching*
*Researched: 2026-03-05*
