# Phase 5: Sparse Integration - Research

**Researched:** 2026-03-06
**Domain:** VelesQL grammar extension, hybrid dense+sparse fusion, REST API sparse endpoints, WASM sparse bindings
**Confidence:** HIGH

## Summary

Phase 5 integrates the sparse vector engine (Phase 4) into all user-facing surfaces: VelesQL grammar, the query executor, the REST API, and WASM bindings. The core challenge is a multi-layered integration touching the pest grammar, AST, parser, query planner/executor, Collection CRUD, REST handlers, and WASM -- all while maintaining backward compatibility with the existing `sparse_vector: Option<SparseVector>` field on `Point` and the current `Option<SparseInvertedIndex>` on `Collection`.

Key findings: (1) The grammar, parser, and AST are cleanly extensible -- `primary_expr` dispatches by pest Rule, and adding `sparse_vector_search` follows the exact pattern of `vector_search`. (2) The existing `FusionStrategy::fuse()` accepts `Vec<Vec<(u64, f32)>>` and is directly usable for hybrid dense+sparse RRF without modification. (3) The current `Collection::upsert()` does NOT index sparse vectors into the sparse index -- this is a CRUD integration gap that must be closed. (4) Named sparse vectors (`BTreeMap<String, SparseVector>`) require a `Point` struct migration touching ~19 files (same scope as Phase 4's `sparse_vector: None` addition).

**Primary recommendation:** Layer the integration bottom-up: Point struct migration + CRUD sparse indexing first, then grammar/parser/AST, then executor+fusion, then REST API, then WASM. Each layer is independently testable.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Named Sparse Vectors:** `Point.sparse_vectors: Option<BTreeMap<String, SparseVector>>` replacing `sparse_vector: Option<SparseVector>`. Default unnamed maps to `""`. Per-name index stored as `BTreeMap<String, SparseInvertedIndex>` in `Collection`. Persistence: `sparse-{name}.idx/terms/wal/meta` files, backward compat for default name.
- **VelesQL SPARSE_NEAR:** Dedicated keyword `SPARSE_NEAR` with optional `USING 'name'` clause. Bind parameters and inline literals (`{12: 0.8, 45: 0.3}`). AST: `Condition::SparseVectorSearch(SparseVectorSearch)` as dedicated variant.
- **Three query modes:** Dense-only (`NEAR`), sparse-only (`SPARSE_NEAR`), hybrid (`NEAR AND SPARSE_NEAR`).
- **Fusion WITH clause:** `WITH (fusion='weighted', dense_w=0.7, sparse_w=0.3)`. Default = implicit RRF k=60.
- **Hybrid execution:** `rayon::join` parallel dense+sparse, 2x oversampling per branch, graceful degradation if one branch empty.
- **RSF (Relative Score Fusion):** `FusionStrategy::RelativeScore` with min-max normalization. `WITH (fusion='rsf', dense_w=0.7, sparse_w=0.3)`.
- **Sparse + payload filter:** `sparse_search_filtered()` with `Option<&dyn Fn(u64) -> bool>`, 4x oversampling, retry with larger if insufficient.
- **ScoreBreakdown extension:** Add `sparse_score`, `sparse_rank`, `dense_rank`, `fusion_method` fields.
- **REST API:** Extend existing `/collections/{name}/points` and `/collections/{name}/search`. Accept both `sparse_vector` (single) and `sparse_vectors` (named map). Auto-detect search mode.
- **term_id u32:** Full u32 space, validation at API boundary, boundary test with `u32::MAX - 1`.
- **WASM sparse search:** In-memory only, expose create/insert/search/hybrid in velesdb-wasm.
- **Conformance cases:** Add sparse cases to `conformance/velesql_parser_cases.json` (should_parse: true/false).

### Claude's Discretion
- Exact oversampling tuning beyond 2x/4x defaults
- Internal planner hybrid detection logic
- Server-side dict-to-array conversion impl
- Conformance test case selection beyond minimum
- Error message wording
- EXPLAIN output format details
- Sequential fallback threshold for rayon::join
- Inline sparse literal size guard-rail threshold
- Default sparse index name convention (empty string vs "default")
- RSF weight normalization edge cases (all equal scores)

### Deferred Ideas (OUT OF SCOPE)
- Cosine-normalized sparse search
- Sparse NEAR_FUSED (multi-sparse-vector fusion)
- VByte / delta encoding for term_id compression
- Sparse vector weight quantization (Float16/Uint8)
- String-to-term_id vocabulary service
- Nested prefetch pipeline composition
- Late interaction / ColBERT reranking
- Server-side BM25 tokenization
- Better normalization beyond min-max for RSF
- Batch hybrid search endpoint
- Sparse index statistics in EXPLAIN
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SPARSE-04 | Hybrid dense+sparse via RRF existing -- `fusion/rrf_merge()` non modifie, fonctionne out-of-the-box | `FusionStrategy::fuse()` takes `Vec<Vec<(u64, f32)>>` -- ScoredDoc-to-tuple bridge needed. rayon::join for parallel execution. RSF as alternative strategy. |
| SPARSE-05 | Extension grammaire VelesQL -- keyword `SPARSE` et clause `vector SPARSE_NEAR $sv` dans grammar.pest | New pest rules: `sparse_vector_search`, `sparse_value`, `sparse_literal`, `sparse_entry`. New AST variant `Condition::SparseVectorSearch`. Parser dispatch in `parse_primary_expr_with_depth`. |
| SPARSE-06 | API REST -- endpoints upsert avec champ `sparse_vector` + search sparse endpoint | Extend `PointRequest` and `SearchRequest` in server types.rs. Custom Deserialize for dict format. utoipa annotations for OpenAPI. |
| SPARSE-07 | term_id u32 (4 milliards de termes) | `SparseVector.indices` already Vec<u32>. Validation at REST/VelesQL boundary. Integration test with u32::MAX-1. |
</phase_requirements>

## Architecture Patterns

### Integration Layers (Bottom-Up)

```
Layer 1: Data Model (Point struct, Collection sparse_index)
  |
Layer 2: CRUD (upsert indexes sparse vectors, get returns them)
  |
Layer 3: Grammar/Parser/AST (pest rules, parser, Condition variant)
  |
Layer 4: Executor (extract sparse search, execute, fuse)
  |
Layer 5: REST API (types, handlers, OpenAPI)
  |
Layer 6: WASM Bindings (in-memory sparse search)
  |
Layer 7: Conformance (parser cases, contract cases)
```

### Current Code Structure (Integration Points)

**Point struct** (`point.rs` line 16-30):
- Currently: `pub sparse_vector: Option<SparseVector>`
- Change to: `pub sparse_vectors: Option<BTreeMap<String, SparseVector>>`
- ~19 files reference `sparse_vector: None` in Point literals (from Phase 4)
- Backward compat serde: accept both `sparse_vector` and `sparse_vectors`

**Collection sparse_index** (`collection/types.rs` line 228-230):
- Currently: `pub(super) sparse_index: Arc<RwLock<Option<SparseInvertedIndex>>>`
- Change to: `pub(super) sparse_index: Arc<RwLock<BTreeMap<String, SparseInvertedIndex>>>`
- `lifecycle.rs` loads single sparse index as `None` or `Some` -- must migrate to map
- `load_sparse_index()` must scan for `sparse-*.meta` files and load each named index

**Collection::upsert()** (`collection/core/crud.rs` line 22-125):
- Currently: Does NOT index sparse vectors at all (gap from Phase 4)
- Must add: for each point with sparse_vectors, insert into corresponding named index
- Lock ordering: sparse_index lock must be documented relative to vector_storage/payload_storage

**Grammar** (`velesql/grammar.pest` line 167-180):
- `primary_expr` lists conditions in precedence order
- Add `sparse_vector_search` rule before `vector_search` (longer keyword match first)
- New rules: `sparse_vector_search`, `sparse_value`, `sparse_literal`, `sparse_entry`

**Parser** (`velesql/parser/conditions.rs` line 117-156):
- `parse_primary_expr_with_depth` dispatches on `Rule::` variants
- Add `Rule::sparse_vector_search => Self::parse_sparse_vector_search(inner)`
- New method: `parse_sparse_vector_search`, `parse_sparse_value`, `parse_sparse_literal`

**AST** (`velesql/ast/condition.rs`):
- Add `SparseVectorSearch(SparseVectorSearch)` variant to `Condition` enum
- New struct `SparseVectorSearch { vector: SparseVectorExpr, index_name: Option<String> }`
- New enum `SparseVectorExpr { Literal(SparseVector), Parameter(String) }`

**Executor** (`collection/search/query/mod.rs` line 98-280):
- `execute_query_with_client` extracts vector search via `extract_vector_search`
- Must add: `extract_sparse_vector_search` parallel extraction
- When both dense and sparse detected: parallel rayon::join execution + RRF fusion
- When sparse only: direct sparse search with optional filter

**Fusion** (`fusion/strategy.rs`):
- `FusionStrategy::fuse(Vec<Vec<(u64, f32)>>)` -- use as-is
- Add `FusionStrategy::RelativeScore { dense_weight: f32, sparse_weight: f32 }`
- Bridge: `ScoredDoc { doc_id, score }` -> `(u64, f32)` is trivial

**ScoreBreakdown** (`collection/search/query/score_fusion/mod.rs` line 32-58):
- Add fields: `sparse_score: Option<f32>`, `sparse_rank: Option<u32>`, `dense_rank: Option<u32>`, `fusion_method: Option<String>`

**REST types** (`velesdb-server/src/types.rs`):
- `PointRequest` (line 78-85): Add `sparse_vector`, `sparse_vectors` fields
- `SearchRequest` (line 93-116): Add `sparse_vector`, `sparse_vectors`, `sparse_index`, `fusion` fields
- New: `SparseVectorInput { indices: Vec<u32>, values: Vec<f32> }` with custom Deserialize for dict format

**REST handlers** (`velesdb-server/src/handlers/search.rs`):
- `search()` handler: detect sparse params, route to appropriate search method
- `points.rs` upsert handler: pass sparse vectors through to Collection::upsert

**WASM** (`velesdb-wasm/src/`):
- Pattern: modules export `#[wasm_bindgen]` wrapper functions calling velesdb-core
- Add `sparse.rs` module: `SparseIndex`, `sparse_search`, `hybrid_search`
- No persistence feature -- in-memory only

### Anti-Patterns to Avoid
- **Modifying fusion/strategy.rs fuse() signature:** The existing RRF fuse works perfectly. Bridge ScoredDoc->tuple at the call site instead.
- **Coupling grammar rules to AST types:** Keep SparseVectorExpr separate from VectorExpr -- they serve different type systems (u32 indices vs f32 arrays).
- **Global lock for named sparse indexes:** Use per-name locking or a RwLock<BTreeMap> where the map lock is held briefly to get/insert entries.
- **Blocking upsert on sparse indexing:** Sparse index insert is cheap (append to posting list), don't introduce async boundaries.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RRF fusion | Custom merge logic | `FusionStrategy::RRF { k: 60 }.fuse()` | Already handles dedup, ranking, sorting |
| Post-filter pattern | Custom filter loop | Follow `search_with_filter()` 4x oversampling pattern | Established, tested pattern in codebase |
| Sparse pest parsing | Manual string parsing | pest grammar rules with `parse_pair` | Type safety, error reporting, conformance |
| Serde dual-format | Manual JSON parsing | Custom `Deserialize` impl with `#[serde(untagged)]` or visitor | serde ecosystem handles edge cases |
| Parallel execution | Manual thread spawn | `rayon::join(|| dense, || sparse)` | Already a dependency, handles thread pool |

## Common Pitfalls

### Pitfall 1: Point Struct Migration Cascade
**What goes wrong:** Changing `sparse_vector` to `sparse_vectors` breaks ~19 files with Point literal construction.
**Why it happens:** Phase 4 added `sparse_vector: None` to every Point construction site.
**How to avoid:** Use find-and-replace systematically. Consider adding a helper method `Point::new()` that doesn't require sparse fields. Use `#[serde(alias = "sparse_vector")]` for backward compat deserialization.
**Warning signs:** Compiler errors in unrelated test files.

### Pitfall 2: Pest Grammar Keyword Ambiguity
**What goes wrong:** `SPARSE_NEAR` might conflict with identifier parsing if not handled as a keyword.
**Why it happens:** pest is PEG-based and matches in order -- `SPARSE_NEAR` must be tried before identifiers.
**How to avoid:** Place `sparse_vector_search` BEFORE `vector_search` in `primary_expr` alternatives. Use `^"SPARSE_NEAR"` for case-insensitive matching. The `^` prefix in pest means case-insensitive.
**Warning signs:** Queries parse but produce wrong AST nodes.

### Pitfall 3: Lock Ordering with Named Indexes
**What goes wrong:** Deadlock when upsert acquires sparse_index lock while holding vector_storage lock.
**Why it happens:** Current lock ordering: config(1) < vector_storage(2) < payload_storage(3). sparse_index must fit into this hierarchy.
**How to avoid:** Document sparse_index lock position (e.g., position 4, after payload_storage). Never hold sparse_index while acquiring vector_storage or config.
**Warning signs:** Deadlocks in multi-threaded tests (but tests run single-threaded, so may not catch this).

### Pitfall 4: Serde Backward Compatibility
**What goes wrong:** Existing serialized Points with `sparse_vector` field fail to deserialize after rename to `sparse_vectors`.
**Why it happens:** Field name change breaks stored JSON.
**How to avoid:** Use `#[serde(alias = "sparse_vector")]` on the new field, plus custom deserialization that accepts both formats. Also accept a single `SparseVector` and wrap it in a map under default name.
**Warning signs:** Deserialization errors when loading Phase 4 test data or persisted collections.

### Pitfall 5: Fusion Score Scale Mismatch
**What goes wrong:** Dense cosine scores (0-1) vs sparse inner product scores (unbounded) produce skewed RRF results.
**Why it happens:** RRF is rank-based not score-based, so this actually works fine. But RSF with min-max normalization can produce degenerate results when all scores are equal (division by zero).
**How to avoid:** In RSF normalization, handle `range < EPSILON` by assigning all scores 0.5 (or 1.0). RRF needs no normalization.
**Warning signs:** All results have identical fusion scores.

### Pitfall 6: WASM Feature Gate
**What goes wrong:** Sparse code that uses `rayon` or persistence features fails to compile for WASM.
**Why it happens:** rayon requires threads, persistence requires mmap/fs -- neither available in WASM.
**How to avoid:** Sparse index core types (SparseInvertedIndex, SparseVector, sparse_search) are already persistence-free. Hybrid execution in WASM must use sequential (not rayon::join). Gate parallel code with `#[cfg(feature = "persistence")]`.
**Warning signs:** WASM build failures with `cargo build -p velesdb-wasm --no-default-features --target wasm32-unknown-unknown`.

### Pitfall 7: Conformance Test Scope
**What goes wrong:** Parser cases pass but contract cases fail because the executor doesn't handle sparse.
**Why it happens:** Parser cases test grammar only. Contract cases test end-to-end via server.
**How to avoid:** Add both parser cases (grammar acceptance) and contract cases (semantic correctness). Parser cases are simpler and should be done first.
**Warning signs:** Parser tests green but integration tests red.

## Code Examples

### Pattern 1: Adding a pest Grammar Rule (from existing vector_search)
```rust
// grammar.pest - existing pattern to follow
vector_search = {
    ^"vector" ~ ^"NEAR" ~ vector_value
}

// New rule (following same pattern):
sparse_vector_search = {
    ^"vector" ~ ^"SPARSE_NEAR" ~ sparse_value ~ (^"USING" ~ string)?
}
sparse_value = { sparse_literal | parameter }
sparse_literal = { "{" ~ sparse_entry ~ ("," ~ sparse_entry)* ~ "}" }
sparse_entry = { integer ~ ":" ~ float }
```

### Pattern 2: Adding a Condition Variant (from existing VectorSearch)
```rust
// ast/condition.rs - existing pattern:
pub enum Condition {
    VectorSearch(VectorSearch),
    // New:
    SparseVectorSearch(SparseVectorSearch),
    // ...
}

pub struct SparseVectorSearch {
    pub vector: SparseVectorExpr,
    pub index_name: Option<String>,
}

pub enum SparseVectorExpr {
    Literal(SparseVector),  // Inline {12: 0.8, 45: 0.3}
    Parameter(String),       // $sv bind parameter
}
```

### Pattern 3: Parser Dispatch (from existing conditions.rs)
```rust
// parser/conditions.rs - line ~127, add new match arm:
match inner.as_rule() {
    // ... existing arms ...
    Rule::sparse_vector_search => Self::parse_sparse_vector_search(inner),
    Rule::vector_search => Self::parse_vector_search(inner),
    // ...
}
```

### Pattern 4: Hybrid Fusion in Executor
```rust
// Simplified hybrid execution pattern:
use rayon::join;
use crate::fusion::strategy::FusionStrategy;

let oversampled_k = limit * 2;
let (dense_results, sparse_results) = rayon::join(
    || self.search_ids_with_adc_if_pq(dense_query, oversampled_k),
    || {
        let idx = self.sparse_index.read();
        if let Some(index) = idx.get(sparse_index_name) {
            sparse_search(index, sparse_query, oversampled_k)
                .into_iter()
                .map(|sd| (sd.doc_id, sd.score))
                .collect()
        } else {
            Vec::new()
        }
    },
);

let strategy = FusionStrategy::RRF { k: 60 };
let fused = strategy.fuse(vec![dense_results, sparse_results])?;
let top_k: Vec<_> = fused.into_iter().take(limit).collect();
```

### Pattern 5: RSF Implementation
```rust
fn normalize_min_max(results: &mut [(u64, f32)]) {
    if results.is_empty() { return; }
    let (min, max) = results.iter().fold(
        (f32::MAX, f32::MIN),
        |(lo, hi), &(_, s)| (lo.min(s), hi.max(s))
    );
    let range = max - min;
    if range > f32::EPSILON {
        for (_, s) in results.iter_mut() {
            *s = (*s - min) / range;
        }
    } else {
        // All scores equal -- normalize to 0.5
        for (_, s) in results.iter_mut() {
            *s = 0.5;
        }
    }
}
```

### Pattern 6: Custom Serde for Dict Format (REST API)
```rust
use serde::{Deserialize, Deserializer};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SparseVectorInput {
    // Canonical: {"indices": [42, 1337], "values": [0.5, 1.2]}
    Parallel { indices: Vec<u32>, values: Vec<f32> },
    // Dict: {"42": 0.5, "1337": 1.2}
    Dict(std::collections::BTreeMap<String, f32>),
}

impl SparseVectorInput {
    fn into_sparse_vector(self) -> Result<SparseVector, String> {
        match self {
            Self::Parallel { indices, values } => {
                if indices.len() != values.len() {
                    return Err("indices and values must have equal length".into());
                }
                Ok(SparseVector::new(
                    indices.into_iter().zip(values).collect()
                ))
            }
            Self::Dict(map) => {
                let pairs: Result<Vec<_>, _> = map.into_iter()
                    .map(|(k, v)| k.parse::<u32>().map(|idx| (idx, v))
                        .map_err(|_| format!("invalid term_id: {k}")))
                    .collect();
                Ok(SparseVector::new(pairs?))
            }
        }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single sparse_vector per point | Named sparse vectors (BTreeMap) | Phase 5 | Multi-model support (BGE-M3, SPLADE title+body) |
| No sparse search in VelesQL | SPARSE_NEAR keyword | Phase 5 | First-class sparse query support |
| Manual hybrid fusion | Implicit RRF auto-fusion | Phase 5 | Zero-config hybrid search |
| RRF only | RRF + RSF | Phase 5 | Score-aware fusion option |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (built-in) + Criterion benchmarks |
| Config file | Cargo.toml workspace test configuration |
| Quick run command | `cargo test -p velesdb-core test_name -- --test-threads=1` |
| Full suite command | `cargo test --workspace --features persistence,gpu,update-check --exclude velesdb-python -- --test-threads=1` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SPARSE-04 | Hybrid dense+sparse RRF fusion returns correct merged results | integration | `cargo test -p velesdb-core test_hybrid_dense_sparse_rrf -- --test-threads=1` | No - Wave 0 |
| SPARSE-04 | RSF fusion normalizes and weights correctly | unit | `cargo test -p velesdb-core test_rsf_normalization -- --test-threads=1` | No - Wave 0 |
| SPARSE-04 | Graceful degradation when one branch returns empty | unit | `cargo test -p velesdb-core test_hybrid_single_branch -- --test-threads=1` | No - Wave 0 |
| SPARSE-05 | VelesQL parser accepts SPARSE_NEAR clause | unit | `cargo test -p velesdb-core test_parse_sparse_near -- --test-threads=1` | No - Wave 0 |
| SPARSE-05 | VelesQL parser accepts hybrid NEAR AND SPARSE_NEAR | unit | `cargo test -p velesdb-core test_parse_hybrid_query -- --test-threads=1` | No - Wave 0 |
| SPARSE-05 | Conformance cases pass in parser | integration | `cargo test -p velesdb-core velesql_parser_conformance -- --test-threads=1` | Yes (existing) |
| SPARSE-06 | REST upsert with sparse_vectors field succeeds | integration | `cargo test -p velesdb-server test_upsert_sparse -- --test-threads=1` | No - Wave 0 |
| SPARSE-06 | REST search with sparse_vector returns results | integration | `cargo test -p velesdb-server test_search_sparse -- --test-threads=1` | No - Wave 0 |
| SPARSE-07 | term_id u32::MAX-1 roundtrips correctly | integration | `cargo test -p velesdb-core test_u32_max_term_id -- --test-threads=1` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p velesdb-core --features persistence -- --test-threads=1` (core crate)
- **Per wave merge:** Full workspace test suite
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Sparse parser tests (new file or add to parser_tests.rs)
- [ ] Hybrid fusion tests (new file or extend fusion tests)
- [ ] REST sparse endpoint tests (extend server integration tests)
- [ ] u32 boundary test (extend sparse index tests)
- [ ] Conformance cases in `conformance/velesql_parser_cases.json`
- [ ] WASM sparse tests (extend lib_tests.rs)

## Open Questions

1. **Lock ordering for sparse_index in named index map**
   - What we know: Current lock order is config(1) < vector_storage(2) < payload_storage(3). sparse_index is currently acquired independently.
   - What's unclear: Where exactly sparse_index fits in the hierarchy when upsert needs to write to both vector_storage and sparse_index.
   - Recommendation: Acquire sparse_index AFTER releasing vector_storage and payload_storage (position 4). The upsert can buffer sparse vectors during the main upsert loop, then batch-insert them into sparse indexes after releasing storage locks.

2. **Backward compat for persistence file naming**
   - What we know: Phase 4 uses `sparse.idx`, `sparse.meta`, etc. for the single unnamed index.
   - What's unclear: Whether to rename existing files or keep them as-is for the default name.
   - Recommendation: Default name `""` maps to `sparse.idx` (no prefix). Named indexes use `sparse-{name}.idx`. On load, scan for both patterns.

3. **Sequential fallback for rayon::join in WASM/test**
   - What we know: rayon is behind `persistence` feature. WASM has no persistence.
   - What's unclear: Best pattern for conditional parallel vs sequential.
   - Recommendation: Use `#[cfg(feature = "persistence")]` to gate rayon::join. Provide a sequential fallback that runs dense then sparse. The overhead is minimal for typical query sizes.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection of all integration points (grammar.pest, ast/, parser/, fusion/, collection/, server/)
- Phase 4 implementation: `index/sparse/` module (types.rs, search.rs, inverted_index.rs, persistence.rs)
- CONTEXT.md user decisions -- comprehensive locked decisions

### Secondary (MEDIUM confidence)
- pest PEG parser documentation (from training data, verified against existing grammar patterns in codebase)
- rayon::join semantics (from training data, verified against existing usage patterns in codebase)
- serde untagged enum pattern (from training data, standard Rust pattern)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in workspace (pest, rayon, serde, axum, utoipa, wasm_bindgen)
- Architecture: HIGH - integration points thoroughly mapped via codebase inspection
- Pitfalls: HIGH - identified from actual code structure and Phase 4 migration experience
- Code examples: HIGH - derived from existing codebase patterns

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable codebase, no external dependencies changing)
