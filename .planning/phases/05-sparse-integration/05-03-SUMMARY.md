---
phase: "05"
plan: "03"
subsystem: fusion-hybrid-search
tags: [sparse, fusion, rsf, rrf, hybrid-search, score-breakdown, filtered-search]
dependency_graph:
  requires: [05-01, 05-02, 04-01, 04-02]
  provides: [hybrid-dense-sparse-execution, rsf-fusion, filtered-sparse-search, score-breakdown-sparse]
  affects: [collection-search-pipeline, velesql-query-dispatch]
tech_stack:
  added: [rsf-fusion-strategy, sparse-search-filter]
  patterns: [rayon-parallel-branches, cfg-feature-gated-parallelism, min-max-normalization, oversampling-filtered-search]
key_files:
  created:
    - crates/velesdb-core/src/collection/search/query/hybrid_sparse.rs
    - crates/velesdb-core/src/collection/search/query/hybrid_sparse_tests.rs
  modified:
    - crates/velesdb-core/src/fusion/strategy.rs
    - crates/velesdb-core/src/fusion/strategy_tests.rs
    - crates/velesdb-core/src/index/sparse/search.rs
    - crates/velesdb-core/src/index/sparse/mod.rs
    - crates/velesdb-core/src/collection/search/query/score_fusion/mod.rs
    - crates/velesdb-core/src/collection/search/query/mod.rs
    - crates/velesdb-core/src/velesql/parser/sparse_search_tests.rs
decisions:
  - RSF uses min-max normalization per branch then weighted sum (dense_weight + sparse_weight = 1.0)
  - Filtered sparse search uses 4x oversampling first pass, 8x second pass if insufficient
  - Parallel branch execution via rayon::join gated on #[cfg(feature = "persistence")] with sequential fallback
  - Graceful degradation returns other branch results when one branch is empty
  - FusionStrategy passed by reference (&FusionStrategy) to avoid needless clone
metrics:
  duration: 24 min
  completed: "2026-03-06T22:16:00Z"
---

# Phase 05 Plan 03: Hybrid Dense+Sparse Fusion Execution Summary

RSF fusion strategy with min-max normalization, filtered sparse search with oversampling, and hybrid dense+sparse query pipeline with parallel branch execution via rayon.

## Task Completion

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | RSF fusion + filtered sparse + ScoreBreakdown | `7955f498` | Done |
| 2 | Hybrid execution pipeline + VelesQL dispatch | `46607e69` | Done |

## What Was Built

### Task 1: RSF Fusion, Filtered Sparse Search, ScoreBreakdown

**RSF Fusion Strategy** (`fusion/strategy.rs`):
- Added `RelativeScore { dense_weight, sparse_weight }` variant to `FusionStrategy`
- `relative_score()` constructor validates weights are non-negative and sum to 1.0
- `fuse_relative_score()` applies min-max normalization per branch then weighted sum
- 6 unit tests covering normalization, edge cases, validation

**Filtered Sparse Search** (`index/sparse/search.rs`):
- `sparse_search_filtered()` accepts `Option<&dyn Fn(u64) -> bool>` filter predicate
- Two-pass oversampling: 4x first attempt, 8x fallback if insufficient results after filtering
- 2 unit tests for filtered and empty-filter cases

**ScoreBreakdown Enrichment** (`collection/search/query/score_fusion/mod.rs`):
- Added `sparse_score`, `sparse_rank`, `dense_rank`, `fusion_method` fields
- `with_sparse()` builder method for sparse score enrichment
- Sparse score included in `components()` and `combine()` aggregation

### Task 2: Hybrid Dense+Sparse Query Execution Pipeline

**Hybrid Execution Module** (`collection/search/query/hybrid_sparse.rs`, 365 lines):
- `extract_sparse_vector_search()` recursively walks condition tree to find SparseVectorSearch
- `resolve_sparse_vector()` handles Literal and Parameter (structured JSON + shorthand map)
- `execute_sparse_search()` for sparse-only queries with optional metadata filter
- `execute_hybrid_search()` default RRF wrapper
- `execute_hybrid_search_with_strategy()` full hybrid with configurable FusionStrategy
- `execute_both_branches()` uses `rayon::join` (persistence feature) or sequential fallback
- `resolve_sparse_results()` and `resolve_fused_results()` hydrate IDs to full SearchResult

**VelesQL Query Dispatch** (`collection/search/query/mod.rs`):
- Sparse/hybrid dispatch block added before existing dense vector dispatch
- Extracts SparseVectorSearch from condition tree
- Builds FusionStrategy from `stmt.fusion_clause` (RSF with weights or RRF default)
- Routes to `execute_sparse_search` or `execute_hybrid_search_with_strategy`

**Integration Tests** (`hybrid_sparse_tests.rs`, 7 tests):
- `test_sparse_only_search` - sparse-only execution with score ordering
- `test_hybrid_dense_sparse_rrf` - hybrid with default RRF fusion
- `test_hybrid_dense_sparse_rsf` - hybrid with explicit RSF weights (0.6/0.4)
- `test_hybrid_empty_sparse_branch` - graceful degradation to dense-only
- `test_resolve_sparse_vector_structured` - JSON `{indices, values}` format
- `test_resolve_sparse_vector_shorthand` - JSON `{index: weight}` map format
- `test_resolve_sparse_vector_missing_param` - error on missing parameter

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Private method `search_ids_with_adc_if_pq`**
- **Found during:** Task 2
- **Issue:** Plan referenced private method from vector.rs module
- **Fix:** Used public `self.search_ids()` with `.unwrap_or_default()`
- **Files modified:** hybrid_sparse.rs

**2. [Rule 3 - Blocking] Non-existent `all_doc_ids()` on SparseInvertedIndex**
- **Found during:** Task 2
- **Issue:** Initial design pre-computed matching doc IDs, but no such API exists
- **Fix:** Pass filter callback directly to `sparse_search_filtered`, checking payload on-the-fly
- **Files modified:** hybrid_sparse.rs

**3. [Rule 1 - Bug] Pre-existing clippy wildcard match**
- **Found during:** Task 2 (clippy pass)
- **Issue:** `other =>` pattern in sparse_search_tests.rs missing type binding
- **Fix:** Changed to `other @ SparseVectorExpr::Parameter(_) =>`
- **Files modified:** velesql/parser/sparse_search_tests.rs

## Verification

- `cargo clippy --workspace --all-targets --features persistence,gpu,update-check --exclude velesdb-python -- -D warnings -D clippy::pedantic`: PASSED
- `cargo test --workspace --features persistence,gpu,update-check --exclude velesdb-python -- --test-threads=1`: PASSED (2793+ tests)
- All 7 new hybrid integration tests pass
- All 6 new RSF fusion tests pass
- All 2 new filtered sparse search tests pass

## Self-Check: PASSED

- All 7 key files: FOUND
- Commit 7955f498: FOUND
- Commit 46607e69: FOUND
