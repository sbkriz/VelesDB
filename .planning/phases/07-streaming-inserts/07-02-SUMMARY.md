---
phase: 07-streaming-inserts
plan: 02
subsystem: database
tags: [streaming, delta-buffer, hnsw, search-merge, simd, brute-force]

requires:
  - phase: 07-streaming-inserts/01
    provides: StreamIngester, DeltaBuffer stub, drain loop, Collection wiring

provides:
  - DeltaBuffer with brute-force SIMD search and metric-aware sorting
  - merge_with_delta free function for dedup-and-merge of HNSW + delta results
  - Delta merge wired into all 6+ Collection search paths
  - Drain loop pushes to delta buffer when active (rebuild in progress)
  - Integration tests proving zero data loss during simulated HNSW rebuilds

affects: [07-streaming-inserts/03, auto-reindex, velesql-executor]

tech-stack:
  added: []
  patterns:
    - "cfg-gated merge_delta helper on Collection for persistence/non-persistence split"
    - "DeltaBuffer brute-force scan using DistanceMetric::calculate + sort_results"
    - "Delta-wins dedup strategy in merge_with_delta"

key-files:
  created: []
  modified:
    - crates/velesdb-core/src/collection/streaming/delta.rs
    - crates/velesdb-core/src/collection/streaming/mod.rs
    - crates/velesdb-core/src/collection/streaming/ingester.rs
    - crates/velesdb-core/src/collection/search/vector.rs
    - crates/velesdb-core/src/collection/search/batch.rs
    - crates/velesdb-core/src/collection/core/graph_api.rs
    - crates/velesdb-core/src/collection/types.rs

key-decisions:
  - "merge_delta as pub(crate) method on Collection with cfg(persistence) gating"
  - "Delta merge at search_ids_with_adc_if_pq level covers search/search_ids/search_with_filter"
  - "Separate merge_delta calls for search_with_ef (uses search_with_quality) and batch/graph paths"
  - "Delta-wins dedup: on ID conflict, delta score replaces HNSW score"
  - "delta_buffer visibility widened from pub(super) to pub(crate) for cross-module access"

patterns-established:
  - "cfg-split pattern: two fn merge_delta impls gated on feature=persistence vs not"
  - "Drain loop delta routing: snapshot vectors before upsert, extend delta after success"

requirements-completed: [STREAM-03, STREAM-04]

duration: 18min
completed: 2026-03-07
---

# Phase 07 Plan 02: DeltaBuffer Search Merge Summary

**DeltaBuffer brute-force SIMD search with metric-aware merge across all Collection search paths for zero data loss during HNSW rebuilds**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-07T14:01:48Z
- **Completed:** 2026-03-07T14:20:08Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Full DeltaBuffer implementation: activate/deactivate_and_drain, push/extend, brute-force search with SIMD distance calculation, metric-aware sorting
- merge_with_delta: deduplicates by ID (delta wins), sorts by metric ordering, truncates to k
- Delta merge wired into search, search_with_ef, search_ids, search_with_filter, search_batch_with_filters, search_batch_parallel, and search_by_embedding (graph)
- Drain loop routes vectors to delta buffer when active (after successful upsert to storage)
- 20 tests including integration tests proving immediate searchability and zero data loss during simulated rebuilds

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement DeltaBuffer search and merge logic** - `6b1e3966` (feat)
2. **Task 2: Wire delta merge into all Collection search paths** - `7c6b6149` (feat)

## Files Created/Modified
- `crates/velesdb-core/src/collection/streaming/delta.rs` - Full DeltaBuffer: activate, deactivate_and_drain, push, extend, search, len, merge_with_delta
- `crates/velesdb-core/src/collection/streaming/mod.rs` - Export merge_with_delta
- `crates/velesdb-core/src/collection/streaming/ingester.rs` - Drain loop delta routing + integration tests
- `crates/velesdb-core/src/collection/search/vector.rs` - merge_delta helper, wired into search_ids_with_adc_if_pq and search_with_ef
- `crates/velesdb-core/src/collection/search/batch.rs` - Per-query delta merge in batch search methods
- `crates/velesdb-core/src/collection/core/graph_api.rs` - Delta merge in search_by_embedding
- `crates/velesdb-core/src/collection/types.rs` - delta_buffer visibility pub(crate), removed dead_code allow

## Decisions Made
- merge_delta as pub(crate) on Collection with two cfg-gated implementations (persistence: calls merge_with_delta; non-persistence: returns results unchanged)
- Delta merge added at search_ids_with_adc_if_pq level to cover search, search_ids, and search_with_filter in one place
- Separate merge_delta calls needed for search_with_ef (different HNSW path) and batch/graph methods (direct HNSW access)
- Delta-wins dedup strategy: when same ID appears in both HNSW and delta results, delta score replaces HNSW score (delta is more recent data)
- delta_buffer field visibility widened from pub(super) to pub(crate) for access from search and core modules

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Delta buffer search merge complete, all search paths covered
- Ready for Plan 03: auto-reindex integration and progressive merge
- The deactivate_and_drain API is ready for the rebuild coordinator to use

---
*Phase: 07-streaming-inserts*
*Completed: 2026-03-07*
