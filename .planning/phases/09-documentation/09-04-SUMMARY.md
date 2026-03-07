---
phase: 09-documentation
plan: 04
subsystem: documentation
tags: [benchmarks, criterion, pq, sparse, hybrid, simd]

requires:
  - phase: 02-pq-core-engine
    provides: PQ quantization implementation benchmarked here
  - phase: 04-sparse-vector-engine
    provides: Sparse inverted index benchmarked here
  - phase: 05-sparse-integration
    provides: Hybrid search (RRF/RSF) referenced in hybrid section
provides:
  - Updated BENCHMARKS.md with v1.5 PQ, sparse, and hybrid search results
affects: [10-release]

tech-stack:
  added: []
  patterns: [criterion-benchmark-documentation]

key-files:
  created: []
  modified: [docs/BENCHMARKS.md]

key-decisions:
  - "Preserved v1.4.1 SIMD kernel numbers since SIMD layer unchanged in v1.5"
  - "Hybrid search section uses estimated latency from component benchmarks (no dedicated hybrid bench exists)"
  - "PQ recall numbers reported as-is from Criterion (30.6% recall@10 on 128D/5K expected for standard PQ)"

patterns-established:
  - "Benchmark doc structure: numbered sections, Test Environment from machine-config.json, reproducibility commands"

requirements-completed: [DOC-05]

duration: 17min
completed: 2026-03-07
---

# Phase 09 Plan 04: Benchmarks Summary

**Updated BENCHMARKS.md with real v1.5 Criterion results for PQ recall/latency, sparse search, and hybrid search estimation**

## Performance

- **Duration:** 17 min
- **Started:** 2026-03-07T17:17:08Z
- **Completed:** 2026-03-07T17:34:31Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Ran pq_recall_benchmark, pq_hnsw_benchmark, sparse_benchmark, and simd_benchmark with Criterion
- Added Test Environment section sourced from benchmarks/machine-config.json
- Added PQ section with recall@10 and latency tables comparing Full/SQ8/PQ storage modes
- Added Sparse Search section with insert and search latency from 10K corpus benchmark
- Added Hybrid Search section with estimated RRF fusion latency from component benchmarks
- Preserved and confirmed dense search SIMD baseline with fresh v1.5 cosine numbers
- Reorganized document into 10 numbered sections with clean markdown tables

## Task Commits

Each task was committed atomically:

1. **Task 1: Run Criterion benchmarks and capture v1.5 numbers** - `7f05d0f0` (docs)

## Files Created/Modified
- `docs/BENCHMARKS.md` - Updated with v1.5 benchmark results across all sections

## Decisions Made
- Preserved v1.4.1 SIMD kernel numbers since the SIMD layer was not modified in v1.5; confirmed cosine similarity with fresh run
- Hybrid search section uses estimated latency derived from individual dense + sparse component benchmarks since no dedicated hybrid benchmark exists
- PQ recall@10 of 30.6% reported as-is -- expected for standard PQ on low-dimensional (128D) synthetic data without OPQ
- SQ8 highlighted as best general-purpose mode (zero recall loss, 4x compression, identical latency)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BENCHMARKS.md complete with real v1.5 numbers
- All documentation plans in Phase 09 can proceed independently
- Ready for Phase 10 (Release) review

---
*Phase: 09-documentation*
*Completed: 2026-03-07*
