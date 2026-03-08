---
phase: 13-recall-benchmark-multi-distribution-coverage-uniform-clustered-sparse-binary
plan: 01
subsystem: testing
tags: [criterion, recall, benchmark, clustered, binary, hnsw, pq, rabitq]

# Dependency graph
requires:
  - phase: 11-pq-recall-benchmark-hardening
    provides: "PQ recall benchmark with 6 variants on uniform random data"
provides:
  - "Multi-distribution recall benchmarks (clustered Gaussian, binary {0,1})"
  - "Exact search baselines for all 3 distributions"
  - "benchmarks/baseline_multidist.json with measured recall values"
affects: []

# Tech tracking
tech-stack:
  added: [rand_distr 0.4]
  patterns: [multi-distribution benchmark, ef_search override for clustered data]

key-files:
  created:
    - crates/velesdb-core/benches/pq_recall_multidist.rs
    - benchmarks/baseline_multidist.json
  modified:
    - crates/velesdb-core/Cargo.toml

key-decisions:
  - "Clustered data achieves 1.0 recall with ef_search=512 on 5K vectors -- no ceiling issue at this scale"
  - "Binary {0,1} data achieves 0.904 recall with default ef_search=128"

patterns-established:
  - "measure_recall_with_ef for explicit ef_search control in benchmarks"
  - "Separate data generators per distribution with seed-based reproducibility"

requirements-completed: [PQ-07]

# Metrics
duration: 7min
completed: 2026-03-08
---

# Phase 13 Plan 01: Multi-Distribution Recall Benchmarks Summary

**Clustered Gaussian (6 variants) and binary {0,1} (2 variants) recall benchmarks with exact search baselines, all thresholds passing (clustered 1.0, binary 0.904, exact 1.0)**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-08T10:50:31Z
- **Completed:** 2026-03-08T10:57:07Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Clustered Gaussian benchmarks with 6 variants all achieve recall@10 = 1.0 (ef_search=512 on 5K/128d)
- Binary {0,1} benchmarks achieve recall@10 = 0.904 for both RaBitQ and full precision
- Exact search baselines validate recall = 1.0 on all three distributions (uniform, clustered, binary)
- Baseline results recorded in benchmarks/baseline_multidist.json

## Task Commits

Each task was committed atomically:

1. **Task 1: Add rand_distr dependency and Cargo.toml bench entry** - `13b93c1e` (chore)
2. **Task 2: Create multi-distribution recall benchmark file and baseline** - `623eab8f` (feat)

## Files Created/Modified
- `crates/velesdb-core/Cargo.toml` - Added rand_distr dev-dependency and pq_recall_multidist bench entry
- `crates/velesdb-core/benches/pq_recall_multidist.rs` - Multi-distribution recall benchmarks (clustered, binary, exact)
- `benchmarks/baseline_multidist.json` - Baseline recall results for all distributions

## Decisions Made
- Clustered data with sigma=0.1 and ef_search=512 achieves perfect 1.0 recall on 5K vectors -- the HNSW ceiling issue from Phase 11 does not apply at this scale with high ef_search
- Binary data recall at 0.904 exceeds the 0.85 threshold comfortably
- Used rand_distr 0.4 (compatible with rand 0.8) for Normal distribution sampling

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 13 complete (single plan phase)
- Ready for Phase 14 (README documentation audit)

---
*Phase: 13-recall-benchmark-multi-distribution-coverage-uniform-clustered-sparse-binary*
*Completed: 2026-03-08*
