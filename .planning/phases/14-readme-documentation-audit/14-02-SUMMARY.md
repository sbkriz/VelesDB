---
phase: 14-readme-documentation-audit
plan: 02
subsystem: docs
tags: [readme, api-reference, velesql, snippets, validation]

# Dependency graph
requires:
  - phase: 14-01
    provides: "README with corrected versions, metrics, badges, and structural cleanup"
provides:
  - "Complete v1.5 API reference with stream/insert, sparse search, and VelesQL v2.2.0 features"
  - "All README code snippets validated against real codebase handlers and grammar"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - README.md

key-decisions:
  - "StreamInsertRequest takes a single point (not array) per the actual handler struct -- corrected from plan's suggested payload shape"
  - "Subqueries replaced with direct payload field comparisons (e.g., product.price, account.total_amount_24h) rather than parameter bindings"
  - "Sparse search note added as blockquote below Search table rather than extra table row (clearer for hybrid mode explanation)"

patterns-established:
  - "All README SQL snippets use only runtime-supported VelesQL constructs (no subqueries)"

requirements-completed: [README-04, README-06]

# Metrics
duration: 2min
completed: 2026-03-08
---

# Phase 14 Plan 02: API Reference Gaps and Snippet Validation Summary

**Complete v1.5 API reference with stream/insert, SPARSE_NEAR, TRAIN QUANTIZER, FUSE BY; all 5 broken subqueries replaced with working payload field comparisons**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-08T12:10:19Z
- **Completed:** 2026-03-08T12:12:43Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added stream/insert endpoint to Points API table and streaming insert request/response example
- Added sparse search request/response example and hybrid search note to Search table
- Added SPARSE_NEAR, TRAIN QUANTIZER, FUSE BY to VelesQL v2.2.0 features list
- Replaced all 5 broken subquery patterns in business scenario SQL with direct payload field comparisons
- Validated all curl endpoint paths against server routes in main.rs

## Task Commits

Each task was committed atomically:

1. **Task 1: Add v1.5 API reference content** - `2f4c9667` (feat)
2. **Task 2: Validate and fix all code snippets** - `f373ab17` (fix)

## Files Created/Modified
- `README.md` - Added v1.5 API reference content (stream/insert, sparse search, VelesQL features) and fixed all broken subquery patterns in business scenario SQL

## Decisions Made
- StreamInsertRequest payload corrected to single point format (id, vector, payload) matching the actual `StreamInsertRequest` struct, rather than the array format suggested in the plan
- Subqueries replaced with contextually appropriate payload field names (product.price, account.total_amount_24h, treatment.success_rate, user.preferred_topic, author.citation_count)
- Sparse search documentation added as a blockquote note below the Search table explaining that sparse and hybrid search use the same `/collections/{name}/search` endpoint with different fields

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected streaming insert payload shape**
- **Found during:** Task 1 (Add v1.5 API reference content)
- **Issue:** Plan suggested `{"points": [...]}` array format for stream/insert endpoint, but actual `StreamInsertRequest` struct accepts a single point `{"id": ..., "vector": [...], "payload": {...}}`
- **Fix:** Used correct single-point payload matching the handler struct definition
- **Files modified:** README.md
- **Verification:** Compared against `crates/velesdb-server/src/types.rs` StreamInsertRequest struct
- **Committed in:** 2f4c9667 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential correctness fix for documentation accuracy. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 14 (README Documentation Audit) complete -- all plans executed
- README.md now has accurate v1.5 content with validated snippets

---
*Phase: 14-readme-documentation-audit*
*Completed: 2026-03-08*
