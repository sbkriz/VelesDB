---
phase: 18-documentation-code-audit
plan: 01
subsystem: docs
tags: [python-sdk, velesql, readme, migration-guide]

requires:
  - phase: 17-hybrid-query-test-demo
    provides: "Correct API signatures validated by integration tests"
provides:
  - "Root README with correct Python SDK usage examples"
  - "Migration guide with correct velesdb.Database class and get_collection() accessor"
affects: [18-02, 18-03, 18-04]

tech-stack:
  added: []
  patterns: ["PSEUDOCODE markers for conceptual VelesQL blocks"]

key-files:
  created: []
  modified:
    - README.md
    - docs/MIGRATION_v1.4_to_v1.5.md

key-decisions:
  - "Business scenario VelesQL blocks with unsupported syntax (NOW(), INTERVAL, *1..3) marked as PSEUDOCODE per user decision rather than rewritten"
  - "get_all() documented as nonexistent with guidance to use get(ids) instead"

patterns-established:
  - "PSEUDOCODE marker: add '-- PSEUDOCODE: conceptual query, not executable VelesQL' at top of non-parseable VelesQL blocks"

requirements-completed: [DOC-01, DOC-04]

duration: 2min
completed: 2026-03-08
---

# Phase 18 Plan 01: Fix Root README and Migration Guide Summary

**Root README Python snippets corrected to use get_collection()/collection.search(vector=), migration guide updated to velesdb.Database, 3 business scenario VelesQL blocks marked as PSEUDOCODE**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-08T18:03:10Z
- **Completed:** 2026-03-08T18:04:38Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Fixed 3 instances of nonexistent `db.search()` and `db.stream_insert()` calls in root README to use `collection.search(vector=...)` and `collection.stream_insert()`
- Added `vector=` keyword argument to positional `search([...])` call in Quick Start section
- Marked 3 business scenario VelesQL blocks containing `NOW()`, `INTERVAL`, or `*1..3` as PSEUDOCODE
- Fixed migration guide to use `velesdb.Database` instead of nonexistent `VelesDB` class
- Replaced `db.collection()` with `db.get_collection()` in migration guide
- Documented that `get_all()` does not exist on Collection

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix root README Python snippets and business scenario VelesQL** - `8bf12807` (docs)
2. **Task 2: Fix migration guide API mismatches** - `0e0dacf0` (docs)

## Files Created/Modified
- `README.md` - Fixed Python SDK examples (db.search -> collection.search), added PSEUDOCODE markers to 3 VelesQL blocks
- `docs/MIGRATION_v1.4_to_v1.5.md` - Fixed VelesDB -> velesdb.Database, db.collection -> db.get_collection, documented get_all() nonexistence

## Decisions Made
- Business scenario VelesQL blocks with unsupported syntax marked as PSEUDOCODE rather than rewritten (per user decision from research phase)
- get_all() replaced with comment explaining to use get(ids) with specific IDs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- README and migration guide now have correct Python SDK signatures
- Ready for 18-02 (SEARCH_MODES.md and VelesQL spec FUSE BY fixes)

---
*Phase: 18-documentation-code-audit*
*Completed: 2026-03-08*
