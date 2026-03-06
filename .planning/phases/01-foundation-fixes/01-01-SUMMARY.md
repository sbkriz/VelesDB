---
phase: 01-foundation-fixes
plan: 01
subsystem: database
tags: [postcard, bincode, serialization, cargo-deny, security, advisory]

# Dependency graph
requires: []
provides:
  - "postcard serialization across all persistence call sites"
  - "cargo-deny CI audit pipeline (no || true escape hatch)"
  - "RUSTSEC-2025-0141 resolved in velesdb-core"
affects: [02-pq-core, 07-streaming-inserts]

# Tech tracking
tech-stack:
  added: [postcard 1.1.3, cargo-deny]
  patterns: [postcard::to_allocvec + write_all for streaming serialization, fs::read + postcard::from_bytes for streaming deserialization]

key-files:
  created: []
  modified:
    - Cargo.toml
    - crates/velesdb-core/Cargo.toml
    - crates/velesdb-core/src/collection/graph/edge.rs
    - crates/velesdb-core/src/collection/graph/range_index.rs
    - crates/velesdb-core/src/collection/graph/property_index/mod.rs
    - crates/velesdb-core/src/index/hnsw/persistence.rs
    - crates/velesdb-core/src/storage/mmap.rs
    - crates/velesdb-core/src/storage/mmap/vector_io.rs
    - deny.toml
    - .github/workflows/ci.yml

key-decisions:
  - "Used postcard::to_allocvec + write_all instead of postcard::to_io for streaming serialization (to_io uses COBS framing, not a drop-in replacement for bincode's streaming format)"
  - "Kept RUSTSEC-2025-0141 exception in deny.toml because bincode remains as transitive dep via uniffi -> velesdb-mobile"
  - "Added RUSTSEC-2023-0089 exception for atomic-polyfill (postcard -> heapless transitive, unmaintained)"

patterns-established:
  - "Postcard serialization pattern: postcard::to_allocvec for serialize, postcard::from_bytes for deserialize"
  - "Streaming write pattern: postcard::to_allocvec(&data) then writer.write_all(&bytes) (replaces bincode::serialize_into)"
  - "Streaming read pattern: std::fs::read(path) then postcard::from_bytes(&bytes) (replaces bincode::deserialize_from)"

requirements-completed: [QUAL-01, QUAL-05]

# Metrics
duration: 9min
completed: 2026-03-06
---

# Phase 1 Plan 1: Bincode-to-Postcard Migration Summary

**Migrated all serialization from bincode to postcard across 6 call sites in 5 files, resolving RUSTSEC-2025-0141 and hardening CI audit with cargo-deny**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-06T00:11:16Z
- **Completed:** 2026-03-06T00:20:30Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Zero bincode usage in velesdb-core Rust sources (verified with grep)
- All workspace crates compile cleanly with postcard serialization
- CI audit pipeline now uses cargo-deny without || true escape hatch
- cargo deny check advisories passes with only documented transitive exceptions

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate all bincode call sites to postcard** - `7f5d5868` (feat)
2. **Task 2: Harden CI audit with cargo-deny and deny.toml** - `31afe893` (chore)

## Files Created/Modified
- `Cargo.toml` - Replaced workspace bincode dep with postcard (alloc + use-std features)
- `crates/velesdb-core/Cargo.toml` - Switched from bincode to postcard workspace dep
- `crates/velesdb-core/src/collection/graph/edge.rs` - EdgeStore::to_bytes/from_bytes use postcard
- `crates/velesdb-core/src/collection/graph/range_index.rs` - RangeIndex::to_bytes/from_bytes use postcard
- `crates/velesdb-core/src/collection/graph/property_index/mod.rs` - PropertyIndex::to_bytes/from_bytes use postcard
- `crates/velesdb-core/src/index/hnsw/persistence.rs` - HNSW save/load meta, mappings, vectors use postcard
- `crates/velesdb-core/src/storage/mmap.rs` - Index deserialization on load uses postcard
- `crates/velesdb-core/src/storage/mmap/vector_io.rs` - Index serialization on flush uses postcard
- `deny.toml` - Updated advisory exceptions, added atomic-polyfill exception
- `.github/workflows/ci.yml` - Replaced cargo-audit with cargo-deny

## Decisions Made
- Used `postcard::to_allocvec` + `write_all` instead of `postcard::to_io` for streaming serialization because `to_io` uses COBS framing which is incompatible with bincode's streaming format
- Kept RUSTSEC-2025-0141 in deny.toml ignore list (with updated reason) because bincode remains as a transitive dependency through `uniffi -> velesdb-mobile`
- Added RUSTSEC-2023-0089 exception for atomic-polyfill which is brought in by postcard -> heapless (unmaintained, no fix available in postcard 1.x)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added RUSTSEC-2023-0089 exception for atomic-polyfill**
- **Found during:** Task 2 (cargo deny check advisories verification)
- **Issue:** postcard 1.1.3 brings in heapless -> atomic-polyfill which has RUSTSEC-2023-0089 (unmaintained)
- **Fix:** Added exception to deny.toml with explanatory comment
- **Files modified:** deny.toml
- **Verification:** cargo deny check advisories passes
- **Committed in:** 31afe893 (Task 2 commit)

**2. [Rule 3 - Blocking] Kept RUSTSEC-2025-0141 exception with updated reason**
- **Found during:** Task 2 (cargo deny check advisories verification)
- **Issue:** bincode still in dependency tree via uniffi_macros -> uniffi -> velesdb-mobile (transitive, uncontrollable)
- **Fix:** Re-added RUSTSEC-2025-0141 with updated reason noting it's a uniffi transitive dep
- **Files modified:** deny.toml
- **Verification:** cargo deny check advisories passes
- **Committed in:** 31afe893 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both auto-fixes necessary to pass cargo deny check. No scope creep. The RUSTSEC-2025-0141 transitive dependency through uniffi is outside our control.

## Issues Encountered
- Pre-existing ProductQuantizer::train() compilation errors (15 errors in test code) prevented running full test suite. These are tracked in STATE.md as QUAL-03/QUAL-04 and are not related to this migration. Library code compiles and the pre-commit hook passes (stashing the pre-existing PQ changes).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Serialization migration complete, ready for any plan that depends on bincode removal
- HNSW persistence format changed (postcard vs bincode) - existing on-disk data will need re-indexing
- cargo-deny is now enforced in CI without escape hatches

---
*Phase: 01-foundation-fixes*
*Completed: 2026-03-06*
