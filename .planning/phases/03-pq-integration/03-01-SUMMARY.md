---
phase: 03-pq-integration
plan: 01
subsystem: database
tags: [serde, quantization, config, error-handling, backward-compat]

requires:
  - phase: 02-pq-core-engine
    provides: ProductQuantizer, RaBitQ, PQ SIMD kernels
provides:
  - QuantizationType enum with None/SQ8/Binary/PQ/RaBitQ variants
  - Backward-compatible QuantizationConfig deserialization (old default_type string + new mode object)
  - TrainingFailed error variant (VELES-029) for TRAIN executor
affects: [03-pq-integration/plan-02, 03-pq-integration/plan-03]

tech-stack:
  added: []
  patterns: [serde custom deserializer for backward compat, tagged enum with field defaults]

key-files:
  created: []
  modified:
    - crates/velesdb-core/src/config.rs
    - crates/velesdb-core/src/error.rs
    - crates/velesdb-core/src/lib.rs

key-decisions:
  - "Custom Deserialize impl on QuantizationConfig for dual-format support (old string vs new tagged object)"
  - "QuantizationType (not QuantizationMode) to avoid collision with velesql/ast/with_clause.rs"
  - "allow(clippy::unnecessary_wraps) on default_oversampling since serde default requires matching field type"
  - "TrainingFailed is recoverable (user can adjust params and retry)"

patterns-established:
  - "Serde backward compat: use intermediate raw struct with Option fields, map old format to new enum"
  - "PQ param defaults: k=256, oversampling=Some(4), opq_enabled=false"

requirements-completed: [PQ-06]

duration: 19min
completed: 2026-03-06
---

# Phase 03 Plan 01: Config & Error Foundation Summary

**QuantizationType tagged enum with backward-compatible serde deserializer and TrainingFailed error variant (VELES-029)**

## Performance

- **Duration:** 19 min
- **Started:** 2026-03-06T14:35:52Z
- **Completed:** 2026-03-06T14:54:59Z
- **Tasks:** 2
- **Files modified:** 17

## Accomplishments
- QuantizationType enum with 5 variants (None, SQ8, Binary, PQ{m,k,opq_enabled,oversampling}, RaBitQ) replaces stringly-typed default_type
- Custom deserializer handles both old config.json format (default_type: "sq8") and new format (mode: {type: "pq", m: 8})
- TrainingFailed error variant with VELES-029 code ready for TRAIN executor
- 14 new tests (11 config + 3 error) covering all backward compat and new format scenarios

## Task Commits

Both tasks committed atomically (pre-commit hook staged all changes together):

1. **Task 1+2: QuantizationType enum + TrainingFailed error** - `0b2122dd` (feat)

**Plan metadata:** [pending]

## Files Created/Modified
- `crates/velesdb-core/src/config.rs` - QuantizationType enum, redesigned QuantizationConfig with custom Deserialize
- `crates/velesdb-core/src/error.rs` - TrainingFailed(String) variant with VELES-029
- `crates/velesdb-core/src/config_tests.rs` - 11 new tests for QuantizationType backward compat + roundtrip
- `crates/velesdb-core/src/error_tests.rs` - 3 new tests for TrainingFailed code, display, recoverability
- `crates/velesdb-core/src/lib.rs` - Export QuantizationType from crate root
- `crates/velesdb-core/src/velesql/ast/train.rs` - TrainStatement AST node (research artifact)
- `crates/velesdb-core/src/velesql/parser/train.rs` - TRAIN parser (research artifact)
- `crates/velesdb-core/src/velesql/parser/select/clause_compound.rs` - Fix missing train field

## Decisions Made
- Custom Deserialize impl on QuantizationConfig for dual-format support (old default_type string vs new tagged mode object) rather than serde untagged enum (more explicit error messages)
- Named enum QuantizationType (not QuantizationMode) to avoid collision with existing velesql/ast/with_clause.rs::QuantizationMode
- TrainingFailed classified as recoverable error (user can add more vectors or change PQ params)
- PQ defaults: k=256 (standard codebook size), oversampling=Some(4) (4x training vectors), opq_enabled=false

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Included uncommitted Phase 03 research artifacts**
- **Found during:** Task 1 (compilation)
- **Issue:** Phase 03 research left uncommitted changes to velesql (TRAIN AST, grammar, parser, validation tests) that added a `train` field to Query struct, breaking compilation
- **Fix:** Included all research artifacts in the commit (train.rs, grammar.pest, parser/train.rs, validation_tests.rs updates, compound clause fix)
- **Files modified:** 10 velesql files
- **Verification:** cargo check, cargo test, cargo clippy all pass
- **Committed in:** 0b2122dd

**2. [Rule 1 - Bug] Fixed missing train field in compound clause parser**
- **Found during:** Task 1 (compilation)
- **Issue:** Query struct gained a `train` field from research but clause_compound.rs constructor didn't include it
- **Fix:** Added `train: None` to the Query initializer
- **Files modified:** crates/velesdb-core/src/velesql/parser/select/clause_compound.rs
- **Committed in:** 0b2122dd

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for compilation. No scope creep -- research artifacts were already written, just uncommitted.

## Issues Encountered
- Pre-commit hook runs full test suite (~2700 tests) requiring 600s timeout for commits
- Linter/formatter auto-applies changes between Read and Edit calls, requiring re-reads

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- QuantizationType enum ready for TRAIN executor (Plan 02) to use PQ variant params
- TrainingFailed error ready for TRAIN executor error paths
- All existing tests pass without modification (2734 passed, 0 failed)

---
*Phase: 03-pq-integration*
*Completed: 2026-03-06*
