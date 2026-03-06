# Deferred Items - Phase 01

## Pre-existing Issues (Out of Scope)

### ProductQuantizer::train() returns struct instead of Result
- **Discovered during:** 01-01 Task 1 verification
- **Impact:** 15 test compilation errors in `crates/velesdb-core/src/quantization/pq.rs`
- **Description:** `ProductQuantizer::train()` returns `ProductQuantizer` directly instead of `Result<ProductQuantizer, Error>`, causing all test code that calls `.unwrap()`, `.is_err()`, `.unwrap_err()` on the return value to fail compilation.
- **Already tracked:** STATE.md blockers mention this as QUAL-03/QUAL-04 targeted for Phase 1.
- **Action:** Do not fix in 01-01 (serialization migration). This is planned for a separate plan.
