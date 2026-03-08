---
phase: 17
slug: hybrid-query-test-demo-coverage
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 17 — Validation Strategy

## Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test + `cargo test` |
| Config file | none — test attributes on functions |
| Quick run command | `cargo test -p velesdb-core --features persistence hyb -- --test-threads=1` |
| Full suite command | `cargo test --workspace --features persistence,gpu,update-check --exclude velesdb-python -- --test-threads=1` |

## Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HYB-01 | VelesQL NEAR+scalar-filter query executes and asserts ranked identity | integration | `cargo test -p velesdb-core --features persistence test_hyb01 -- --test-threads=1` | Wave 0 |
| HYB-02 | Hybrid BM25+cosine fusion produces different ranking than pure vector on divergence corpus | integration | `cargo test -p velesdb-core --features persistence test_hyb02 -- --test-threads=1` | Wave 0 |
| HYB-03 | GraphCollection edges + MATCH traversal returns real results | integration | `cargo test -p velesdb-core --features persistence test_hyb03 -- --test-threads=1` | Wave 0 |
| HYB-04 | Python pseudocode files labeled; no runnable file uses only print() without explicit pseudocode labeling | manual | `grep -l "PSEUDOCODE" examples/python/*.py` — expect 3 files | n/a (modify existing) |
| HYB-05 | ecommerce_recommendation uses hybrid_search() not manual HashMap merge | integration (example) | `cargo test -p ecommerce-recommendation -- --test-threads=1` | Exists (example has `#[cfg(test)]`) |

## Wave 0 Gaps

The following file must be created before any verify commands can run:

- [ ] `crates/velesdb-core/tests/hybrid_credibility_tests.rs` — covers HYB-01, HYB-02, HYB-03
  - Imports needed: `velesdb_core::{Database, DistanceMetric, Point, GraphEdge, velesql::Parser}`, `serde_json::json`, `std::collections::HashMap`, `tempfile::TempDir`

HYB-04 and HYB-05 modify existing files; no new test infrastructure needed for those.

## Sampling Rate

- **Per task commit:** `cargo test -p velesdb-core --features persistence hyb -- --test-threads=1`
- **Per wave merge:** `cargo test --workspace --features persistence --exclude velesdb-python -- --test-threads=1`
- **Phase gate:** Full suite green before `/gsd:verify-work`

## Anti-Pattern Checks

These must NOT appear in the final test file:

| Anti-pattern | Why Blocked |
|---|---|
| `assert!(!results.is_empty())` as sole assertion | Zero ranking information; Phase 17 explicitly replaces this pattern |
| `assert!(result.is_ok())` as sole assertion | Parser-only; must be followed by execution and identity check |
| Score magnitude assertions on RRF results (e.g., `results[0].score > 0.9`) | RRF max is ~0.0083; assert rank order (`.point.id`) not score magnitude |

## nyquist_compliant Conditions

Set `nyquist_compliant: true` when all of the following are true:

- [ ] `hybrid_credibility_tests.rs` exists on disk
- [ ] Each test function (`test_hyb01`, `test_hyb02`, `test_hyb03`) has an `<automated>` verify command in its plan task
- [ ] `cargo test -p velesdb-core --features persistence hyb -- --test-threads=1` passes (all three tests green)

Set `wave_0_complete: true` when `hybrid_credibility_tests.rs` is created and compiles (even if tests are not yet passing).
