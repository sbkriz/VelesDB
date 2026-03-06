---
phase: 2
slug: pq-core-engine
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-06
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in + proptest + criterion |
| **Config file** | `crates/velesdb-core/Cargo.toml` (dev-dependencies) |
| **Quick run command** | `cargo test -p velesdb-core --features persistence -- --test-threads=1 pq` |
| **Full suite command** | `cargo test --workspace --features persistence,gpu,update-check --exclude velesdb-python -- --test-threads=1` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p velesdb-core --features persistence -- --test-threads=1 pq`
- **After every plan wave:** Run `cargo test --workspace --features persistence,gpu,update-check --exclude velesdb-python -- --test-threads=1`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Wave 0 — TDD Self-Bootstrap

All plans in this phase use `type: tdd` tasks with `tdd="true"`. Each TDD task creates its own test file as part of the RED phase (write failing test first, then implement). This means:

- **No separate Wave 0 plan is needed** -- test stubs are created inline by each task's RED step.
- **Nyquist compliance is satisfied** because every task produces both tests and implementation.
- **Test files are created before production code** within each task execution, not as a separate prerequisite.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | TDD Self-Bootstrap | Status |
|---------|------|------|-------------|-----------|-------------------|--------------------|--------|
| 02-01-01 | 01 | 1 | PQ-01 | integration | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_pq_codebook_quality` | Yes (tdd=true) | pending |
| 02-01-02 | 01 | 1 | PQ-01 | integration | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_pq_recall_property` | Yes (tdd=true) | pending |
| 02-01-03 | 01 | 1 | PQ-01 | unit | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_parallel_subspace_training` | Yes (tdd=true) | pending |
| 02-02-01 | 02 | 2 | PQ-02 | unit | `cargo test -p velesdb-core -- --test-threads=1 test_adc_simd_dispatch` | Yes (tdd=true) | pending |
| 02-02-02 | 02 | 2 | PQ-02 | unit | `cargo test -p velesdb-core -- --test-threads=1 test_lut_size_constraint` | Yes (tdd=true) | pending |
| 02-02-03 | 02 | 2 | PQ-02 | unit | `cargo test -p velesdb-core -- --test-threads=1 test_adc_batch_correctness` | Yes (tdd=true) | pending |
| 02-03-01 | 03 | 2 | PQ-ADV-01 | integration | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_rabitq_recall` | Yes (tdd=true) | pending |
| 02-03-02 | 03 | 2 | PQ-ADV-01 | unit | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_rabitq_training` | Yes (tdd=true) | pending |
| 02-04-01 | 04 | 3 | PQ-03 | integration | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_opq_recall_improvement` | Yes (tdd=true) | pending |
| 02-04-02 | 04 | 3 | QUANT-ADV-01 | unit | `cargo test -p velesdb-core --features persistence,gpu -- --test-threads=1 test_gpu_kmeans` | Yes (tdd=true) | pending |
| 02-02-04 | 02 | 2 | PQ-04 | unit | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_rescore_default_active` | Yes (tdd=true) | pending |
| 02-02-05 | 02 | 2 | PQ-04 | unit | `cargo test -p velesdb-core --features persistence -- --test-threads=1 test_rescore_configurable` | Yes (tdd=true) | pending |

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LUT fits in L1 cache | PQ-02 | Compile-time size validation, not runtime observable | Verify `m * k * 4 <= 8192` assertion in `precompute_lut` |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify (TDD self-bootstrap)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covered via TDD self-bootstrap (no separate Wave 0 needed)
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (TDD self-bootstrap satisfies Nyquist)
