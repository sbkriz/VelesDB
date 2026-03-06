---
phase: 05-sparse-integration
plan: "02"
subsystem: velesql-sparse
tags: [velesql, parser, pest, sparse-vectors, conformance]
dependency_graph:
  requires: [04-01, 04-02, 04-03]
  provides: [sparse-grammar, sparse-ast, rsf-fusion-type]
  affects: [05-03, 05-04]
tech_stack:
  added: []
  patterns: [pest-grammar-extension, conformance-driven-parser-testing]
key_files:
  created:
    - crates/velesdb-core/src/velesql/parser/sparse_search_tests.rs
  modified:
    - crates/velesdb-core/src/velesql/grammar.pest
    - crates/velesdb-core/src/velesql/ast/condition.rs
    - crates/velesdb-core/src/velesql/ast/fusion.rs
    - crates/velesdb-core/src/velesql/ast/mod.rs
    - crates/velesdb-core/src/velesql/mod.rs
    - crates/velesdb-core/src/velesql/parser/conditions.rs
    - crates/velesdb-core/src/velesql/parser/mod.rs
    - crates/velesdb-core/src/velesql/parser/select/clause_limit_with.rs
    - crates/velesdb-core/src/velesql/explain.rs
    - crates/velesdb-core/src/collection/search/query/pushdown.rs
    - crates/velesdb-core/src/filter/conversion.rs
    - conformance/velesql_parser_cases.json
decisions:
  - "SPARSE_NEAR placed before vector_search in PEG primary_expr alternatives for longest-match-first ordering"
  - "SparseVectorExpr enum separates Literal(SparseVector) from Parameter(String) for type-safe query binding"
  - "Rsf added as FusionStrategyType variant with dense_weight/sparse_weight fields on FusionClause"
metrics:
  duration: "26 min"
  completed: "2026-03-06"
---

# Phase 05 Plan 02: VelesQL SPARSE_NEAR Grammar + Parser + AST + Conformance Summary

SPARSE_NEAR keyword added to VelesQL pest grammar with SparseVectorSearch AST type, Rsf fusion variant, and 6 conformance cases covering param/literal/USING/hybrid/negative patterns.

## Task Results

### Task 1: Add SPARSE_NEAR grammar rules and AST types
**Commit:** `329e861e`

Added pest grammar rules for sparse vector search: `sparse_vector_search`, `sparse_value`, `sparse_literal`, `sparse_entry`. Created `SparseVectorSearch` struct and `SparseVectorExpr` enum in `ast/condition.rs`. Added `Rsf` variant to `FusionStrategyType` with `dense_weight`/`sparse_weight` fields on `FusionClause`. Updated exhaustive match arms in explain.rs, pushdown.rs, and conversion.rs to handle new `Condition::SparseVectorSearch` variant.

### Task 2: Wire parser dispatch and add conformance cases
**Commit:** `b1f04bb7`

Wired `Rule::sparse_vector_search` dispatch in `conditions.rs` parser with `parse_sparse_vector_search` and `parse_sparse_value` methods. Added RSF fusion strategy parsing (`dense_w`, `sparse_w` keys) in `clause_limit_with.rs`. Created 8 parser unit tests in `sparse_search_tests.rs`. Added 6 conformance cases (P021-P026) to `conformance/velesql_parser_cases.json` covering: param, literal, USING, hybrid dense+sparse, missing value (negative), empty literal (negative).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Non-exhaustive match arms for new Condition variant**
- **Found during:** Task 1
- **Issue:** Adding `Condition::SparseVectorSearch` variant required handling in 3 exhaustive match expressions
- **Fix:** Added match arms in explain.rs (grouped with vector search), pushdown.rs (merged with GraphMatch per clippy), conversion.rs (identity filter)
- **Files modified:** explain.rs, pushdown.rs, conversion.rs
- **Commit:** 329e861e

**2. [Rule 3 - Blocking] Pre-existing clippy/test failures in unstaged files**
- **Found during:** Task 1 and Task 2 (pre-commit hook)
- **Issue:** Unstaged modifications from 05-01 work caused clippy errors (or_insert_with -> or_default in crud.rs) and a test assertion mismatch (database_tests.rs "required" vs "must be > 0")
- **Fix:** Fixed clippy in crud.rs, corrected test assertion in database_tests.rs
- **Files modified:** crud.rs, database_tests.rs
- **Commits:** 329e861e, b1f04bb7

**3. [Rule 3 - Blocking] Pre-commit hook auto-staging unstaged 05-01 files**
- **Found during:** Task 1 and Task 2
- **Issue:** Pre-commit hook runs `cargo fmt --all` and `cargo test` which touch unstaged files, causing them to be auto-staged
- **Impact:** Both task commits include 05-01 sparse migration changes (Point struct, collection types, persistence) alongside 05-02 grammar changes
- **Decision:** Accepted -- the 05-01 changes were ready and tested; separating them would require complex partial staging

## Verification

- All 8 sparse parser unit tests pass
- All 26 conformance cases pass (including 6 new P021-P026)
- Full workspace clippy clean (pedantic, -D warnings)
- Full workspace test suite passes (single-threaded)

## Self-Check: PASSED

- All 4 key files verified present on disk
- Commit 329e861e verified in git log
- Commit b1f04bb7 verified in git log
