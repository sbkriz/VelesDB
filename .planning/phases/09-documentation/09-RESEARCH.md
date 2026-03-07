# Phase 9: Documentation - Research

**Researched:** 2026-03-07
**Domain:** Documentation (README, rustdoc, OpenAPI, migration guide, benchmarks, changelog)
**Confidence:** HIGH

## Summary

Phase 9 is a documentation-only phase with no code logic changes. The work spans six distinct deliverables: README v1.5 update (DOC-01), rustdoc completeness (DOC-02), OpenAPI spec generation (DOC-03), migration guide (DOC-04), benchmarks update (DOC-05), and changelog completion (DOC-06).

Key findings from the codebase investigation: (1) `#![warn(missing_docs)]` is already active in velesdb-core with **zero missing documentation warnings** -- DOC-02 reduces to fixing 10 broken intra-doc links; (2) the OpenAPI spec currently omits graph endpoints and lacks utoipa path annotations on some handlers -- the `paths()` macro in `lib.rs` lists 21 endpoints but graph handlers (add_edge, traverse_graph, etc.) are absent; (3) the VelesQL spec (`docs/VELESQL_SPEC.md`) and guides (`QUANTIZATION.md`, `SEARCH_MODES.md`) contain no mention of sparse vectors, SPARSE_NEAR, PQ training, or TRAIN QUANTIZER; (4) benchmark infrastructure exists (Criterion suites for `pq_recall` and `sparse`) but `docs/BENCHMARKS.md` is stuck at v1.4.1 numbers.

**Primary recommendation:** Structure work as 4-5 plans: (1) README + Changelog, (2) Rustdoc + broken links, (3) OpenAPI spec generation, (4) Migration guide + guides updates, (5) Benchmarks. Plans 1-4 are text-only and can be fast; plan 5 requires running Criterion benchmarks.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- README: Keep existing structure (1525 lines), reorganize to highlight v1.5 features, add "What's New in v1.5" section with feature cards + code examples covering ALL functional/performance improvements between v1.4.5 and v1.5
- README badges: Add PQ recall@10 + sparse search latency badges alongside existing dense HNSW + SIMD badges
- README header: Change from "v1.4.1 Released" to "v1.5.0 Released"
- Benchmarks: 3 priority benchmarks: (1) PQ recall@k, (2) Sparse search latency, (3) Hybrid search recall+latency. Streaming throughput EXCLUDED
- Benchmarks: Dense search baseline preserved as section 1 (existing)
- Benchmarks: Generate numbers via Criterion suites (pq_recall, simd_benchmark) + new sparse/hybrid benches
- Benchmarks: Format as markdown tables, "Test Environment" section at top from machine-config.json
- Rustdoc: velesdb-core scope only, `#![warn(missing_docs)]` in lib.rs (already present), all public types/functions must have doc comments
- OpenAPI: Regenerate from utoipa annotations, add missing annotations on sparse/streaming endpoints, store as docs/openapi.json + docs/openapi.yaml
- Migration guide: Must cover bincode->postcard wire format, QuantizationConfig PQ variant, VelesQL SPARSE_NEAR syntax

### Claude's Discretion
- Migration guide: format and depth (Claude decides)
- Changelog: complete the [Unreleased] section of existing CHANGELOG.md (2663 lines, Keep a Changelog format)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DOC-01 | README v1.5 -- recalculated metrics, v1.5 features, updated examples | README structure analyzed (1525 lines), current badges identified, v1.4.1 header located at line 15, feature sections mapped |
| DOC-02 | rustdoc complete API publique velesdb-core | `#![warn(missing_docs)]` already active, 0 missing doc warnings, 10 broken intra-doc link warnings identified |
| DOC-03 | OpenAPI spec v1.5 -- sparse + streaming endpoints | utoipa 5.x + utoipa-swagger-ui 9.x identified, 21 endpoints in paths() macro, graph endpoints missing, sparse annotations present in types but handler annotations need verification |
| DOC-04 | Migration guide v1.4 -> v1.5 | No existing migration guide, breaking changes cataloged: postcard wire format, QuantizationConfig PQ variant, SPARSE_NEAR VelesQL, bincode removal |
| DOC-05 | BENCHMARKS.md v1.5 -- real PQ recall, sparse latency | Criterion suites exist (pq_recall_benchmark.rs, sparse_benchmark.rs), machine-config.json present, current BENCHMARKS.md is v1.4.1 |
| DOC-06 | CHANGELOG.md v1.5 -- all features, fixes, breaking changes | CHANGELOG.md (2663 lines), [Unreleased] section exists with review fixes, needs all v1.5 feature entries |

</phase_requirements>

## Standard Stack

### Core
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| utoipa | 5.x | OpenAPI spec generation from Rust annotations | Already in use, derives OpenAPI 3.0 from handler annotations |
| utoipa-swagger-ui | 9.x | Swagger UI serving (optional feature) | Already configured with `swagger-ui` feature flag |
| Criterion | (workspace) | Benchmark framework | Already used for all benches in `crates/velesdb-core/benches/` |
| cargo doc | rustc bundled | Rustdoc generation | Standard Rust toolchain |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `serde_json` / `serde_yaml` | Serialize OpenAPI spec to JSON/YAML | For generating docs/openapi.json and docs/openapi.yaml |
| `machine-config.json` | Hardware reference for benchmarks | Include in benchmark "Test Environment" section |

### Alternatives Considered
None -- this phase uses only existing tools.

## Architecture Patterns

### Documentation File Layout
```
README.md                          # v1.5 update (DOC-01)
CHANGELOG.md                       # Complete [Unreleased] section (DOC-06)
docs/
├── BENCHMARKS.md                  # v1.5 benchmark results (DOC-05)
├── VELESQL_SPEC.md                # Update with SPARSE_NEAR, TRAIN QUANTIZER
├── openapi.json                   # NEW: generated OpenAPI spec (DOC-03)
├── openapi.yaml                   # NEW: generated OpenAPI spec (DOC-03)
├── MIGRATION_v1.4_to_v1.5.md      # NEW: migration guide (DOC-04)
├── guides/
│   ├── QUANTIZATION.md            # Update with PQ section
│   └── SEARCH_MODES.md            # Update with sparse search mode
crates/velesdb-core/src/lib.rs     # Already has #![warn(missing_docs)] (DOC-02)
crates/velesdb-server/src/lib.rs   # OpenAPI paths() macro update (DOC-03)
```

### Pattern 1: OpenAPI Spec Generation
**What:** Use utoipa derive macros + a test/script to dump the generated spec to disk
**When to use:** DOC-03 -- generating docs/openapi.json and docs/openapi.yaml

The existing test `test_openapi_spec_generation` in `crates/velesdb-server/src/lib.rs` already generates the spec in-memory. To persist it, either:
- Add a `#[test]` that writes to `docs/openapi.json` and `docs/openapi.yaml`
- Or create a small binary/script that does it

**Recommended approach:** Add a test that generates and writes the files, plus a CI check that verifies the committed spec matches the generated one (round-trip validation).

```rust
// In crates/velesdb-server/src/lib.rs tests
#[test]
fn generate_openapi_spec_files() {
    let openapi = ApiDoc::openapi();
    let json = openapi.to_pretty_json().unwrap();
    std::fs::write("../../docs/openapi.json", &json).unwrap();
    // For YAML: use serde_yaml or utoipa's to_yaml if available
}
```

### Pattern 2: Changelog -- Keep a Changelog Format
**What:** Follow the existing CHANGELOG.md format strictly
**Format:** `## [Unreleased]` with subsections: `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, `### Security`

### Pattern 3: Benchmark Results Format
**What:** Markdown tables with environment header
**Format from CONTEXT.md decisions:**
```markdown
## Test Environment
- **CPU**: Intel Core i9-14900KF
- **RAM**: 64GB DDR5
- **OS**: Windows 11
- **Rust**: 1.92.0

## PQ Recall@k
| Config | Recall@10 | Recall@50 | Recall@100 | Latency |
|--------|-----------|-----------|------------|---------|
| Full precision | ... | ... | ... | ... |
| PQ m=8 k=256 | ... | ... | ... | ... |
```

### Anti-Patterns to Avoid
- **Placeholder benchmark numbers:** Never write fake numbers. If benchmarks cannot be run, mark as "TBD -- run `cargo bench ...` to populate"
- **Stale version references:** Grep for "v1.4" and "1.4" in all modified files to ensure no stale references remain
- **OpenAPI spec drift:** The committed spec file must be re-generable from code. A CI test should enforce this

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenAPI spec | Manual JSON/YAML | utoipa derive macros + `ApiDoc::openapi().to_pretty_json()` | Keeps spec in sync with handler signatures |
| YAML conversion | Custom serializer | `serde_yaml::to_string` or utoipa's built-in YAML support | Standard, correct output |
| Benchmark data | Manual measurement | Criterion framework (existing benches) | Reproducible, statistical rigor |

## Common Pitfalls

### Pitfall 1: Stale v1.4 References
**What goes wrong:** README or docs still mention "v1.4.1" or show old benchmark numbers
**Why it happens:** Large files (1525-line README) make it easy to miss scattered version references
**How to avoid:** After all edits, grep the entire repo for `v1.4`, `1.4.1`, `1.4.5` to catch stragglers
**Warning signs:** Badge text, header announcements, changelog links still pointing to old version

### Pitfall 2: OpenAPI Spec Missing Endpoints
**What goes wrong:** New v1.5 endpoints (sparse search, stream insert) not in the generated spec
**Why it happens:** utoipa requires explicit `#[utoipa::path(...)]` annotations AND listing in `paths()` macro
**How to avoid:** Verify every route in `build_router()` (main.rs) has a corresponding entry in `paths()` (lib.rs)
**Warning signs:** Graph endpoints (`add_edge`, `traverse_graph`, `get_edges`, `get_node_degree`, `stream_traverse`) are already missing from the current spec

### Pitfall 3: Broken Intra-Doc Links
**What goes wrong:** `cargo doc --no-deps` warns about unresolved links
**Why it happens:** References to private items, cross-module paths, or renamed types
**How to avoid:** Fix all 10 existing warnings (see list below)
**Current warnings:**
1. `get_stats` links to private `STATS_TTL`
2. `streaming/mod.rs` links to `Collection::upsert` (wrong path)
3. `delta.rs` links to private `DeltaBuffer::state`
4. `vector_collection.rs` links to `BackpressureError::BufferFull` (3 occurrences)
5. `compact` links to private `compact_with_prefix`
6. `load_from_disk` links to private `load_from_disk_with_prefix`
7. Unresolved `:REL` link
8. Unresolved `choose_strategy_with_cbo`

### Pitfall 4: Benchmark Reproducibility
**What goes wrong:** Benchmark numbers vary between runs, making them unreliable
**Why it happens:** CPU throttling, background processes, cold vs warm cache
**How to avoid:** Use `machine-config.json` to document test environment; run benchmarks multiple times; report p50/p95/p99 from Criterion statistical output
**Warning signs:** Numbers that seem too good or inconsistent with prior baselines

### Pitfall 5: Migration Guide Incomplete Breaking Changes
**What goes wrong:** Users upgrade and hit undocumented breaking changes
**Why it happens:** Breaking changes scattered across 8 phases of commits
**How to avoid:** Cross-reference STATE.md decisions list for all breaking changes:
- bincode -> postcard (Phase 1, 01-01)
- QuantizationConfig PQ variant (Phase 3, 03-01)
- QuantizationType enum (not QuantizationMode)
- VelesQL SPARSE_NEAR syntax (Phase 5, 05-02)
- Point struct now has `sparse_vector` field (Phase 4, 04-01)
- Named sparse vectors in Point (Phase 5)
- Custom Deserialize on Point for backward compat (Phase 5)

## Code Examples

### Generating OpenAPI Spec to Files
```rust
// Source: existing pattern in crates/velesdb-server/src/lib.rs
use utoipa::OpenApi;

let openapi = ApiDoc::openapi();
let json = openapi.to_pretty_json().expect("serialize JSON");
let yaml = openapi.to_yaml().expect("serialize YAML");
std::fs::write("docs/openapi.json", &json).unwrap();
std::fs::write("docs/openapi.yaml", &yaml).unwrap();
```

### Adding utoipa Path Annotation to a Handler
```rust
// Source: existing pattern in crates/velesdb-server/src/handlers/search.rs
#[utoipa::path(
    post,
    path = "/collections/{name}/search",
    tag = "search",
    params(("name" = String, Path, description = "Collection name")),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse),
    )
)]
pub async fn search(...) { ... }
```

### Fixing Broken Intra-Doc Links
```rust
// Instead of linking to private items:
/// Results are cached for 30 seconds to avoid re-scanning.
// Or use the full path if it needs to be public:
/// Results are cached for [`crate::collection::core::statistics::STATS_TTL`].
```

### VelesQL SPARSE_NEAR Example for Docs
```sql
-- Sparse search (SPLADE/BM42 vectors)
SELECT * FROM docs WHERE vector SPARSE_NEAR $sv LIMIT 10

-- Hybrid dense + sparse with RRF fusion
SELECT * FROM docs
WHERE vector NEAR $dense AND vector SPARSE_NEAR $sparse
FUSE BY RRF(k=60)
LIMIT 10

-- Train PQ quantizer
TRAIN QUANTIZER ON my_collection WITH (m=8, k=256)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SQ8 + Binary only | SQ8 + Binary + PQ + RaBitQ | Phase 2-3 | QUANTIZATION.md must cover PQ |
| Dense-only search | Dense + Sparse + Hybrid | Phase 4-5 | SEARCH_MODES.md must cover sparse |
| bincode serialization | postcard serialization | Phase 1 | Migration guide breaking change |
| No plan cache | CompiledPlanCache two-level | Phase 6 | Feature worth mentioning in README |
| Batch-only inserts | Streaming inserts | Phase 7 | New feature for README + changelog |

## Existing Documentation State

| File | Lines | Version | Needs Update |
|------|-------|---------|--------------|
| README.md | 1525 | v1.4.1 | YES -- header, badges, features, examples |
| CHANGELOG.md | 2663 | [Unreleased] partial | YES -- complete v1.5 entries |
| docs/BENCHMARKS.md | 327 | v1.4.1 | YES -- add PQ/sparse/hybrid results |
| docs/VELESQL_SPEC.md | 938 | v1.4 | YES -- add SPARSE_NEAR, TRAIN QUANTIZER |
| docs/guides/QUANTIZATION.md | 199 | v1.4 (SQ8/Binary only) | YES -- add PQ section |
| docs/guides/SEARCH_MODES.md | 518 | v1.4 (dense only) | YES -- add sparse mode |
| docs/openapi.json | N/A | Does not exist | CREATE |
| docs/openapi.yaml | N/A | Does not exist | CREATE |
| Migration guide | N/A | Does not exist | CREATE |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (built-in) + Criterion for benchmarks |
| Config file | existing workspace Cargo.toml |
| Quick run command | `cargo test -p velesdb-server test_openapi -- --test-threads=1` |
| Full suite command | `cargo test --workspace --features persistence,gpu,update-check --exclude velesdb-python -- --test-threads=1` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOC-01 | README shows v1.5 content | manual | Visual inspection + grep for "v1.4" | N/A |
| DOC-02 | Zero cargo doc warnings | unit | `cargo doc -p velesdb-core --no-deps 2>&1 \| grep -c warning` | Existing (10 warnings to fix) |
| DOC-03 | OpenAPI spec round-trip | unit | `cargo test -p velesdb-server test_openapi -- --test-threads=1` | Existing tests in lib.rs |
| DOC-04 | Migration guide covers breaking changes | manual | Grep for key terms in guide | N/A |
| DOC-05 | Benchmark numbers are real | smoke | `cargo bench -p velesdb-core --bench pq_recall_benchmark -- --noplot` | Existing bench files |
| DOC-06 | Changelog complete | manual | Visual inspection | N/A |

### Sampling Rate
- **Per task commit:** `cargo doc -p velesdb-core --no-deps 2>&1 | grep warning` (zero expected)
- **Per wave merge:** `cargo test -p velesdb-server -- --test-threads=1` (OpenAPI tests pass)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] OpenAPI round-trip CI test (generate spec, compare with committed file) -- new test needed
- [ ] `docs/openapi.json` and `docs/openapi.yaml` -- files to be created
- [ ] `docs/MIGRATION_v1.4_to_v1.5.md` -- new file

## Open Questions

1. **Benchmark numbers availability**
   - What we know: Criterion suites exist for PQ recall and sparse search. Machine config documented.
   - What's unclear: Whether the planner should actually RUN benchmarks during phase execution or document how to run them. Running full benchmarks can take 10+ minutes.
   - Recommendation: Run benchmarks and capture real numbers. Use `--noplot` flag. If time-constrained, run with smaller datasets and note the configuration.

2. **OpenAPI YAML generation**
   - What we know: utoipa 5.x has `to_pretty_json()`. YAML support may require `serde_yaml`.
   - What's unclear: Whether utoipa 5.x has native `to_yaml()` method.
   - Recommendation: Check at implementation time. If no native YAML, serialize via `serde_yaml::to_string(&openapi)` since the OpenApi struct implements Serialize.

3. **Graph endpoints in OpenAPI**
   - What we know: Graph handlers (add_edge, traverse_graph, get_edges, get_node_degree, stream_traverse) are re-exported but NOT in the OpenAPI `paths()` macro.
   - What's unclear: Whether graph handlers have `#[utoipa::path]` annotations at all.
   - Recommendation: Add graph endpoints to OpenAPI spec as part of DOC-03 (they are public API).

## Sources

### Primary (HIGH confidence)
- `crates/velesdb-server/src/lib.rs` -- OpenAPI macro configuration, 21 listed paths, test suite
- `crates/velesdb-server/Cargo.toml` -- utoipa 5.x, utoipa-swagger-ui 9.x versions
- `crates/velesdb-core/src/lib.rs` -- `#![warn(missing_docs)]` confirmed at line 47
- `cargo doc -p velesdb-core --no-deps` -- 10 warnings (broken links only, zero missing docs)
- `crates/velesdb-core/benches/` -- 38 benchmark files including pq_recall_benchmark.rs, sparse_benchmark.rs
- `benchmarks/machine-config.json` -- i9-14900KF, 64GB DDR5, Rust 1.92.0

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` -- accumulated decisions listing all breaking changes across phases

### Tertiary (LOW confidence)
- utoipa `to_yaml()` availability -- needs verification at implementation time

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools already in use in the project
- Architecture: HIGH -- documentation patterns well-understood, files identified
- Pitfalls: HIGH -- specific warnings and gaps identified from actual cargo doc output

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable -- documentation tooling changes slowly)
