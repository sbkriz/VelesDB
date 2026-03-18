---
name: velesdb-release
description: >
  Orchestrate VelesDB version releases with SemVer compliance, full documentation audit,
  changelog generation, local CI validation, and multi-platform publishing (crates.io, PyPI, npm).
  Use when preparing a new release, bumping versions, or deploying to registries.
context: fork
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Agent, WebFetch
argument-hint: <version> [patch|minor|major]
---

# VelesDB Release Manager

## Invocation

```
/velesdb-release 1.6.0 minor
/velesdb-release 1.5.2 patch
/velesdb-release 2.0.0 major
```

- `$0` = target SemVer version (e.g., `1.6.0`)
- `$1` = release type: `patch`, `minor`, or `major` (default: `patch`)

## Critical Rules

1. **SemVer strict** — version must follow Semantic Versioning 2.0.0.
   - `major`: breaking API changes
   - `minor`: new features, backward compatible
   - `patch`: bug fixes only
2. **No AI attribution in docs** — NEVER mention Claude, Codex, OpenAI, Copilot,
   or any AI tool as co-author or contributor in ANY `.md` file, changelog, or
   documentation. Strip any existing `Co-Authored-By` AI references from docs.
3. **Date every document** — every `.md` file must have a `Last updated: YYYY-MM-DD`
   line. Use today's date for files modified during the release.
4. **Metrics must be fresh** — if a document contains benchmarks, metrics, or
   comparisons without dates or older than 30 days, re-run the measurement tools.
   If measurement is impossible (mobile, browser, GPU), ASK THE USER whether to
   keep or remove the section. Never silently keep stale metrics.
5. **Changelog = features + breaking changes only** — do not list refactors,
   internal fixes, CI changes, or dependency bumps unless they affect end users.
6. **Local CI before push** — run all checks locally to catch failures before
   they hit GitHub Actions. Skip platform-specific checks (mobile, WASM browser,
   GPU) that cannot run on the current machine, but note them.
7. **Every `.md` must be read** — scan ALL directories for markdown files. None
   may be skipped. Flag any that reference the old version.
8. **Examples and demos must compile** — all code snippets, demo projects, and
   example files must be updated and verified to compile/run.
9. **Publish to all registries** — crates.io, PyPI, npm, GitHub Releases. Verify
   each deployment succeeds.

## Workflow

Execute these phases in order. ASK the user for confirmation before proceeding
to the next phase. Report progress after each phase.

---

### Phase 1: Pre-flight Validation

```
PREVIOUS_TAG = !`git describe --tags --abbrev=0 2>/dev/null || echo "none"`
CURRENT_BRANCH = !`git rev-parse --abbrev-ref HEAD`
DIRTY_FILES = !`git status --porcelain`
```

**Checks:**
- [ ] On `main` branch (or user-approved release branch)
- [ ] Working directory clean (no uncommitted changes)
- [ ] Version `$0` does not already exist as a git tag
- [ ] Version `$0` is valid SemVer and consistent with release type `$1`
- [ ] Previous version tag exists for changelog diff
- [ ] `cargo audit` has no critical vulnerabilities
- [ ] MSRV is still 1.83 in root `Cargo.toml`

If any check fails, report it and ASK the user how to proceed.

---

### Phase 2: Changelog Generation

**Source:** all commits between `PREVIOUS_TAG` and `HEAD`.

```bash
git log ${PREVIOUS_TAG}..HEAD --oneline --no-merges
```

**Rules:**
- Group by: **Added** (feat:), **Fixed** (fix: that affect users), **Breaking Changes** (break:, BREAKING CHANGE)
- ONLY include user-facing changes. Skip: refactor, ci, chore, docs-only, test-only
- Write human-readable descriptions, not commit messages
- Do NOT attribute any change to AI tools
- If `$1` is `major`, the Breaking Changes section is mandatory

**Output:** draft the changelog entry and SHOW IT TO THE USER for approval
before writing to `CHANGELOG.md`.

---

### Phase 3: Version Bump

Update version string `$0` in ALL of these files:

| File | Field |
|------|-------|
| `Cargo.toml` (workspace root) | `workspace.package.version` |
| Each `crates/*/Cargo.toml` | `package.version` (if not inherited) |
| `crates/velesdb-wasm/package.json` | `version` |
| `sdk/typescript/package.json` | `version` (if exists) |
| Any `pyproject.toml` | `version` |

After updating, run:
```bash
cargo check --workspace
```

Report which files were updated. ASK the user to confirm before continuing.

---

### Phase 4: Documentation Audit

**4a. Scan every `.md` file in the entire repository:**
```
Glob: **/*.md
```

For each file:
1. Check for version references to the OLD version → update to `$0`
2. Check for `Last updated:` date → update to today
3. Check for AI attribution (Claude, OpenAI, Codex, Copilot, "AI-generated") → REMOVE
4. Check for stale metrics/benchmarks (no date, or date > 30 days) → flag for re-run

**4b. Metrics refresh:**
- Benchmarks in `benchmarks/` or `docs/` → run `cargo bench` if possible
- Performance comparisons → re-run `scripts/compare_perf.py` if baseline exists
- If a metric cannot be refreshed (needs GPU, mobile, browser), ASK the user:
  "Section X in file Y has metrics from DATE. I cannot re-run this locally.
  Keep as-is, update the date only, or remove the section?"

**4c. Feature lists and comparisons:**
- Verify feature matrices match actual crate features in `Cargo.toml`
- Verify API endpoint lists match actual routes in `velesdb-server`
- Verify VelesQL syntax docs match the pest grammar

**4d. Report** all changes made and any files that need user decision.

**4e. Feature inventory cross-check:**

Run the automated feature claims audit:
```bash
python3 scripts/check-feature-claims.py
```

This script introspects public API exports of each crate/SDK and cross-references
with documentation claims. It checks:

- **Per-crate capabilities**: public exports vs README feature lists for velesdb-core,
  velesdb-python, velesdb-wasm, TypeScript SDK, LangChain/LlamaIndex integrations
- **SDK parity**: features claimed in SDK docs must actually be exported
- **Roadmap vs delivered**: items marked "in progress" in `.epics/` must not be
  presented as delivered in README
- **Feature gaps**: capabilities exported but not documented (missed marketing)
- **False claims**: capabilities documented but not exported (misleading)

If the script reports gaps:
1. For MISSING (documented but not exported): either implement the feature or remove
   the claim from docs. ASK the user which approach.
2. For UNDOC (exported but not documented): add documentation for the feature.
3. For ROADMAP misrepresentations: clarify the status in README (e.g., "coming soon"
   vs presented as available).

Also run the performance claims audit:
```bash
python3 scripts/check-perf-claims.py
```

This ensures all performance numbers in docs are consistent across files and
match actual benchmark results (within 20% tolerance).

---

### Phase 5: Examples and Demos Validation

**5a. Scan all example/demo directories:**
```
Glob: examples/**/*
Glob: demos/**/*
Glob: **/examples/**/*
```

**5b. For each code example:**
- Verify version references are updated to `$0`
- Verify `Cargo.toml` dependencies point to `$0`
- Run `cargo check` on Rust examples (skip WASM/mobile if not buildable locally)
- Run `python -c "import ..."` on Python examples if possible

**5c. For each documentation snippet:**
- Verify code blocks in `.md` files use current API signatures
- Cross-reference with doctests (they should already pass from Phase 6)

Report which examples were validated and which were skipped (with reason).

---

### Phase 5b: Benchmark and Metrics Refresh

**Purpose:** ensure all published performance data is current and reproducible.

**5b-1. Run smoke benchmarks locally:**
```bash
cargo bench -p velesdb-core --bench smoke_test -- --noplot
python3 scripts/export_smoke_criterion.py
python3 scripts/compare_perf.py \
  --current benchmarks/results/latest.json \
  --baseline benchmarks/baseline.json \
  --threshold 15
```

**5b-2. Verify threshold consistency:**
- `benchmarks/baseline.json` → `threshold_percent` must match
- `.github/workflows/ci.yml` → `--threshold` must match
- `docs/reference/PERFORMANCE_SLO.md` → documented threshold must match
- ALL THREE must agree. If they don't, fix the discrepancy.

**5b-3. Check benchmark docs freshness:**
- Read `docs/BENCHMARKS.md` — if metrics are older than 30 days, re-run:
  ```bash
  cargo bench -p velesdb-core --bench simd_benchmark -- --noplot
  ```
- Read `docs/reference/PERFORMANCE_SLO.md` — update baseline version to `$0`
  if new baselines were recorded
- Read `benchmarks/baseline.json` — update `version` field if re-baselined
- Run `python3 scripts/benchmark_visualization.py` to regenerate charts if data changed
- Run `python3 scripts/generate-performance-proofpack.py` if SLO data changed

**5b-4. If benchmarks cannot run** (missing hardware, OS-specific):
ASK the user: "Cannot run benchmark X locally. Options:
  a) Keep existing metrics with current date
  b) Remove metrics section until CI re-runs
  c) Skip and note in release"

**5b-5. Update `benchmarks/machine-config.json`** if running on different hardware.

---

### Phase 6: Local CI Validation

Run the full local CI suite to catch issues before pushing:

```bash
# Format
cargo fmt --all -- --check

# Lint (strict, mirrors CI)
cargo clippy --workspace --all-targets --features persistence,gpu,update-check \
  --exclude velesdb-python -- -D warnings -D clippy::pedantic

# Doctests
cargo test --doc --package velesdb-core

# Full test suite
cargo test --workspace --features persistence,gpu,update-check \
  --exclude velesdb-python -- --test-threads=1

# Security
cargo audit

# WASM build check (no persistence)
cargo check -p velesdb-wasm --no-default-features --target wasm32-unknown-unknown 2>/dev/null || echo "WASM check skipped (target not installed)"

# Feature claims audit
python3 scripts/check-feature-claims.py

# Performance claims audit
python3 scripts/check-perf-claims.py
```

**Platform-specific checks to NOTE but not block on:**
- Mobile builds (iOS/Android) — cannot run on Windows/Linux desktop
- Browser WASM runtime tests — needs wasm-pack + browser
- GPU tests — needs wgpu-compatible hardware
- Python wheel build — needs maturin + Python environment

Report: passed checks, failed checks, skipped checks (with reason).
If any check FAILS, stop and ASK the user before continuing.

---

### Phase 7: Commit and Push (WITHOUT tag)

**Only after user approval of all previous phases.**

**IMPORTANT:** Do NOT create the tag yet. Push the release commit first so that
the standard CI pipeline (`ci.yml`) validates everything on `main`. The tag is
created in Phase 9 only after CI is green.

```bash
# Stage all changes
git add -A

# Commit (NO AI co-author)
git commit -m "chore(release): v$0"

# Push the commit only — NO tag
git push origin main
```

ASK the user: "Release commit pushed to main. Waiting for CI to validate before tagging."

---

### Phase 8: Wait for CI Green

**Purpose:** ensure the release commit passes ALL CI checks before creating the
immutable version tag. This prevents orphan tags that require version re-roll.

```bash
# Monitor the CI run on main
gh run list --branch main --limit 5 --json name,status,conclusion,createdAt

# Wait for the CI Success gate
gh run watch $(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
```

**Checks that must pass:**
- Lint & Format
- Tests (full workspace)
- Security audit
- VelesQL Conformance
- Perf Smoke Test
- Contract and Doc Guards

**If CI fails:** fix the issue, amend or add a commit, push again, and wait for
the next CI run. Do NOT proceed to tagging until CI is fully green.

Report CI status and ASK the user: "CI is green. Ready to create tag v$0 and trigger publishing?"

---

### Phase 9: Tag and Publish

**Only after CI is green on the release commit.**

```bash
# Create annotated tag on the validated commit
git tag -a "v$0" -m "Release v$0"

# Push the tag — this triggers release.yml
git push origin "v$0"
```

**9a. Monitor publishing workflows:**
```bash
gh run list --branch "v$0" --limit 5
gh run watch $(gh run list --branch "v$0" --limit 1 --json databaseId --jq '.[0].databaseId')
```

**9b. Verify each registry:**

| Registry | Verification command |
|----------|---------------------|
| crates.io | `cargo search velesdb-core --limit 1` |
| PyPI | `pip index versions velesdb 2>/dev/null` or check pypi.org |
| npm | `npm view @velesdb/wasm version 2>/dev/null` |
| GitHub Releases | `gh release view "v$0"` |

**9c. If any publishing fails**, report the error and ASK the user for next steps.
Since the tag is on a CI-validated commit, publishing failures are registry-side
issues (credentials, rate limits) — not code problems.

---

### Phase 10: Post-Release Summary

Generate a final summary:

```
## Release v$0 Summary

**Type:** $1
**Date:** YYYY-MM-DD
**Previous version:** PREVIOUS_TAG

### Changelog
[paste approved changelog]

### Files Modified
[list all files changed]

### Documentation Updated
[count of .md files audited / updated]

### Metrics Refreshed
[list benchmarks re-run, or "none needed"]

### Publishing Status
- crates.io: OK/FAILED
- PyPI: OK/FAILED/SKIPPED
- npm: OK/FAILED/SKIPPED
- GitHub Releases: OK/FAILED

### Skipped Checks
[list platform-specific checks that were skipped]
```

---

## Error Handling

- If ANY phase fails, STOP and report to the user. Do not proceed automatically.
- If a file cannot be updated (permissions, format), report and ask for guidance.
- If metrics cannot be refreshed, always ask — never silently skip or keep stale data.
- If publishing fails, do NOT retry automatically — report the error and wait.

## Related Files

- `CLAUDE.md` — Project build commands and architecture
- `docs/contributing/TDD_RULES.md` — Quality gates (>80% coverage)
- `benchmarks/baseline.json` — Performance regression baseline
- `scripts/compare_perf.py` — Benchmark comparison tool
- `scripts/export_smoke_criterion.py` — Criterion export tool
- `.github/workflows/` — CI and release workflows
