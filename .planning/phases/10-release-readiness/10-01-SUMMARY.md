---
phase: 10-release-readiness
plan: 01
subsystem: release
tags: [version-bump, crates-io, semver, publish-guard]

# Dependency graph
requires:
  - phase: 09-documentation
    provides: "All v1.5 features documented and benchmarked"
provides:
  - "All workspace crates at version 1.5.0"
  - "publish = false guards on velesdb-python and velesdb-wasm"
  - "Cargo publish dry-run validated for velesdb-core"
affects: [10-release-readiness]

# Tech tracking
tech-stack:
  added: []
  patterns: ["workspace version inheritance via version.workspace = true"]

key-files:
  created: []
  modified:
    - Cargo.toml
    - Cargo.lock
    - crates/velesdb-python/Cargo.toml
    - crates/velesdb-wasm/Cargo.toml
    - crates/velesdb-python/pyproject.toml
    - sdks/typescript/package.json
    - crates/tauri-plugin-velesdb/guest-js/package.json
    - integrations/langchain/pyproject.toml
    - integrations/llamaindex/pyproject.toml
    - demos/rag-pdf-demo/pyproject.toml

key-decisions:
  - "Inter-crate deps use workspace = true inheritance, no explicit version in crate Cargo.toml files"
  - "Downstream crate dry-run fails expected: velesdb-core 1.5.0 not yet on crates.io, resolves during ordered publish"
  - "velesdb-core dry-run fully validates (package + compile), confirming publish readiness"

patterns-established:
  - "publish = false in Cargo.toml for crates distributed via PyPI/npm instead of crates.io"

requirements-completed: [REL-01, REL-05]

# Metrics
duration: 3min
completed: 2026-03-08
---

# Phase 10 Plan 01: Version Bump Summary

**Atomic version bump 1.4.5 to 1.5.0 across 10 files with publish guards and cargo publish dry-run validation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T23:13:01Z
- **Completed:** 2026-03-07T23:16:02Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Bumped all 14+ version locations from 1.4.5 to 1.5.0 (Cargo workspace, TypeScript, Python, integrations, demos)
- Added publish = false to velesdb-python and velesdb-wasm to prevent accidental crates.io publish
- Validated velesdb-core packages, compiles, and passes cargo publish --dry-run
- Confirmed all 5 downstream crates correctly depend on velesdb-core 1.5.0 via workspace inheritance

## Task Commits

Each task was committed atomically:

1. **Task 1: Atomic version bump 1.4.5 to 1.5.0 + publish guards** - `083e1d6f` (chore)
2. **Task 2: Cargo publish dry-run validation** - verification only, no file changes

## Files Created/Modified
- `Cargo.toml` - Workspace version 1.4.5 -> 1.5.0, velesdb-core dep 1.5.0
- `Cargo.lock` - Updated lock file for new versions
- `crates/velesdb-python/Cargo.toml` - Added publish = false
- `crates/velesdb-wasm/Cargo.toml` - Added publish = false
- `crates/velesdb-python/pyproject.toml` - Version 1.5.0
- `sdks/typescript/package.json` - Version 1.5.0
- `crates/tauri-plugin-velesdb/guest-js/package.json` - Version 1.5.0
- `integrations/langchain/pyproject.toml` - Version 1.5.0
- `integrations/llamaindex/pyproject.toml` - Version 1.5.0
- `demos/rag-pdf-demo/pyproject.toml` - Version 1.5.0

## Decisions Made
- Inter-crate dependencies use `workspace = true` inheritance rather than explicit path + version. The bump-version.ps1 script patterns for explicit deps don't match, but this is correct -- workspace inheritance handles versioning automatically.
- Downstream crate dry-runs failing with "velesdb-core 1.5.0 not found on crates.io" is expected behavior -- the release workflow publishes in dependency order (core first).
- velesdb-core dry-run fully succeeds (packaging + compilation), confirming all metadata and file inclusions are correct.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed bump-version.ps1 emoji encoding issue**
- **Found during:** Task 1 (version bump)
- **Issue:** PowerShell parser failed on emoji characters in the script under MSYS2/Git Bash
- **Fix:** Created emoji-free temp copy to run the script successfully
- **Files modified:** None (temp file cleaned up)
- **Verification:** Script ran and bumped 9 files correctly

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor environment workaround, no scope change.

## Issues Encountered
- bump-version.ps1 inter-crate dependency patterns assume explicit `path + version` format, but crates use `workspace = true`. Not a real issue -- workspace inheritance is the correct pattern.
- WASM pkg/package.json was updated by the script (it exists on disk from a prior wasm-pack build). This is harmless.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All versions at 1.5.0, ready for git tag and release workflow
- velesdb-core validated for crates.io publishing
- publish guards in place for PyPI/npm-only crates

---
## Self-Check: PASSED

All files exist, commit hash verified, version 1.5.0 confirmed, publish guards in place.

---
*Phase: 10-release-readiness*
*Completed: 2026-03-08*
