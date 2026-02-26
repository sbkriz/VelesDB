# Expert review Rust/Craftsman — `velesdb-core` + `VelesQL`

Date: 2026-02-26  
Scope: `crates/velesdb-core` (core runtime + `velesql` parser/cache/execution paths)

## Executive summary

I found **4 priority risks** that can create production incidents:

1. **Query cache correctness/safety risk**: cache key uses only a 64-bit hash, so a collision can return the wrong AST for another query.
2. **LRU implementation drift**: order queue is not updated on cache hit and can accumulate duplicates, causing non-LRU behavior and extra evictions.
3. **Potential DoS via LIKE/ILIKE**: current matcher allocates `O(text_len * pattern_len)` boolean matrix, unbounded by guardrails.
4. **Potential panic/DoS in mmap retrieval**: corrupted/untrusted index offset triggers `assert!` and can crash process.

I also list one medium-risk hardening topic (parser recursion depth).

---

## Detailed findings

## 1) Cache collision can return wrong parsed query (correctness + security)

### Evidence
- `QueryCache` stores entries as `FxHashMap<u64, Query>` where key is only `hash_query(query)` (no original SQL string check).  
- On hit, `cache.get(&hash)` returns cached query directly.

### Why this is dangerous
A 64-bit hash collision (accidental or crafted) means query A can receive AST of query B. If the cache is used by shared clients, this is a **query confusion** bug and can become a security boundary issue.

### Impact
- Wrong filtering/authorization logic at query layer.
- Hard-to-reproduce correctness bugs under load.

### Recommended fix
- Store `(original_query, parsed_query)` as value and validate full-string equality on hit.
- Optionally use a keyed hasher (`ahash`/SipHash keyed) to reduce chosen-collision risk.

---

## 2) LRU behavior is not true LRU and can over-evict

### Evidence
- On cache hit, code increments stats and returns without moving key to MRU position.
- On insert, code always `push_back(hash)` and never de-duplicates queue.

### Why this is dangerous
- Frequently used queries can still be evicted as “old”.
- Duplicate hash entries in deque can trigger redundant pop/remove cycles and higher lock contention under churn.

### Impact
- Lower hit-rate than expected.
- Latency regressions due to repeated parsing.

### Recommended fix
- Maintain explicit node/index for O(1) move-to-back on hit.
- Prevent duplicate queue entries (remove old position before push).
- Add invariant checks in tests: `order.len() == cache.len()` and no duplicates.

---

## 3) LIKE/ILIKE matcher can become memory/CPU amplification vector

### Evidence
- `like_match_impl` allocates a DP matrix: `vec![vec![false; n + 1]; m + 1]`.
- Complexity is O(m*n) memory and CPU.

### Why this is dangerous
For long payload strings and patterns, this can consume large memory and CPU (DoS vector), especially in multi-tenant/server mode.

### Impact
- Request-level latency spikes.
- Potential OOM for worst-case inputs.

### Recommended fix
- Replace full matrix with rolling 1D DP (`O(min(m,n))` memory).
- Enforce guardrails: max pattern length and/or max `m*n` budget.
- Add benchmark + adversarial tests for pathological patterns.

---

## 4) `retrieve_ref` uses `assert!` for data validation, can panic process

### Evidence
- `retrieve_ref` validates alignment with `assert!(offset % align_of::<f32>() == 0, ...)` before pointer cast.

### Why this is dangerous
If on-disk index is corrupted (or maliciously tampered), an assertion panic can terminate the process, turning data corruption into service outage.

### Impact
- Crash/DoS on read path.
- Lower resilience for crash-recovery scenarios.

### Recommended fix
- Replace `assert!` with fallible error (`io::ErrorKind::InvalidData`) and quarantining logic.
- Add corruption test that verifies graceful error, no panic.

---

## 5) Medium risk: parser recursion depth is unbounded

### Evidence
- `parse_primary_expr` recursively calls `parse_or_expr` / `parse_primary_expr` for nested groups/NOT.
- No depth budget/guard.

### Why this is dangerous
Very deeply nested expressions can trigger stack overflow (DoS) and unpredictable failures.

### Recommended fix
- Thread a depth counter with max threshold (configurable), return parse error when exceeded.

---

## Action plan (prioritized)

## P0 (this sprint)
1. **Cache correctness hardening**
   - Store full query text with parsed AST in cache value.
   - Verify equality on hit; treat hash collision as miss.
2. **Replace panic path in `retrieve_ref`**
   - Convert alignment assert to recoverable error.
3. **LIKE guardrails**
   - Add max pattern/input budget and reject over-limit conditions.

## P1
4. **True LRU implementation**
   - Move-to-back on hit, deduplicate order structure.
5. **Parser depth limits**
   - Add recursion depth guard in parser condition handling.

## P2
6. **Security/perf regression suite**
   - Add fuzz corpus and adversarial perf tests for parser + LIKE + cache collisions.

---

## Acceptance criteria (complets)

### AC-1 Cache correctness
- Given two different queries with forced same hash bucket/path, cache never returns wrong AST.
- Unit/integration tests cover collision scenario explicitly.
- No regression in cache hit-rate benchmark beyond agreed tolerance (e.g. <= 2%).

### AC-2 LRU invariants
- On hit, key becomes MRU.
- Internal order structure has no duplicates.
- Invariant tests pass under concurrent parse workload.

### AC-3 LIKE resilience
- Memory usage scales linearly with input length (no 2D matrix allocation).
- Requests exceeding configured budget fail fast with explicit error.
- Adversarial test corpus (long wildcard patterns) passes within latency envelope.

### AC-4 Storage robustness
- Corrupted/misaligned offset returns structured `InvalidData` error, no panic.
- Crash-recovery tests include tampered index case.

### AC-5 Parser hardening
- Deep nesting above limit returns deterministic parse error.
- Fuzz tests show no stack overflow/panic across defined corpus/time budget.

### AC-6 Observability
- Metrics added for: cache collision fallback count, LIKE guardrail rejections, parser depth-limit rejections, invalid offset read errors.
- Dashboards/alerts documented.

---

## Next priorities

1. **Ship P0 hardening patch first** (cache correctness + panic removal + LIKE guardrails).
2. **Run a focused perf baseline** before/after P0 and P1.
3. **Add security review gate** in CI (fuzz smoke + adversarial tests).
4. **Document operational limits** (max query complexity/pattern size) in public docs.
