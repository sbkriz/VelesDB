# Migration Guide: VelesDB v1.4 to v1.5

**Last Updated**: 2026-03-07

This guide covers all breaking changes when upgrading from VelesDB v1.4.x to v1.5.0, along with migration steps and troubleshooting tips.

---

## Overview

VelesDB v1.5 introduces Product Quantization, sparse vector search, hybrid search with fusion strategies, streaming ingestion, and query plan caching. Several of these features require changes to the on-disk format, public API types, and query grammar.

**Breaking changes at a glance:**

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 1 | Wire format: bincode to postcard | Data re-import required | High |
| 2 | `QuantizationConfig` extended with PQ variant | Exhaustive match arms | Low |
| 3 | VelesQL grammar extensions | Parser updates if custom | Medium |
| 4 | `Point` struct new field | All Point literals | Low |
| 5 | Dependency changes | Cargo.toml updates | Minimal |
| 6 | REST API additions | Client SDK updates | Low |

---

## Breaking Change 1: On-Disk Wire Format (bincode to postcard)

### What Changed

The serialization library for on-disk persistence was migrated from `bincode` to `postcard` (RUSTSEC-2025-0141 advisory on bincode 1.3).

### Impact

**Existing persisted collection data on disk is NOT backward-compatible.** Collections created with v1.4 cannot be opened directly by v1.5.

### Migration Steps

1. **Export your data before upgrading.** Use the REST API or Rust SDK to read all points from each collection:

   ```bash
   # REST API: paginate through all points
   curl http://localhost:8080/collections/my_collection/points?limit=1000&offset=0
   ```

   ```python
   # Python SDK
   import velesdb
   db = velesdb.Database("./data")
   coll = db.get_collection("my_collection")
   # get_all() does not exist; use coll.get(ids) with specific IDs
   # or iterate using available query methods
   # Save to JSON/Parquet/etc.
   ```

2. **Upgrade VelesDB** to v1.5.

3. **Re-create collections** with the same configuration:

   ```python
   db.create_collection("my_collection", dimension=768, metric="cosine")
   ```

4. **Re-insert data** from your export:

   ```python
   db.get_collection("my_collection").upsert(points)
   ```

### Notes

- `bincode` remains as a transitive dependency in `velesdb-mobile` via `uniffi`. This is acknowledged in `deny.toml` and does not affect `velesdb-core` serialization.
- The `postcard` format is more compact and avoids the unsoundness issues in bincode 1.3.

---

## Breaking Change 2: QuantizationConfig Extended

### What Changed

The `QuantizationConfig` enum now includes a `ProductQuantization` variant alongside the existing `SQ8` and `Binary` variants. A new `QuantizationType` enum was also introduced (named to avoid collision with the VelesQL AST `QuantizationMode`).

### Impact

Code that exhaustively matches on `QuantizationConfig` must handle the new variant:

```rust
// v1.4
match config {
    QuantizationConfig::Scalar(sq8) => { /* ... */ }
    QuantizationConfig::Binary(bin) => { /* ... */ }
}

// v1.5 -- add the new arm
match config {
    QuantizationConfig::Scalar(sq8) => { /* ... */ }
    QuantizationConfig::Binary(bin) => { /* ... */ }
    QuantizationConfig::ProductQuantization(pq) => { /* ... */ }
}
```

### Backward Compatibility

A custom `Deserialize` implementation supports both the old string format and the new tagged object format. Existing serialized configurations will deserialize correctly.

### PQ Defaults

| Parameter | Default |
|-----------|---------|
| `k` (codebook size) | 256 |
| `rescore_oversampling` | `Some(4)` |
| `opq_enabled` | `false` |

---

## Breaking Change 3: VelesQL Grammar Extensions

### What Changed

Three new grammar constructs were added to VelesQL:

#### SPARSE_NEAR Clause

Search using sparse vectors (SPLADE, BM42, or custom sparse embeddings):

```sql
SELECT * FROM docs WHERE vector SPARSE_NEAR $sv LIMIT 10
```

The `SPARSE_NEAR` clause accepts a `SparseVectorExpr` which has two variants:
- **Literal**: inline sparse vector data
- **Parameter**: a `$name` reference resolved at runtime

#### FUSE BY Clause -- PLANNED SYNTAX

> **Note:** `FUSE BY` is planned syntax (not yet implemented in the v1.5 grammar). Use `USING FUSION(...)` for hybrid fusion queries today.

Combine dense and sparse search results with a fusion strategy:

```sql
-- PLANNED: syntax not yet implemented
-- Reciprocal Rank Fusion (default k=60)
SELECT * FROM docs
WHERE vector NEAR $dense AND vector SPARSE_NEAR $sparse
FUSE BY RRF(k=60)  -- PLANNED: not yet implemented
LIMIT 10

-- PLANNED: syntax not yet implemented
-- Reciprocal Score Fusion with weights
SELECT * FROM docs
WHERE vector NEAR $dense AND vector SPARSE_NEAR $sparse
FUSE BY RSF(dense_weight=0.7, sparse_weight=0.3)  -- PLANNED: not yet implemented
LIMIT 10
```

**Current working syntax (v1.5):**

```sql
-- Use USING FUSION(...) instead of FUSE BY:
SELECT * FROM docs
WHERE vector NEAR $dense AND vector SPARSE_NEAR $sparse
USING FUSION(strategy = 'rrf', k = 60)
LIMIT 10
```

#### TRAIN QUANTIZER Command

Explicitly train a Product Quantizer on a collection:

```sql
TRAIN QUANTIZER ON my_collection WITH (m=8, k=256)
```

Training is explicit and not automatic. The quantizer must be trained before PQ-compressed search is available.

### Impact

If you maintain a custom VelesQL parser (e.g., in a third-party SDK), it must be updated to handle `SPARSE_NEAR` and `TRAIN QUANTIZER`. (`FUSE BY` is planned syntax -- the current working equivalent is `USING FUSION(...)`.)

---

## Breaking Change 4: Point Struct Changes

### What Changed

The `Point` struct now includes a `sparse_vector` field:

```rust
pub struct Point {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Option<Payload>,
    pub sparse_vector: Option<BTreeMap<String, SparseVector>>,  // NEW
}
```

### Impact

All code constructing `Point` literals must include the new field:

```rust
// v1.4
let point = Point {
    id: 1,
    vector: vec![0.1, 0.2, 0.3],
    payload: None,
};

// v1.5
let point = Point {
    id: 1,
    vector: vec![0.1, 0.2, 0.3],
    payload: None,
    sparse_vector: None,  // Add this
};
```

### Backward Compatibility

A custom `Deserialize` implementation handles the old format: if `sparse_vector` is absent in serialized data, it defaults to `None`. If it contains a bare `SparseVector` (old format), it is automatically wrapped into a `BTreeMap` with a default key.

---

## Breaking Change 5: New Dependencies

### What Changed

| Dependency | Change | Reason |
|-----------|--------|--------|
| `postcard` | Added (replaces `bincode`) | Serialization (RUSTSEC-2025-0141) |
| `rand 0.8` | Promoted from dev to production | k-means++ initialization for PQ training |

### Impact

- If you pin dependencies via `Cargo.lock`, update it after upgrading.
- `rand 0.8` is now a required production dependency. This should not conflict with most crate ecosystems but may affect WASM targets if `rand` feature flags are not configured.

---

## Breaking Change 6: REST API Changes

### New Endpoints and Fields

| Endpoint / Field | Type | Description |
|-----------------|------|-------------|
| `sparse_vector` in upsert body | New field | Attach sparse vectors to points |
| `POST /collections/{name}/search/sparse` | New endpoint | Sparse-only search |
| `POST /collections/{name}/stream/insert` | New endpoint | Streaming batch ingestion |
| `fusion_strategy` in hybrid search | New field | RRF (default) or RSF fusion |

### Upsert Request Body (v1.5)

```json
{
  "points": [
    {
      "id": 1,
      "vector": [0.1, 0.2, 0.3],
      "payload": {"title": "Example"},
      "sparse_vector": {
        "default": {"indices": [0, 5, 12], "values": [0.8, 0.3, 0.5]}
      }
    }
  ]
}
```

The `sparse_vector` field is optional. Omitting it is equivalent to v1.4 behavior.

### Hybrid Search Default

When both dense and sparse vectors are provided in a search request, RRF with `k=60` is used as the default fusion strategy.

---

## Migration Checklist

- [ ] Export all collection data from v1.4 instance
- [ ] Update `velesdb-core` (and SDK dependencies) to v1.5.0
- [ ] Re-create collections and re-import data
- [ ] Update exhaustive `QuantizationConfig` match arms to include `ProductQuantization`
- [ ] Add `sparse_vector: None` to all `Point` struct literals
- [ ] Update any custom VelesQL parsers for `SPARSE_NEAR`, `TRAIN QUANTIZER` (note: `FUSE BY` is planned -- use `USING FUSION(...)` for now)
- [ ] Update REST API client code to handle new fields if applicable
- [ ] Run test suite to verify no regressions

---

## FAQ / Troubleshooting

### Q: Can I open v1.4 data files with v1.5?

**A:** No. The on-disk format changed from bincode to postcard. You must export, upgrade, and re-import. There is no automatic migration tool.

### Q: Will my v1.4 QuantizationConfig JSON still deserialize?

**A:** Yes. The custom `Deserialize` implementation supports both the old string format (`"scalar"`) and the new tagged object format (`{"type": "product_quantization", ...}`).

### Q: Do I need sparse vectors?

**A:** No. Sparse vectors are entirely optional. If you do not use `sparse_vector` in your points or queries, v1.5 behaves identically to v1.4 for dense search.

### Q: What if my VelesQL queries do not use the new syntax?

**A:** Existing v1.4 VelesQL queries continue to work without modification. The new syntax (`SPARSE_NEAR`, `TRAIN QUANTIZER`) is additive. (`FUSE BY` is planned for a future release -- use `USING FUSION(...)` for hybrid fusion.)

### Q: Is the REST API backward compatible?

**A:** Yes for existing endpoints. The `sparse_vector` field in upsert is optional, and existing search endpoints continue to work. New endpoints (`/search/sparse`, `/stream/insert`) are additive.

---

*VelesDB v1.5.0 -- March 2026*
